import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.config import cfg
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt)')
flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/video.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.80, 'minimum confidence for an object to be counted (0.80 = 80%)')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_enum('count_mode', 'unique', ['unique', 'frame'],
                  "'unique': count each physical object once (tracked across frames); "
                  "'frame': count every detection in every frame")
flags.DEFINE_integer('min_hits', 3, 'consecutive-ish frames an object must be seen before it is counted (unique mode)')
flags.DEFINE_integer('max_missed', 15, 'frames an object can disappear before its track is dropped (unique mode)')
flags.DEFINE_float('max_dist', 0.10, 'max centroid movement between frames, as a fraction of the frame diagonal (unique mode)')
flags.DEFINE_list('count_classes', None, 'optional list of class names to count, e.g. person,car (default: all)')
flags.DEFINE_list('ignore_classes', ['person'], 'class names to never detect or count (pass --ignore_classes= for none)')


class ObjectCounter:
    """Running counter.

    unique mode: a simple centroid tracker gives each object an ID. The counter goes up by one
                 the first time a track has been seen `min_hits` times, so an object sitting in
                 view for 500 frames is still counted once. When a counted object has been gone
                 for more than `max_missed` frames, the counter goes back down by one.
    frame mode:  the counter goes up by one for every qualifying detection in every frame.
    """

    def __init__(self, mode, min_hits, max_missed, max_dist_px):
        self.mode = mode
        self.min_hits = min_hits
        self.max_missed = max_missed
        self.max_dist_px = max_dist_px
        self.total = 0          # objects currently in frame (goes up and down)
        self.seen_per_class = {}  # every object counted since start, per class (never decreases)
        self.per_class = {}
        self.tracks = {}
        self.next_id = 0

    def _count(self, cls_name):
        self.total += 1
        self.seen_per_class[cls_name] = self.seen_per_class.get(cls_name, 0) + 1
        self.per_class[cls_name] = self.per_class.get(cls_name, 0) + 1

    def _uncount(self, cls_name):
        self.total = max(0, self.total - 1)
        n = self.per_class.get(cls_name, 0) - 1
        if n > 0:
            self.per_class[cls_name] = n
        else:
            self.per_class.pop(cls_name, None)

    def update(self, detections):
        """detections: list of (cx, cy, class_name) in pixel coordinates."""
        if self.mode == 'frame':
            for _, _, cls_name in detections:
                self._count(cls_name)
            return

        # build all (distance, track_id, det_index) pairs of the same class, match closest first
        pairs = []
        for tid, t in self.tracks.items():
            for di, (cx, cy, cls_name) in enumerate(detections):
                if cls_name != t['cls']:
                    continue
                d = np.hypot(cx - t['pos'][0], cy - t['pos'][1])
                if d <= self.max_dist_px:
                    pairs.append((d, tid, di))
        pairs.sort()

        matched_tracks, matched_dets = set(), set()
        for d, tid, di in pairs:
            if tid in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(tid)
            matched_dets.add(di)
            t = self.tracks[tid]
            t['pos'] = detections[di][:2]
            t['hits'] += 1
            t['missed'] = 0

        # unmatched detections start new tracks
        new_ids = set()
        for di, (cx, cy, cls_name) in enumerate(detections):
            if di not in matched_dets:
                self.tracks[self.next_id] = {'pos': (cx, cy), 'cls': cls_name,
                                             'hits': 1, 'missed': 0, 'counted': False}
                new_ids.add(self.next_id)
                self.next_id += 1

        # age out tracks that were not seen this frame
        for tid, t in list(self.tracks.items()):
            if tid not in matched_tracks and tid not in new_ids:
                t['missed'] += 1
                if t['missed'] > self.max_missed:
                    if t['counted']:
                        self._uncount(t['cls'])
                    del self.tracks[tid]

        # count confirmed tracks once
        for t in self.tracks.values():
            if not t['counted'] and t['hits'] >= self.min_hits:
                t['counted'] = True
                self._count(t['cls'])


def draw_counter(frame_bgr, counter):
    lines = ['COUNT: %d' % counter.total]
    for name, n in sorted(counter.per_class.items(), key=lambda x: -x[1])[:6]:
        lines.append('%s: %d' % (name, n))
    h = 14 + 26 * len(lines)
    w = 260
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (8, 8), (8 + w, 8 + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0, frame_bgr)
    for i, text in enumerate(lines):
        scale, thick = (0.9, 2) if i == 0 else (0.6, 1)
        cv2.putText(frame_bgr, text, (18, 36 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 255, 0) if i == 0 else (255, 255, 255), thick, cv2.LINE_AA)
    return frame_bgr


def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)
    all_names = set(class_names.values())
    allowed = set(FLAGS.count_classes) if FLAGS.count_classes else all_names
    allowed = allowed - set(FLAGS.ignore_classes or [])

    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    try:
        vid = cv2.VideoCapture(int(video_path))
    except ValueError:
        vid = cv2.VideoCapture(video_path)

    out = None
    if FLAGS.output:
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    counter = None

    while True:
        return_value, frame = vid.read()
        if not return_value:
            print('Video has ended or failed, try a different video format!')
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame.shape[:2]
        if counter is None:
            diag = float(np.hypot(frame_w, frame_h))
            counter = ObjectCounter(FLAGS.count_mode, FLAGS.min_hits, FLAGS.max_missed, FLAGS.max_dist * diag)

        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=FLAGS.score,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=FLAGS.score,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        # score_threshold here is what enforces the 80% confidence cutoff
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        boxes_np, scores_np = boxes.numpy(), scores.numpy()
        classes_np, valid_np = classes.numpy(), valid_detections.numpy()
        pred_bbox = [boxes_np, scores_np, classes_np, valid_np]

        # collect qualifying detections as pixel-space centroids (boxes are normalized ymin,xmin,ymax,xmax)
        detections = []
        for i in range(valid_np[0]):
            if scores_np[0][i] < FLAGS.score:
                continue
            cls_name = class_names[int(classes_np[0][i])]
            if cls_name not in allowed:
                continue
            ymin, xmin, ymax, xmax = boxes_np[0][i]
            cx = (xmin + xmax) / 2.0 * frame_w
            cy = (ymin + ymax) / 2.0 * frame_h
            detections.append((cx, cy, cls_name))

        counter.update(detections)

        image = utils.draw_bbox(frame, pred_bbox, allowed_classes=list(allowed))
        result = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        result = draw_counter(result, counter)

        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f   COUNT: %d" % (fps, counter.total))

        if not FLAGS.dont_show:
            cv2.imshow("result", result)
        if out:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if counter is not None:
        final_counts = sorted(counter.seen_per_class.items(), key=lambda x: (-x[1], x[0]))
        print("\nFinal object counts (name, count):")
        print(final_counts)
        for name, n in final_counts:
            print("  %-20s %d" % (name, n))
    vid.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
