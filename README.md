To train/save model: python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 

To run model on webcam: python detect_video.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --video 0 --output ./detections/results.avi

To run counter script: python detect_counter.py --weights ./checkpoints/yolov4-416 --video 0 

