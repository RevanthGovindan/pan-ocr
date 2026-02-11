yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=80 imgsz=640 batch=8 device=mps

to extract:
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=test_images/ conf=0.5