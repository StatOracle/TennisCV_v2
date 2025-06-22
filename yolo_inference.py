from ultralytics import YOLO
model = YOLO('yolov8x')


model = YOLO("models/yolo5_last.pt")
print(model.names)
