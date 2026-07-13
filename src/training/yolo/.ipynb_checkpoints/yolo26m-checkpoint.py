from ultralytics import YOLO


yaml_path = 'yolo_dataset/dataset.yaml'

model = YOLO('yolo26m.pt')
model.train(data=yaml_path, epochs=40, imgsz=640, device="cuda")
model.save("weights/yolo26m.pt")