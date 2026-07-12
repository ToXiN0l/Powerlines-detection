from ultralytics import YOLO

import torch
print(torch.cuda.is_available())

yaml_path = 'yolo_dataset/dataset.yaml'

model = YOLO('yolo26x.pt')
model.train(data=yaml_path, epochs=40, imgsz=640, device="cuda")
model.save("weights/yolo26x.pt")