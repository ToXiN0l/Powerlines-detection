from ultralytics import YOLO
from src.training.yolo.yolo26m import YAML_PATH, save_results, download_dataset

download_dataset()

model = YOLO('yolo26x.pt')
model.train(data=YAML_PATH, epochs=60, imgsz=640, device="cuda")
model.save("weights/yolo26x.pt")

save_results(model)