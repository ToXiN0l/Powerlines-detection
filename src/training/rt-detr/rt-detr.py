from ultralytics import RTDETR
from src.training.yolo.yolo26m import YAML_PATH, WEIGHTS_PATH, save_results, download_dataset

download_dataset()

model = RTDETR("rtdetr-x.pt")
model.train(
    data=YAML_PATH, 
    epochs=60, 
    imgsz=640, 
    device="cuda",
    amp=False,
    lr0=0.0001,
)
save_path = f"{WEIGHTS_PATH}/{model.overrides.get('model')}"
model.save(save_path)

save_results(model)