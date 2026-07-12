from ultralytics import YOLO


WEIGHTS_PATH = "weights/yolo26x.pt"
VAL_PATH = "yolo_dataset/dataset.yaml"
model = YOLO(WEIGHTS_PATH)

metrics = model.val(
    data=VAL_PATH,
    imgsz=640,
    device=0
)

print(f"mAP50-95:  {metrics.box.map:.4f}")
print(f"mAP50:     {metrics.box.map50:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")