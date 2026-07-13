import io
import time
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image

app = FastAPI(title="Powerlines Detection API")

MODEL_PATHS = {
    "light" : "weights/yolo26m.pt",
    "balanced" : "weights/yolo26x.pt",
    "quality" : "weights/rtdetr-x.pt",
}

models = {name: YOLO(path) for name, path in MODEL_PATHS.items()}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Form("light"),
    threshold: float = Form(0.25),
):
    if model not in models:
        model = "light"
    threshold = min(max(threshold, 0.0), 1.0)

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    start_time = time.perf_counter()
    results = models[model](image, conf=threshold)
    inference_time_ms = (time.perf_counter() - start_time) * 1000

    annotated_array = results[0].plot(font_size=0.2, line_width=1)
    annotated_image = Image.fromarray(annotated_array[..., ::-1])

    img_buffer = io.BytesIO()
    annotated_image.save(img_buffer, format="JPEG", quality=90)
    img_buffer.seek(0)

    detections_count = len(results[0].boxes)

    return StreamingResponse(
        img_buffer,
        media_type="image/jpeg",
        headers={
            "X-Detections-Count": str(detections_count),
            "X-Model-Used": model,
            "X-Inference-Time-Ms": f"{inference_time_ms:.1f}",
        },
    )

app.mount("/", StaticFiles(directory="static", html=True), name="static")