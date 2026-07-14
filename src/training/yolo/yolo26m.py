import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO
from huggingface_hub import snapshot_download

def save_results(model):
    model_name = os.path.basename(model.ckpt_path).replace('.pt', '')

    csv_path = os.path.join(model.trainer.save_dir, "results.csv")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    metrics = [col for col in df.columns if col != 'epoch']

    fig, axes = plt.subplots((len(metrics) + 1) // 2, 2, figsize=(15, 3 * len(metrics) // 2))
    axes = axes.flatten()
    sns.set_theme(style="whitegrid")

    for i, metric in enumerate(metrics):
        sns.lineplot(data=df, x='epoch', y=metric, ax=axes[i], marker="o")
        axes[i].set_title(metric)
        axes[i].set_ylabel("")

    plt.tight_layout()
    plt.savefig(f"metrics_{model_name}.png")
    plt.show()

def download_dataset(repo_name="ToXiN0/Powerlines-detection", local_name="yolo_dataset"):
    snapshot_download(
        repo_id=repo_name,
        repo_type="dataset",
        local_dir=local_name,
    )

YAML_PATH = 'yolo_dataset/dataset.yaml'

model = YOLO('yolo26m.pt')
model.train(data=YAML_PATH, epochs=40, imgsz=640, device="cuda")
model.save("weights/yolo26m.pt")