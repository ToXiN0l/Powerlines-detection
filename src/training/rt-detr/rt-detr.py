import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import RTDETR
from src.training.yolo.yolo26m import yaml_path

model = RTDETR("rtdetr-x.pt")
model.train(
    data=yaml_path, 
    epochs=60, 
    imgsz=640, 
    device="cuda",
    amp=False,
    lr0=0.0001,
    # batch=4
)
model.save("weights/rtdetr-v2-x.pt")

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
plt.savefig("metrics_seaborn.png")
plt.show()