import os
import json
import random
import shutil


BASE_DIR = "dataset/insulators"
JSON_PATH = os.path.join(BASE_DIR, "annotation_data.json")
IMG_DIR = os.path.join(BASE_DIR, "images")
OUT_DIR = "yolo_dataset"
split_ratio = 0.8


for split in ['train', 'val']:
    os.makedirs(os.path.join(OUT_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'labels', split), exist_ok=True)

with open(JSON_PATH, 'r', encoding='utf-8') as f:
    coco_data = json.load(f)

categories = {cat['id']: i for i, cat in enumerate(coco_data['categories'])}
category_names = [cat['name'] for cat in coco_data['categories']]

annotations_by_img = {}
for ann in coco_data['annotations']:
    img_id = ann['image_id']
    if img_id not in annotations_by_img:
        annotations_by_img[img_id] = []
    annotations_by_img[img_id].append(ann)

images = coco_data['images']
random.shuffle(images)
split_idx = int(len(images) * split_ratio)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

def process_split(img_list, split_name):
    for img in img_list:
        file_name = img['file_name']
        flat_name = file_name.replace('/', '_').replace('\\', '_')

        src_img_path = os.path.join(IMG_DIR, file_name)
        dst_img_path = os.path.join(OUT_DIR, 'images', split_name, flat_name)
        
        txt_name = os.path.splitext(flat_name)[0] + '.txt'
        txt_path = os.path.join(OUT_DIR, 'labels', split_name, txt_name)
        
        # Копируем картинку
        if os.path.exists(src_img_path):
            if not os.path.exists(dst_img_path):
                shutil.copy(src_img_path, dst_img_path)
        else:
            print(f"Предупреждение: Файл не найден: {src_img_path}")
            continue
        
        with open(txt_path, 'w') as txt_file:
            img_anns = annotations_by_img.get(img['id'], [])
            for ann in img_anns:
                x_min, y_min, w, h = ann['bbox']
                
                img_w, img_h = img['width'], img['height']
                x_center = (x_min + w / 2) / img_w
                y_center = (y_min + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                
                cat_id = categories[ann['category_id']]
                txt_file.write(f"{cat_id} {x_center} {y_center} {norm_w} {norm_h}\n")

process_split(train_imgs, 'train')
process_split(val_imgs, 'val')

yaml_path = os.path.join(OUT_DIR, 'dataset.yaml')
with open(yaml_path, 'w', encoding='utf-8') as f:
    f.write(f"path: {os.path.abspath(OUT_DIR)}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n\n")
    f.write(f"nc: {len(category_names)}\n")
    f.write(f"names: {category_names}\n")

print(yaml_path)