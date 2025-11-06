# prepare.py - FINAL 100% CORRECTED
import os
import shutil
import cv2
import glob
import xml.etree.ElementTree as ET
from pathlib import Path

# Create YOLO folders
for split in ['train', 'val']:
    os.makedirs(f'data/yolo/{split}/images', exist_ok=True)
    os.makedirs(f'data/yolo/{split}/labels', exist_ok=True)

# === 1. NEU: XML in annotations/, images in subfolders ===
neu_converted = 0
for split in ['train', 'validation']:
    img_base = f'data/NEU-DET/{split}/images'
    ann_base = f'data/NEU-DET/{split}/annotations'
    yolo_split = 'train' if split == 'train' else 'val'

    if not os.path.exists(img_base):
        print(f"Warning: {img_base} not found")
        continue
    if not os.path.exists(ann_base):
        print(f"Warning: {ann_base} not found")
        continue

    img_paths = glob.glob(f'{img_base}/**/*.jpg', recursive=True)
    print(f"Found {len(img_paths)} NEU images in {split}")

    for img_path in img_paths:
        # Get relative path: e.g., crazing/crazing_1.jpg
        rel_path = Path(img_path).relative_to(img_base)
        stem = rel_path.stem  # crazing_1
        xml_name = f"{stem}.xml"
        xml_path = os.path.join(ann_base, xml_name)

        if not os.path.exists(xml_path):
            continue  # No annotation

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []

        for obj in root.iter('object'):
            b = obj.find('bndbox')
            x1 = int(b.find('xmin').text)
            y1 = int(b.find('ymin').text)
            x2 = int(b.find('xmax').text)
            y2 = int(b.find('ymax').text)
            cx = (x1 + x2) / (2 * w)
            cy = (y1 + y2) / (2 * h)
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            boxes.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if boxes:
            base = Path(img_path).stem
            new_img = f'data/yolo/{yolo_split}/images/{base}.jpg'
            new_lbl = f'data/yolo/{yolo_split}/labels/{base}.txt'
            shutil.copy(img_path, new_img)
            with open(new_lbl, 'w') as f:
                f.write('\n'.join(boxes))
            neu_converted += 1

print(f"NEU: {neu_converted} images converted")

# === 2. DAGM: Use 'labels/' folder + '_labels.PNG' suffix ===
dagm_converted = 0
for typ in ['Train', 'Test']:
    split = 'train' if typ == 'Train' else 'val'
    img_dir = f'data/CompetitionData/Class1/{typ}/Good'
    lbl_dir = f'data/CompetitionData/Class1/{typ}/labels'  # ← 'labels/', not 'Label'

    if not os.path.exists(img_dir):
        print(f"Warning: {img_dir} not found")
        continue
    if not os.path.exists(lbl_dir):
        print(f"Warning: {lbl_dir} not found")
        continue

    img_paths = glob.glob(f'{img_dir}/*.PNG')  # Case-insensitive
    print(f"Found {len(img_paths)} DAGM images in {typ}/Good")

    for img_path in img_paths:
        base_name = Path(img_path).stem  # e.g., "0001"
        ext = Path(img_path).suffix.lower()  # ".png"
        lbl_name = f"{base_name}_labels{ext}"  # ← "_labels.PNG"
        lbl_path = os.path.join(lbl_dir, lbl_name)

        if not os.path.exists(lbl_path):
            # Try lowercase
            alt_name = f"{base_name}_labels.png"
            alt_path = os.path.join(lbl_dir, alt_name)
            if os.path.exists(alt_path):
                lbl_path = alt_path
            else:
                continue  # No label → skip

        img = cv2.imread(img_path)
        mask = cv2.imread(lbl_path, 0)
        if img is None or mask is None or mask.max() == 0:
            continue

        h, w = img.shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 10 or bh < 10:
                continue
            cx = (x + bw/2) / w
            cy = (y + bh/2) / h
            boxes.append(f"0 {cx:.6f} {cy:.6f} {bw/w:.6f} {bh/h:.6f}")

        if boxes:
            new_img = f'data/yolo/{split}/images/{base_name}{ext}'
            new_lbl = f'data/yolo/{split}/labels/{base_name}.txt'
            shutil.copy(img_path, new_img)
            with open(new_lbl, 'w') as f:
                f.write('\n'.join(boxes))
            dagm_converted += 1

print(f"DAGM: {dagm_converted} images converted")

