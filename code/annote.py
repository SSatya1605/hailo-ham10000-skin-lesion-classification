"""
HAM10000 → YOLOv8 Binary Dataset Converter
===========================================
Converts HAM10000 segmentation masks + GroundTruth.csv
into YOLOv8 detection format with 2 classes:
    0 = Benign
    1 = Malignant

Class mapping:
    Benign    → NV, BKL, DF, VASC
    Malignant → MEL, BCC, AKIEC

Usage:
    python ham10000_to_yolo_binary.py
    python ham10000_to_yolo_binary.py --data_dir /path/to/HAM10000 --output_dir /path/to/dataset
"""

import cv2
import numpy as np
import os
import pandas as pd
import shutil
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Binary class mapping ────────────────────────────────────────────────────
BENIGN    = 0
MALIGNANT = 1

CLASS_NAMES = {
    0: "Benign",
    1: "Malignant"
}

# Original 7 classes → binary label
MALIGNANT_CLASSES = {"MEL", "BCC", "AKIEC"}
BENIGN_CLASSES    = {"NV", "BKL", "DF", "VASC"}
ALL_COLUMNS       = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


# ── Parse arguments ─────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="HAM10000 → YOLOv8 Binary converter")
    parser.add_argument("--data_dir",   type=str, default="HAM10000",
                        help="Path to HAM10000 folder (default: HAM10000)")
    parser.add_argument("--output_dir", type=str, default="dataset_binary",
                        help="Output dataset folder (default: dataset_binary)")
    parser.add_argument("--val_split",  type=float, default=0.2,
                        help="Validation split ratio (default: 0.2)")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed (default: 42)")
    return parser.parse_args()


# ── Get binary class id from one-hot row ────────────────────────────────────
def get_binary_class(row):
    """
    Returns 0 (Benign) or 1 (Malignant) based on the
    one-hot encoded columns in GroundTruth.csv.
    """
    for col in ALL_COLUMNS:
        if float(row[col]) == 1.0:
            if col in MALIGNANT_CLASSES:
                return MALIGNANT
            else:
                return BENIGN
    # Fallback: pick highest value column
    values = {col: float(row[col]) for col in ALL_COLUMNS}
    top_class = max(values, key=values.get)
    return MALIGNANT if top_class in MALIGNANT_CLASSES else BENIGN


# ── Convert segmentation mask → YOLO bounding box ──────────────────────────
def mask_to_yolo_bbox(mask_path, image_w, image_h):
    """
    Reads a binary segmentation mask and returns normalized
    YOLO bounding box (cx, cy, w, h) for the largest contour.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Clamp to image bounds
    x = max(0, x)
    y = max(0, y)
    w = min(w, image_w - x)
    h = min(h, image_h - y)

    if w <= 0 or h <= 0:
        return None

    # Normalize — YOLO format
    cx = (x + w / 2) / image_w
    cy = (y + h / 2) / image_h
    nw = w / image_w
    nh = h / image_h

    return cx, cy, nw, nh


# ── Create output folder structure ──────────────────────────────────────────
def create_dirs(output_dir):
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)


# ── Write data.yaml ─────────────────────────────────────────────────────────
def write_yaml(output_dir):
    yaml_content = f"""# HAM10000 Binary YOLOv8 Dataset
# Benign vs Malignant skin lesion detection

path: {os.path.abspath(output_dir)}
train: images/train
val:   images/val

nc: 2
names:
  0: Benign
  1: Malignant
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"  data.yaml → {yaml_path}")


# ── Main conversion ─────────────────────────────────────────────────────────
def convert(args):
    data_dir   = args.data_dir
    output_dir = args.output_dir

    image_dir = os.path.join(data_dir, "images")
    mask_dir  = os.path.join(data_dir, "masks")
    csv_path  = os.path.join(data_dir, "GroundTruth.csv")

    # Validate paths
    for path, name in [(image_dir, "images/"),
                       (mask_dir,  "masks/"),
                       (csv_path,  "GroundTruth.csv")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find {name} at: {path}")

    print(f"\n{'='*55}")
    print(f"  HAM10000 → YOLOv8 Binary Converter")
    print(f"  Classes: 0=Benign | 1=Malignant")
    print(f"{'='*55}")
    print(f"  Source : {os.path.abspath(data_dir)}")
    print(f"  Output : {os.path.abspath(output_dir)}")
    print(f"  Val    : {args.val_split * 100:.0f}%")
    print(f"{'='*55}\n")

    # Load CSV and assign binary labels
    df = pd.read_csv(csv_path)
    df["class_id"] = df.apply(get_binary_class, axis=1)

    print(f"  Total images       : {len(df)}")
    print(f"  Benign    (0)      : {(df['class_id'] == 0).sum()}")
    print(f"  Malignant (1)      : {(df['class_id'] == 1).sum()}\n")

    # Stratified split — keeps benign/malignant ratio in both sets
    train_df, val_df = train_test_split(
        df,
        test_size=args.val_split,
        stratify=df["class_id"],
        random_state=args.seed
    )

    print(f"  Train split        : {len(train_df)}")
    print(f"    Benign           : {(train_df['class_id'] == 0).sum()}")
    print(f"    Malignant        : {(train_df['class_id'] == 1).sum()}")
    print(f"  Val split          : {len(val_df)}")
    print(f"    Benign           : {(val_df['class_id'] == 0).sum()}")
    print(f"    Malignant        : {(val_df['class_id'] == 1).sum()}\n")

    create_dirs(output_dir)

    stats = {"train": {"done": 0, "skipped": 0},
             "val":   {"done": 0, "skipped": 0}}

    for split, split_df in [("train", train_df), ("val", val_df)]:
        img_out = os.path.join(output_dir, "images", split)
        lbl_out = os.path.join(output_dir, "labels", split)

        print(f"  Processing {split} set...")

        for _, row in tqdm(split_df.iterrows(),
                           total=len(split_df), ncols=70):
            img_id   = row["image"]
            class_id = row["class_id"]

            # Support .jpg and .png
            img_path = os.path.join(image_dir, f"{img_id}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(image_dir, f"{img_id}.png")

            mask_path = os.path.join(mask_dir,
                                     f"{img_id}_segmentation.png")

            if not os.path.exists(img_path) or \
               not os.path.exists(mask_path):
                stats[split]["skipped"] += 1
                continue

            img = cv2.imread(img_path)
            if img is None:
                stats[split]["skipped"] += 1
                continue
            h, w = img.shape[:2]

            bbox = mask_to_yolo_bbox(mask_path, w, h)
            if bbox is None:
                stats[split]["skipped"] += 1
                continue

            cx, cy, nw, nh = bbox

            # Copy image
            shutil.copy(img_path,
                        os.path.join(img_out, f"{img_id}.jpg"))

            # Write YOLO label
            label_path = os.path.join(lbl_out, f"{img_id}.txt")
            with open(label_path, "w") as f:
                f.write(f"{class_id} "
                        f"{cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            stats[split]["done"] += 1

        print(f"    Done    : {stats[split]['done']}")
        print(f"    Skipped : {stats[split]['skipped']}\n")

    write_yaml(output_dir)

    print(f"{'='*55}")
    print(f"  Conversion complete!")
    print(f"  Train : {stats['train']['done']} images")
    print(f"  Val   : {stats['val']['done']} images")
    print(f"\n  Training command:")
    print(f"  yolo detect train \\")
    print(f"    data={os.path.abspath(output_dir)}/data.yaml \\")
    print(f"    model=yolov8n.pt \\")
    print(f"    epochs=50 \\")
    print(f"    imgsz=640 \\")
    print(f"    batch=16 \\")
    print(f"    device=0")
    print(f"{'='*55}\n")


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    convert(args)