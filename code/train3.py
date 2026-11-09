from ultralytics import YOLO

# Skin lesion detection - Benign vs Malignant
model = YOLO("yolov8n.pt")

model.train(
    data="/home/aisd_user03/project/software_development/data/data.yaml",

    # Training
    epochs=100,
    patience=20,

    # Image settings
    imgsz=640,
    batch=16,

    # Optimizer
    optimizer="SGD",
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,

    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,

    # Learning rate scheduler
    cos_lr=True,

    # Class imbalance handling (Benign ~73% / Malignant ~27%)
    cls=1.5,
    box=7.5,
    dfl=1.5,

    # Augmentation - dermoscopy specific
    augment=True,

    fliplr=0.5,
    flipud=0.5,

    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,

    degrees=180.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,

    mosaic=0.5,
    mixup=0.1,
    copy_paste=0.0,

    label_smoothing=0.1,

    # Performance
    amp=True,
    cache=True,
    rect=False,

    plots=True,
    verbose=True,
    save=True,
    save_period=10,
    val=True,

    device=0,
    workers=8,
    project="runs/detect",
    name="skin_lesion",
)

print("Training completed!")