from ultralytics import YOLO

model = YOLO("/home/aisd_user03/project/software_development/runs/detect/runs/detect/skin_lesion/weights/best.pt")
model.export(
    format   = "onnx",
    imgsz    = 640,
    simplify = True,
    opset    = 12,
    half     = False,   # Keep FP32 for RPi CPU
    dynamic  = False,
)