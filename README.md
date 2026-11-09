# HAM10000 Skin Lesion Detection using YOLOv8 and Hailo-8L

**Edge AI | Deep Learning | Computer Vision | Medical Image Analysis**

An end-to-end Edge AI project for binary skin lesion detection using YOLOv8 and the HAM10000-derived dataset. The trained model is converted into ONNX format, compiled into Hailo Executable Format (HEF), and deployed on Raspberry Pi 5 with a Hailo-8L AI accelerator for live camera inference.

---

## Project Overview

Skin lesion analysis is an important application of computer vision in medical imaging. This project explores the use of deep learning and hardware acceleration to distinguish between benign and malignant skin lesions.

The project covers the complete workflow, from dataset preparation and model training to model conversion, hardware compilation, and real-time inference on an edge device.

The primary objective is to investigate how deep learning models can be deployed efficiently on resource-constrained hardware using dedicated AI accelerators.

## Key Features

- Binary skin lesion detection: Benign vs Malignant.
- YOLOv8-based deep learning model.
- Transfer learning using pretrained YOLO weights.
- Image augmentation and model optimization.
- ONNX model export.
- Hailo Dataflow Compiler Studio integration.
- HEF generation for Hailo-8L.
- Raspberry Pi 5 deployment.
- Live camera inference using HailoRT.
- CPU and accelerator utilization monitoring.

## Technology Stack

| Category | Technology |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| Framework | Ultralytics YOLO |
| Model | YOLOv8 |
| Dataset | HAM10000-derived dataset |
| Model Export | ONNX |
| Model Compilation | Hailo Dataflow Compiler |
| Edge Accelerator | Hailo-8L |
| Hardware | Raspberry Pi 5 |
| Inference Runtime | HailoRT |
| Operating System | Ubuntu / Raspberry Pi OS |

## Dataset

The project uses a binary-labelled dataset derived from HAM10000 (Human Against Machine with 10000 training images).

The original HAM10000 dataset contains approximately 10,000 dermatoscopic images belonging to seven diagnostic categories.

For this implementation, the target categories are:

| Class ID | Class |
|---|---|
| 0 | Benign |
| 1 | Malignant |

The original dataset contains the following diagnostic categories:

- AKIEC – Actinic keratoses and intraepithelial carcinoma
- BCC – Basal cell carcinoma
- BKL – Benign keratosis-like lesions
- DF – Dermatofibroma
- MEL – Melanoma
- NV – Melanocytic nevi
- VASC – Vascular lesions

The binary class mapping is used for the detection task. The mapping methodology should be documented alongside the dataset preparation scripts.

**Dataset reference:**  
https://doi.org/10.1038/sdata.2018.161

The dataset is not included in this repository. Users must obtain it separately and comply with its applicable licensing and usage conditions.

## Model Architecture

The project uses YOLOv8 Nano (YOLOv8n) as the initial training model.

YOLOv8 is a convolutional neural network-based object detection architecture capable of identifying objects and predicting their bounding boxes and associated classes.

The model is initialized using pretrained weights and fine-tuned on the binary skin lesion dataset.

### Training Configuration

| Parameter | Configuration |
|---|---|
| Model initialization | yolov8n.pt |
| Epochs | 100 |
| Early stopping patience | 20 |
| Input resolution | 640 × 640 |
| Batch size | 16 |
| Optimizer | SGD |
| Initial learning rate | 0.001 |
| Final learning rate factor | 0.01 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Learning rate scheduler | Cosine |
| Warmup epochs | 3 |
| Mixed precision | Enabled |
| Number of classes | 2 |

### Data Augmentation

The following augmentation techniques are applied during training:

- Horizontal and vertical flipping
- HSV augmentation
- Image rotation
- Translation
- Scaling
- Mosaic augmentation
- MixUp augmentation

These transformations are used to improve the model's ability to generalize across different image appearances.

### Training

The model is trained using the Ultralytics YOLO framework.

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="configs/data.yaml",
    epochs=100,
    patience=20,
    imgsz=640,
    batch=16,
    optimizer="SGD",
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    cos_lr=True,
    augment=True,
    amp=True,
    plots=True,
    save=True,
    device=0
)
```

The full training configuration is available in the source code.

## ONNX Model Export

After training, the model is exported into ONNX format.

ONNX provides an interoperable representation of the trained model and serves as the input to the Hailo compilation workflow.

Example:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/skin_lesion/weights/best.pt")

model.export(
    format="onnx",
    imgsz=640,
    simplify=True
)
```

The exported ONNX model is used in the subsequent Hailo compilation process.

## Hailo-8L Model Compilation

The ONNX model is compiled into Hailo Executable Format (HEF) using the Hailo toolchain.

The conversion process involves the following stages:

1. Export the trained YOLO model into ONNX format.
2. Prepare representative calibration images.
3. Launch Hailo Dataflow Compiler Studio.
4. Upload and parse the ONNX model.
5. Inspect the original ONNX graph and generated Hailo graph.
6. Identify the correct starting and ending nodes.
7. Compile the model for Hailo-8L.
8. Generate the final HEF file.

### Launch Hailo Dataflow Compiler Studio

```bash
hailo dfc-studio
```

Open the local address displayed in the terminal.

Upload the ONNX model and inspect the computation graph.

### HEF Compilation

An example compilation command is shown below:

```bash
hailomz compile yolov8s \
    --ckpt best.onnx \
    --calib-path /path/to/calibration/images \
    --hw-arch hailo8l \
    --classes 2 \
    --start-node-names images \
    --end-node-names "<verified-output-node-names>"
```

The architecture name, input node, output nodes and compilation options must match the actual ONNX graph and installed Hailo Model Zoo version.

The calibration dataset should contain representative training images and should use preprocessing consistent with the deployed model.

**Note:** The supplied training configuration initializes YOLOv8n, whereas the recorded Hailo deployment command references a YOLOv8s HEF. The exact model variant used for the final deployment must be verified against the actual ONNX and HEF artifacts before reproducing this workflow.

## Raspberry Pi 5 Deployment

The compiled HEF model is deployed on Raspberry Pi 5 with the Hailo-8L AI accelerator.

The Hailo detection application is used to perform inference using the Raspberry Pi camera.

### Hardware Requirements

- Raspberry Pi 5
- Hailo-8L AI HAT+
- Compatible Raspberry Pi camera
- MicroSD card
- Compatible power supply

### Live Camera Inference

The following command was used for live inference:

```bash
hailo-detect \
    --hef-path /home/satya/heffile/yolov8s.hef \
    --input rpi \
    --labels-json /home/satya/heffile/labels.json
```

The application captures camera frames and performs inference using the Hailo accelerator.

Predictions are displayed with bounding boxes, class labels and confidence scores.

### Labels Configuration

The Hailo detection application uses a JSON configuration similar to:

```json
{
    "detection_threshold": 0.5,
    "max_boxes": 200,
    "labels": [
        "Unlabeled",
        "Benign",
        "Malignant"
    ]
}
```

The additional Unlabeled entry is used by the detection application as a background/unlabelled class entry. It does not represent an additional trained lesion category.

The label ordering must be consistent with the model output and the runtime's class-index handling.

## Hardware Performance

During live inference, CPU usage and Hailo accelerator utilization were monitored using system monitoring tools and Hailo utilities.

The recorded observations were:

| Metric | Observed Value |
|---|---:|
| Maximum CPU utilization | Approximately 25% |
| Hailo accelerator utilization | Approximately 50% |

These values are observations from the documented deployment session, not standardized benchmark measurements.

The results demonstrate the use of dedicated hardware acceleration for edge inference while avoiding exclusive dependence on the Raspberry Pi CPU.

## Results and Demonstrations

The repository includes:

- Training performance plots
- Model evaluation outputs
- Confusion matrix
- Hailo model graph screenshots
- Live detection samples
- Raspberry Pi hardware demonstration
- Hardware utilization screenshots

See the `results/`, `docs/` and `assets/` directories.

## Repository Structure

```text
hailo-ham10000-yolov8-edge-ai/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── data.yaml.example
│   └── training_config.py
│
├── dataset/
│   └── README.md
│
├── src/
│   ├── train.py
│   ├── validate.py
│   ├── predict.py
│   └── export_onnx.py
│
├── models/
│   └── README.md
│
├── hailo/
│   ├── compilation_notes.md
│   ├── model_parsing.md
│   └── screenshots/
│
├── deployment/
│   ├── raspberry_pi_setup.md
│   ├── live_inference.md
│   └── labels.json
│
├── results/
│   ├── confusion_matrix.png
│   ├── results.png
│   ├── training_metrics.csv
│   └── detection_samples/
│
├── docs/
│   └── architecture.png
│
└── assets/
    └── demo.mp4
```

## Limitations

- The original HAM10000 dataset has a significant class imbalance.
- Binary class mapping may introduce ambiguity between diagnostic categories.
- Detection quality depends on the quality and consistency of bounding-box annotations.
- Performance on the training dataset does not guarantee generalization to independent clinical data.
- Quantization and hardware compilation may affect model accuracy.
- The system has not been clinically validated.

## Future Improvements

- Improve class imbalance handling.
- Evaluate additional lightweight CNN architectures.
- Investigate model quantization and optimization.
- Measure inference latency and throughput under controlled conditions.
- Implement improved real-time camera processing.
- Investigate model explainability techniques.
- Evaluate performance on independent dermatoscopic datasets.

## References

1. Tschandl, P., Rosendahl, C., and Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data. https://doi.org/10.1038/sdata.2018.161
2. Ultralytics YOLO Documentation: https://docs.ultralytics.com/
3. Hailo Developer Zone: https://hailo.ai/developer-zone/

---

**Project Category:** Deep Learning | Computer Vision | Edge AI | Hardware Acceleration

**Hardware Platform:** Raspberry Pi 5 + Hailo-8L AI Accelerator

**Disclaimer:** This project is intended for educational and experimental research purposes only. It is not a medical diagnostic system and must not be used to make clinical decisions.
