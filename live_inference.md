# Live Skin Lesion Detection on Raspberry Pi 5

## Overview

The trained and compiled YOLOv8 model is deployed on Raspberry Pi 5 using the Hailo-8L AI accelerator.

The Hailo detection application is used to perform inference using a live Raspberry Pi camera input.

## Hardware

- Raspberry Pi 5
- Hailo-8L AI accelerator
- Compatible Raspberry Pi camera
- Compiled YOLOv8 HEF model

## Labels Configuration

The Hailo detection application uses a JSON configuration containing:

- Detection threshold: 0.5
- Maximum detection boxes: 200
- Background/unlabelled entry
- Benign
- Malignant

The additional unlabelled entry is used for the detection application's label configuration and is not a third trained lesion class.

## Live Inference

Run the detection application:

```bash
hailo-detect \
    --hef-path /path/to/yolov8.hef \
    --input rpi \
    --labels-json /path/to/labels.json
```

The application captures camera frames and performs inference using the Hailo accelerator.

Predictions are displayed with bounding boxes and confidence scores.

## Hardware Monitoring

During deployment, system CPU usage and Hailo accelerator utilization were monitored.

The recorded observations were:

| Metric | Observation |
|---|---:|
| Maximum CPU usage | Approximately 25% |
| Hailo utilization | Approximately 50% |

These values represent observations from the documented test session and are not guaranteed performance figures.

## Limitations

- Class imbalance remains a concern in the dataset.
- Detection quality depends on the training annotations.
- The system is an experimental implementation, not a clinically validated diagnostic tool.
- Hardware utilization depends on the inference workload and runtime configuration.

## Future Improvements

- Evaluate inference latency and FPS under controlled conditions.
- Improve class balance.
- Evaluate performance on independent dermatoscopic images.
- Investigate model quantization and optimization.
- Develop a more robust real-time camera inference interface.
