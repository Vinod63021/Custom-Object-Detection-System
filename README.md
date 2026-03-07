# 🎯 Custom Object Detection System
## Bangle Detection using YOLOv8

> A complete custom object detection pipeline — from raw images to trained model — built when general-purpose models aren't accurate enough for specialized objects.

---

## 📌 Overview

This project demonstrates how to build a **custom object detection model from scratch** using YOLOv8. Instead of relying on pre-trained models like COCO (which don't know what a bangle is), we collect our own images, annotate them with a custom GUI tool, and train a dedicated model — giving us full control over accuracy.

This pipeline can be reused for any custom object: vehicles, number plates, medical instruments, industrial parts, and more.

---

## 🚀 Features

- ✅ Custom GUI annotation tool (Tkinter-based)
- ✅ YOLO-format dataset preparation
- ✅ YOLOv8 Nano training pipeline
- ✅ Bounding box detection with confidence scores
- ✅ Evaluation with Precision, Recall, and mAP metrics
- ✅ Scalable to multi-class detection tasks

---

## 🗂️ Project Structure

```
YOLOv8-Bangle-Detection/
│
├── dataset/
│   ├── images/
│   │   ├── train/          # Training images
│   │   └── val/            # Validation images
│   ├── labels/
│   │   ├── train/          # YOLO annotation files
│   │   └── val/
│   ├── classes.txt         # Class names
│   └── data.yaml           # YOLO config file
│
├── runs/
│   └── detect/
│       └── train/
│           └── weights/
│               ├── best.pt # Best trained model
│               └── last.pt
│
├── annotator.py            # Custom annotation tool
├── train.py                # Model training script
├── detect.py               # Inference / detection script
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/yolov8-bangle-detection.git
cd yolov8-bangle-detection
```

**2. Install dependencies**
```bash
pip install ultralytics opencv-python pillow
```

**3. Verify GPU support (optional)**
```bash
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

---

## 📸 Dataset Preparation

### Step 1 — Collect Images
Gather 80–100+ images of bangles from:
- Mobile camera photos
- Google Images
- Online datasets

Place all images into `dataset/images/train/` and `dataset/images/val/`.

### Step 2 — Annotate Images
Run the custom annotation tool:
```bash
python annotator.py
```
- Draw bounding boxes around bangles
- Select the class label
- Export labels in YOLO format automatically

### Step 3 — Verify data.yaml
```yaml
path: dataset

train: images/train
val:   images/val

nc: 1

names:
  0: bangle
```

---

## 🏋️ Training the Model

```bash
python train.py
```

**Training parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| epochs | 50 | Training iterations |
| imgsz | 512 | Input image size |
| batch | 4 | Batch size |
| device | 0 | GPU (use `cpu` if no GPU) |

After training, the best model is saved to:
```
runs/detect/train/weights/best.pt
```

---

## 🔍 Running Detection

```bash
python detect.py
```

Or directly with the Ultralytics CLI:
```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=test.jpg conf=0.4
```

The model will draw bounding boxes on detected bangles and display the result.

---

## 📊 Evaluation Metrics

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| Precision | TP / (TP + FP) | How many detections are correct |
| Recall | TP / (TP + FN) | How many actual objects were found |
| mAP | Mean over IoU thresholds | Overall detection accuracy |

Training results and graphs are saved to `runs/detect/train/`.

---

## 🖥️ System Requirements

| Component | Minimum |
|-----------|---------|
| CPU | Intel Core i5 |
| RAM | 8 GB |
| GPU | Optional (NVIDIA recommended) |
| Storage | 10 GB free |
| Python | 3.8+ |

---

## 🔮 Future Extensions

This pipeline is designed to scale. After validating with bangles, the same workflow applies to:

- **Vehicle Detection** — car, truck, bus, bike, auto
- **Number Plate Detection** — crop + OCR pipeline
- **Smart Traffic Monitoring** — real-time vehicle counting and ANPR

```
Vehicle Detection → Number Plate Detection → OCR → Traffic Analytics
```

---

## 📦 Requirements

```
ultralytics
opencv-python
pillow
torch
torchvision
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
