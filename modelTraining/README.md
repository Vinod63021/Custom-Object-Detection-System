# 🚗 Automatic Number Plate Recognition (ANPR) - YOLOv8

This project trains a **license plate detection model** using **YOLOv8**.
The trained model can detect vehicle license plates from images or video and can be used as the first step in an **ANPR (Automatic Number Plate Recognition) system**.

---

# 📌 Project Overview

This repository contains:

* Training pipeline for **YOLOv8 license plate detection**
* Dataset preparation in **YOLO format**
* GPU training using **PyTorch + CUDA**
* Exported trained model for inference

The trained model can be used for:

* 🚗 Vehicle monitoring
* 🅿 Parking automation
* 🚦 Smart traffic systems
* 🛂 Security checkpoints
* 📊 Vehicle counting systems

---

# 🧠 Model Details

| Feature           | Value                |
| ----------------- | -------------------- |
| Model             | YOLOv8s              |
| Framework         | PyTorch              |
| GPU               | NVIDIA RTX 3050      |
| Dataset Size      | 2976 training images |
| Validation Images | 698                  |
| Classes           | 1 (license plate)    |
| Image Size        | 512                  |
| Epochs            | 50                   |

### Final Model Performance

| Metric    | Score |
| --------- | ----- |
| Precision | 0.855 |
| Recall    | 0.729 |
| mAP50     | 0.757 |
| mAP50-95  | 0.355 |

---

# 📂 Project Structure

```
ANPR_Training
│
├── dataset
│   ├── train
│   │   ├── images
│   │   └── labels
│   │
│   ├── val
│   │   ├── images
│   │   └── labels
│   │
│   └── data.yaml
│
├── scripts
│   └── train.py
│
├── runs
│   └── detect
│       └── train5
│           └── weights
│               ├── best.pt
│               └── last.pt
│
├── yolov8s.pt
├── requirements.txt
└── README.md
```

---

# 📦 Installation

### 1️⃣ Clone Repository

```
git clone https://github.com/yourusername/anpr-yolov8.git
cd anpr-yolov8
```

### 2️⃣ Create Virtual Environment

```
python -m venv venv
```

Activate environment:

Windows

```
venv\Scripts\activate
```

Linux / Mac

```
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

Or manually:

```
pip install ultralytics torch torchvision opencv-python
```

---

# 📊 Dataset Format

Dataset must follow **YOLO format**.

```
dataset
│
├── train
│   ├── images
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   │
│   └── labels
│       ├── img1.txt
│       ├── img2.txt
│       └── ...
│
├── val
│   ├── images
│   └── labels
│
└── data.yaml
```

Example label file:

```
0 0.52 0.43 0.32 0.11
```

Format:

```
class_id x_center y_center width height
```

All values are **normalized (0-1)**.

---

# ⚙️ Training the Model

Run training using:

```
python scripts/train.py
```

Example training configuration:

```
model = YOLO("yolov8s.pt")

model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=512,
    batch=4,
    device=0,
    workers=2,
    amp=False
)
```

Training time on **RTX 3050 GPU**:

```
~1 hour 30 minutes
```

---

# 🏆 Best Model

After training, the best model is saved at:

```
runs/detect/train5/weights/best.pt
```

This is the **final trained license plate detector**.

---

# 🔍 Running Inference

Test the model on an image:

```
from ultralytics import YOLO

model = YOLO("runs/detect/train5/weights/best.pt")

results = model("test.jpg", show=True)
```

Or using CLI:

```
yolo detect predict model=runs/detect/train5/weights/best.pt source=test.jpg
```

---

# 📸 Example Output

The model detects license plates with bounding boxes:

```
+---------------------+
|   LICENSE PLATE     |
+---------------------+
```

---

# 🚀 Future Improvements

Possible upgrades:

* 🔤 License plate OCR (EasyOCR / Tesseract)
* 🚗 Vehicle type detection
* 📊 Vehicle counting system
* 📹 Real-time CCTV detection
* ⚡ TensorRT optimization for edge devices
* 🌐 Web dashboard for monitoring

---

# 📜 License

This project is open source under the **MIT License**.

---

# 👨‍💻 Author

**Vinod Kumar**

Computer Science Engineer
AI / Computer Vision Enthusiast

---

⭐ If you like this project, consider giving it a star!
