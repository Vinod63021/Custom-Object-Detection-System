import torch
from ultralytics import YOLO


def main():

    # Stability settings
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Load YOLOv8 Nano
    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=512,
        batch=4,
        workers=2,
        device=0,
        amp=False
    )


if __name__ == "__main__":
    main()