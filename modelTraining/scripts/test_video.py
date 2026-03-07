from ultralytics import YOLO

model = YOLO(r"C:\Users\vinod\Downloads\ANPR_Training\runs\detect\train5\weights\best.pt")

model.predict(
    source=r"C:\Users\vinod\Downloads\ANPR_Training\scripts\traffic.mp4",
    show=True,
    save=True,
    conf=0.4
)

print("Video processed")