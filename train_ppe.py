from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Train on Construction-PPE dataset
results = model.train(
    data="construction-ppe.yaml",
    epochs=10,
    imgsz=640,
    batch=8,
    project="runs",
    name="ppe_detection"
)

print("PPE training completed!")