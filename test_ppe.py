from ultralytics import YOLO
from pathlib import Path

# Load our trained PPE model
model = YOLO("models/best.pt")

# Find a test image
test_folder = Path("datasets/construction-ppe/images/test")
images = list(test_folder.glob("*.jpg"))

if not images:
    print("No test images found.")
    exit()

image_path = images[0]

print(f"Testing image: {image_path}")

# Run PPE detection
results = model.predict(
    source=str(image_path),
    conf=0.25,
    save=True
)

print("Detection completed!")