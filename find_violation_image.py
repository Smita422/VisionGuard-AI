from ultralytics import YOLO
from pathlib import Path


model = YOLO("models/best.pt")

test_folder = Path("datasets/construction-ppe/images/test")

images = list(test_folder.glob("*.jpg"))

print(f"Checking {len(images)} test images...\n")


violation_classes = {
    "no_helmet",
    "no_goggle",
    "no_gloves",
    "no_boots",
    "none"
}


for image_path in images:

    results = model.predict(
        source=str(image_path),
        conf=0.25,
        verbose=False
    )

    detections = []

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            detections.append(class_name)


    violations = [
        item for item in detections
        if item in violation_classes
    ]


    if violations:

        print("Violation image found!")
        print("Image:", image_path)
        print("Detected:", detections)
        print("Violations:", violations)

        break