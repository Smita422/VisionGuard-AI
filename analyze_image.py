from ultralytics import YOLO
from safety_rules import analyze_safety


# Load trained PPE model
model = YOLO("models/best.pt")


# Test image
image_path = "datasets/construction-ppe/images/test/image1003.jpg"


# Run YOLO detection
results = model.predict(
    source=image_path,
    conf=0.25
)


# Extract detected class names
detections = []

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        detections.append(class_name)


# Remove duplicate class names
detections = list(set(detections))


print("\nDetected Objects:")
for detection in detections:
    print("-", detection)


# Analyze safety
incidents = analyze_safety(detections)


print("\nSafety Analysis:")

if not incidents:
    print("✅ No safety violations detected.")

else:
    for incident in incidents:
        print(f"⚠️ {incident['type']}")
        print(f"Severity: {incident['severity']}")
        print(f"Message: {incident['message']}")