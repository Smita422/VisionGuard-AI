from ultralytics import YOLO


# ------------------------------------------------
# Load trained PPE model
# ------------------------------------------------

model = YOLO("models/best.pt")


# ------------------------------------------------
# Test image
# ------------------------------------------------

image_path = "datasets/construction-ppe/images/test/image1003.jpg"


# ------------------------------------------------
# Run YOLO detection
# ------------------------------------------------

results = model.predict(
    source=image_path,
    conf=0.25
)


# ------------------------------------------------
# Store persons and PPE detections
# ------------------------------------------------

persons = []
ppe_items = []


# ------------------------------------------------
# PPE classes
# ------------------------------------------------

ppe_classes = {
    "helmet",
    "gloves",
    "vest",
    "boots",
    "goggles",
    "no_helmet",
    "no_gloves",
    "no_boots",
    "no_goggle",
    "none"
}


# ------------------------------------------------
# Extract detections
# ------------------------------------------------

for result in results:

    for box in result.boxes:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detection = {
            "class": class_name,
            "bbox": [x1, y1, x2, y2]
        }

        if class_name == "Person":
            persons.append(detection)

        elif class_name in ppe_classes:
            ppe_items.append(detection)


# ------------------------------------------------
# Function to calculate bounding-box center
# ------------------------------------------------

def get_center(bbox):

    x1, y1, x2, y2 = bbox

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return center_x, center_y


# ------------------------------------------------
# Check whether PPE belongs to a person
# ------------------------------------------------

def is_inside(person_bbox, ppe_bbox):

    px1, py1, px2, py2 = person_bbox

    ppe_center_x, ppe_center_y = get_center(ppe_bbox)

    return (
        px1 <= ppe_center_x <= px2
        and
        py1 <= ppe_center_y <= py2
    )


# ------------------------------------------------
# Required PPE
# ------------------------------------------------

required_ppe = {
    "helmet",
    "goggles",
    "vest",
    "gloves",
    "boots"
}


# ------------------------------------------------
# Violation class mapping
# ------------------------------------------------

violation_mapping = {
    "no_helmet": "helmet",
    "no_goggle": "goggles",
    "no_gloves": "gloves",
    "no_boots": "boots"
}


# ------------------------------------------------
# Person-wise Safety Assessment
# ------------------------------------------------

print("\nPerson-wise Safety Assessment:")


for i, person in enumerate(persons):

    person_ppe = []

    # Find PPE associated with this person
    for ppe in ppe_items:

        if is_inside(person["bbox"], ppe["bbox"]):

            person_ppe.append(ppe["class"])


    # Remove duplicate detections
    person_ppe = set(person_ppe)


    print(f"\nPerson {i + 1}")


    # Track whether an actual violation was detected
    unsafe = False


    # Check every required PPE item
    for item in required_ppe:

        # PPE was positively detected
        if item in person_ppe:

            print(f"{item.capitalize():8}: ✓")


        # Explicit violation was detected
        elif any(
            violation_class in person_ppe
            and ppe_name == item
            for violation_class, ppe_name in violation_mapping.items()
        ):

            print(f"{item.capitalize():8}: ✗")

            print(
                f"  Violation: Missing {item.capitalize()}"
            )

            unsafe = True


        # Neither PPE nor violation was detected
        else:

            print(f"{item.capitalize():8}: ?")


    # ------------------------------------------------
    # Final safety status
    # ------------------------------------------------

    if unsafe:

        print("\nStatus: UNSAFE")

    else:

        print("\nStatus: SAFE")