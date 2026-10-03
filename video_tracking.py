from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Webcam
video_source = 0

print("Starting object tracking...")
print("Press Q to quit.")

# Run tracking
results = model.track(
    source=video_source,
    conf=0.25,
    persist=True,
    stream=True,
    tracker="bytetrack.yaml",
    show=True,
    verbose=False
)

# Process each frame
for result in results:

    if result.boxes is None:
        continue

    # Check if tracking IDs exist
    if result.boxes.is_track and result.boxes.id is not None:

        tracking_ids = result.boxes.id.int().cpu().tolist()
        class_ids = result.boxes.cls.int().cpu().tolist()

        for track_id, class_id in zip(
            tracking_ids,
            class_ids
        ):

            class_name = model.names[class_id]

            print(
                f"Track ID: {track_id} | "
                f"Object: {class_name}"
            )