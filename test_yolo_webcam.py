import cv2
import numpy as np
from ultralytics import YOLO
import torch
import ultralytics.nn.tasks
import ultralytics.nn.modules
import torch.nn.modules.container
import torch.nn.modules.conv

# Allowlist required classes
torch.serialization.add_safe_globals([
    ultralytics.nn.tasks.DetectionModel,
    torch.nn.modules.container.Sequential,
    ultralytics.nn.modules.Conv,
    torch.nn.modules.conv.Conv2d
])

# Load model
model = YOLO("model/yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Preprocess frame
    resized = cv2.resize(frame, (640, 640))
    normalized = resized / 255.0
    rgb_frame = cv2.cvtColor(normalized.astype(np.float32), cv2.COLOR_BGR2RGB)

    # Run inference
    results = model(rgb_frame, verbose=False)
    detections = results[0].boxes

    # Draw bounding boxes
    for box in detections:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {cls} ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            print(f"Detected: Class {cls}, Confidence {conf:.2f}")

    # Display frame
    cv2.imshow("YOLO Webcam Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()