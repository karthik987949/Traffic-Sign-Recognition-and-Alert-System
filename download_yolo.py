from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # Automatically downloads if not present
model.export(format="pt")  # Exports to .pt format