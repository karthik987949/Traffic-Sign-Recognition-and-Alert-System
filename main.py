import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from utils.preprocess import draw_bounding_box  # Remove preprocess_frame import
import threading
import time
import torch
import ultralytics.nn.tasks
import ultralytics.nn.modules
import torch.nn.modules.container
import torch.nn.modules.conv

# Allowlist required classes for PyTorch
torch.serialization.add_safe_globals([
    ultralytics.nn.tasks.DetectionModel,
    torch.nn.modules.container.Sequential,
    ultralytics.nn.modules.Conv,
    torch.nn.modules.conv.Conv2d
])

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# Load YOLOv8 model
model = YOLO("model/yolov8n.pt")

# Define traffic sign classes (aligned with COCO)
sign_classes = {
    11: "Stop Sign",
    9: "Traffic Light",
    8:"U Turn"
}

class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Recognition System")
        self.root.geometry("1200x600")

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(pady=10)

        self.video_label = tk.Label(self.main_frame)
        self.video_label.pack(side=tk.LEFT, padx=20)

        self.sign_label = tk.Label(self.main_frame, text="No Sign Detected", font=("Arial", 14))
        self.sign_label.pack(side=tk.LEFT, padx=20)

        self.alert_label = tk.Label(self.root, text="No Sign Detected", font=("Arial", 14), fg="red")
        self.alert_label.pack(pady=10)

        self.start_button = tk.Button(self.root, text="Start Live Detection", command=self.start_live_detection)
        self.start_button.pack(pady=5)
        self.upload_button = tk.Button(self.root, text="Upload Video", command=self.upload_video)
        self.upload_button.pack(pady=5)
        self.stop_button = tk.Button(self.root, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(pady=5)

        self.cap = None
        self.running = False
        self.last_alert = None
        self.alert_cooldown = 5
        self.last_alert_time = time.time()

    def start_live_detection(self):
        self.cap = cv2.VideoCapture(0)  # Try index 0 first
        if not self.cap.isOpened():
            print("Error: Could not open webcam (index 0), trying index 1")
            self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            self.alert_label.config(text="Error: Could not open webcam")
            print("Error: Could not open webcam")
            return
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.upload_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        threading.Thread(target=self.update_frame, daemon=True).start()

    def upload_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                self.alert_label.config(text="Error: Could not open video file")
                print("Error: Could not open video file")
                return
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.upload_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            threading.Thread(target=self.update_frame, daemon=True).start()

    def stop_detection(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_button.config(state=tk.NORMAL)
        self.upload_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.alert_label.config(text="No Sign Detected")
        self.video_label.config(image='')
        self.sign_label.config(image='', text="No Sign Detected")

    def update_frame(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                self.stop_detection()
                break

            print(f"Raw frame shape: {frame.shape}")  # Debug print
            orig_height, orig_width = frame.shape[:2]
            scale_x, scale_y = orig_width / 640, orig_height / 640

            # Use raw frame for inference (Ultralytics handles preprocessing)
            results = model(frame, verbose=False, imgsz=640)
            detections = results[0].boxes
            print(f"Number of detections: {len(detections)}")  # Debug print
            for box in detections:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"Detected: Class {cls}, Confidence {conf:.2f}")  # Debug print

            alert_text = "No Sign Detected"
            sign_image = None
            for box in detections:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > 0.1:  # Lowered threshold
                    label = sign_classes.get(cls, "Unknown")
                    alert_text = f"Detected: {label}"
                    box_coords = box.xyxy[0]
                    x1, y1, x2, y2 = map(int, box_coords)  # No scaling needed (Ultralytics returns in original frame coords)
                    frame = draw_bounding_box(frame, [x1, y1, x2, y2], label, conf)

                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(orig_width, x2), min(orig_height, y2)
                    if x2 > x1 and y2 > y1:
                        sign_image = frame[y1:y2, x1:x2]
                        sign_image = cv2.resize(sign_image, (200, 200))

                    if label != self.last_alert or (time.time() - self.last_alert_time > self.alert_cooldown):
                        self.speak_alert(label)
                        self.last_alert = label
                        self.last_alert_time = time.time()

            self.update_ui(frame, sign_image, alert_text)
            time.sleep(0.03)

    def speak_alert(self, label):
        threading.Thread(target=engine.say, args=(label,), daemon=True).start()
        engine.runAndWait()

    def update_ui(self, frame, sign_image, alert_text):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((640, 480), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        if sign_image is not None:
            sign_rgb = cv2.cvtColor(sign_image, cv2.COLOR_BGR2RGB)
            sign_img = Image.fromarray(sign_rgb)
            sign_imgtk = ImageTk.PhotoImage(image=sign_img)
            self.sign_label.imgtk = sign_imgtk
            self.sign_label.config(image=sign_imgtk, text=alert_text)
        else:
            self.sign_label.config(image='', text=alert_text)

        self.alert_label.config(text=alert_text)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignApp(root)
    app.run()