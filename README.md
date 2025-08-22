# 🚦 Traffic Sign Recognition and Alert System

A real-time computer vision system to detect traffic signs using YOLOv8, display Indian traffic rule alerts, and provide voice notifications. Built to enhance road safety awareness.

![Netlfix Clone Banner](assets/thumbnai.png) 
## 🔥 Features
- 📹 **Real-Time Detection**: Processes live webcam feed or video files (MP4, AVI, MOV).
- 🚸 **Traffic Sign Recognition**: Identifies GTSRB signs (e.g., Stop, Speed Limit, No Entry) using YOLOv8.
- 🇮🇳 **Indian Traffic Alerts**: Displays fines (e.g., ₹1,000 for Stop violation) per the Motor Vehicles Act, 2019.
- 🔊 **Voice Notifications**: Speaks sign names via `pyttsx3` (e.g., "Stop" every 5 seconds).
- 🖼️ **User Interface**: Tkinter GUI shows live video, cropped sign images, and text alerts.
- 🛠️ **Debugging**: Logs detections and saves frames for analysis.

## 📺 Expected Outputs
Tested on August 8, 2025, with a webcam and stop sign image on a phone:

- **Person in Front of Camera (No Sign)**:
  - **UI**: "No Sign Detected" (no bounding box or cropped sign).
  - **Voice**: No audio.
  - **Console**:
    ```
    Raw frame shape: (480, 640, 3)
    Number of detections: 1
    Raw detection: Class 0, Confidence 0.90, Label Speed Limit 20
    Skipping detection: Class 0 (likely person, mapped to Speed Limit 20)
    Valid traffic sign detections: 0
    Saved frame with no detections for debugging
    ```
  - **Saved Frame**: `no_detection_frame_<timestamp>.jpg` in project directory.

- **Stop Sign on Phone (~20% of Frame)**:
  - **UI**: Green bounding box, cropped stop sign, alert: "Stop: Obey sign, ₹1,000 fine for violation".
  - **Voice**: "Stop" spoken every 5 seconds.
  - **Console**:
    ```
    Raw frame shape: (480, 640, 3)
    Number of detections: 1
    Raw detection: Class 14, Confidence 0.85, Label Stop
    Valid traffic sign detections: 1
    ```

## 🛠️ Tech Stack
- ✅ Python 3.10
- ✅ YOLOv8 (`ultralytics==8.0.28`)
- ✅ OpenCV
- ✅ Tkinter
- ✅ pyttsx3
- ✅ Pillow, pywin32, comtypes

## 🚀 Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/traffic-sign-recognition.git
   ```

2. **Install Dependencies**:
   ```bash
   pip install opencv-python ultralytics==8.0.28 pillow pyttsx3 pywin32 comtypes labelimg
   ```

3. **Set Up File Structure**:
   ```
   traffic-sign-recognition/
   ├── model/
   │   └── yolov8n.pt
   ├── utils/
   │   └── preprocess.py
   ├── main.py
   ├── test_main_image.py
   ├── data.yaml
   ├── README.md
   ```

4. **Download YOLOv8 Model**:
   - Get `yolov8n.pt` from [Ultralytics Releases](https://github.com/ultralytics/assets/releases).
   - Place in `model/yolov8n.pt`.

5. **Create `utils/preprocess.py`**:
   ```python
   import cv2
   import numpy as np

   def draw_bounding_box(frame, box, label, conf):
       x1, y1, x2, y2 = box
       color = (0, 255, 0)  # Green for traffic signs
       thickness = 2
       cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
       cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
       return frame
   ```

6. **Create `data.yaml`**:
   ```yaml
   train: ./dataset/train
   val: ./dataset/test
   nc: 43
   names: ['Speed Limit 20', 'Speed Limit 30', 'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70', 'Speed Limit 80', 'Speed Limit 100', 'Speed Limit 120', 'End of Speed Limit', 'No Passing', 'No Passing for Vehicles Over 3.5t', 'Right of Way', 'Priority Road', 'Yield', 'Stop', 'No Vehicles', 'No Entry', 'General Caution', 'Dangerous Curve Left', 'Dangerous Curve Right', 'Double Curve', 'Bumpy Road', 'Slippery Road', 'Road Narrows Right', 'Road Work', 'Traffic Signals', 'Pedestrians', 'Children Crossing', 'Bicycles Crossing', 'Beware of Ice/Snow', 'Wild Animals Crossing', 'End of Restrictions', 'Turn Right Ahead', 'Turn Left Ahead', 'Ahead Only', 'Go Straight or Right', 'Go Straight or Left', 'Keep Right', 'Keep Left', 'Roundabout', 'End of No Passing', 'End of No Passing for Vehicles Over 3.5t', 'Customs']
   ```

7. **Run the Application**:
   ```bash
   cd traffic-sign-recognition
   python main.py
   ```
   - Click "Start Live Detection".
   - Test with a webcam (stand 1–2 meters away) or show a stop sign on your phone.
   - Use "Upload Video" for video files or run `test_main_image.py` with a stop sign image.

8. **Test with Static Image**:
   - Save a stop sign image to `C:/Users/<YourUsername>/Downloads/stop_sign.jpg` (e.g., from [Pexels](https://www.pexels.com/search/stop%20sign/)).
   - Run:
     ```bash
     python test_main_image.py
     ```

## 📚 What I Practiced
- 🐍 Python programming with OpenCV and Tkinter
- 🤖 Computer vision with YOLOv8 for object detection
- 🔊 Text-to-speech integration with `pyttsx3`
- 🚦 Indian traffic rule integration
- 🐛 Debugging real-time detection issues

## 📌 Future Scope
- 🎯 Fine-tune YOLOv8 on Indian traffic sign dataset
- 📱 Add support for mobile camera input
- 🌐 Integrate with real-time traffic APIs
- 🔐 Enhance GUI with user authentication
- 📊 Add analytics for detected signs

## 👨‍💻 Made with ❤️ by [Your Name]
📫 Reach me on [GitHub](https://github.com/your-username) | [LinkedIn](https://linkedin.com/in/your-linkedin)
