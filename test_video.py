import cv2
cap = cv2.VideoCapture("C:\Users\LUCKY\OneDrive\Desktop\EDP\assets\Screenshot 2025-08-08 180901.png")
if not cap.isOpened():
    print("Error: Could not open video")
    exit()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Video Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()