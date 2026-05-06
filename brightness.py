import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import screen_brightness_control as sbc
import numpy as np
import urllib.request
import os
import ssl

# 1. Bypass SSL verification for the download ONLY
ssl._create_default_https_context = ssl._create_unverified_context

model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading model file... please wait.")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete.")

# 2. Setup the Detector (Modern Task API)
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        # Process image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detection_result = detector.detect(mp_image)
        
        if detection_result.hand_landmarks:
            hand_lms = detection_result.hand_landmarks[0]
            h, w, _ = frame.shape
            
            # Thumb (4) and Index (8)
            x1, y1 = int(hand_lms[4].x * w), int(hand_lms[4].y * h)
            x2, y2 = int(hand_lms[8].x * w), int(hand_lms[8].y * h)
            
            # Distance -> Brightness
            dist = np.hypot(x2-x1, y2-y1)
            bright = np.interp(dist, [20, 200], [0, 100])
            
            try:
                sbc.set_brightness(int(bright))
            except:
                pass
            
            # Visuals
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"Brightness: {int(bright)}%", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Program closed.")
