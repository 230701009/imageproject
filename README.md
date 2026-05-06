# Gesture Brightness Control System

A Python-based computer vision project that controls system screen brightness using hand gestures through a webcam.

This project uses:
- OpenCV for webcam handling
- MediaPipe for hand tracking
- NumPy for distance calculations
- screen_brightness_control for brightness adjustment

---

# Features

- Real-time hand tracking
- Controls brightness using thumb and index finger distance
- Live webcam display with gesture visualization
- Automatic MediaPipe model download
- Simple and lightweight project

---

# Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- screen_brightness_control

---

# Project Working

The webcam detects the user's hand using MediaPipe Hand Landmarker.

The system tracks:
- Thumb tip landmark (4)
- Index finger tip landmark (8)

The distance between these two fingers is calculated and mapped to screen brightness levels from 0% to 100%.

- Fingers close together → Lower brightness
- Fingers farther apart → Higher brightness

---
gesture-brightness-control/
│
├── gesture_brightness.py
├── hand_landmarker.task
├── README.md
