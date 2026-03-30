import cv2
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
camera = cv2.VideoCapture(0)

canvas = None

print("Controls: '1' for Ribbon, '2' for Mandala, '3' for Distortion, 'c' to Clear, 'q' to Quit")

while camera.isOpened():
    success, frame = camera.read()
    if not success: break

    frame = cv2.flip(frame, 1) # horizontal flip so frame matches mirror view
    height, width, _ = frame.shape
    if canvas is None: canvas = np.zeros_like(frame) # new blank (black) canvas
    
    # BGR to RGB conversion for mediapipe to work
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for fingers in results.multi_hand_landmarks:
            index = fingers.landmark[8]
            thumb = fingers.landmark[4]
            
            indexX, indexY = int(index.x * width), int(index.y * height)
            thumbX, thumbY = int(thumb.x * width), int(thumb.y * height)
            
            # distance between index and thumb
            distance = math.hypot(indexX - thumbX, indexY - thumbY)

    cv2.imshow("hands", frame)

    # key input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('c'): canvas = np.zeros_like(frame)
    elif key in [ord('1'), ord('2'), ord('3')]:
        mode = int(chr(key))
        points = []

camera.release()
cv2.destroyAllWindows()