import cv2
import mediapipe as mp
import numpy as np
import math
import pyautogui
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
camera = cv2.VideoCapture(0)

last_action_time = 0      
action_cooldown = 1.0     
ripples = []              
current_action = "Waiting for gesture..."

while camera.isOpened():
    success, frame = camera.read()
    if not success: break

    frame = cv2.flip(frame, 1) # horizontal flip so frame matches mirror view
    height, width, _ = frame.shape
    
    # BGR to RGB conversion for mediapipe to work
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Add HUD Text
    cv2.putText(frame, f"Status: {current_action}", (width // 2 - 100, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    if results.multi_hand_landmarks and results.multi_handedness:
        for fingers, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

            index = fingers.landmark[8]
            thumb = fingers.landmark[4]

            # "Right" or "Left" hand label
            label = handedness.classification[0].label 
            
            indexX, indexY = int(index.x * width), int(index.y * height)
            thumbX, thumbY = int(thumb.x * width), int(thumb.y * height)
            
            # distance between index and thumb
            distance = math.hypot(indexX - thumbX, indexY - thumbY)

            color = (0, 255, 0) if label == "Right" else (0, 0, 255) # green for right hand, red for left hand

            # draw line between fingers
            cv2.line(frame, (indexX, indexY), (thumbX, thumbY), color, 3)

            # draw circles at fingertips
            cv2.circle(frame, (indexX, indexY), 10, (255, 255, 255), -1) # inner white circle
            cv2.circle(frame, (indexX, indexY), 15, color, 2) # outer circle  
            cv2.circle(frame, (thumbX, thumbY), 10, (255, 255, 255), -1)
            cv2.circle(frame, (thumbX, thumbY), 15, color, 2)

            # distance text display
            cv2.putText(frame, f"Distance: {int(distance)}", (indexX, indexY - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # handedness display
            cv2.putText(frame, f"Hand: {label}", (thumbX, thumbY + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            current_time = time.time()
            
            # pinch fingers
            if distance < 20 and (current_time - last_action_time) > action_cooldown and label == "Right":
                
                if indexX > width * 0.7:  # hand is on the RIGHT side
                    pyautogui.press('nexttrack')
                    current_action = "Next Track ->"
                    ripples.append([indexX, indexY, 5]) # Spawn a ripple
                    last_action_time = current_time
                    
                elif indexX < width * 0.3: # hand is on the LEFT side
                    pyautogui.press('prevtrack')
                    current_action = "<- Previous Track"
                    ripples.append([indexX, indexY, 5]) 
                    last_action_time = current_time
                    
                else: # hand is in the MIDDLE
                    pyautogui.press('playpause')
                    current_action = "Play / Pause"
                    ripples.append([indexX, indexY, 5]) 
                    last_action_time = current_time

    for r in ripples[:]: # iterate over a copy of the list
        rx, ry, radius = r[0], r[1], r[2]
        
        cv2.circle(frame, (rx, ry), radius, (255, 255, 255), 2) 
        # expand radius
        r[2] += 8 
    
        if r[2] > 150:
            ripples.remove(r)

    cv2.imshow("hands", frame)

    # key input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key in [ord('1'), ord('2'), ord('3')]:
        mode = int(chr(key))
        points = []

camera.release()
cv2.destroyAllWindows()