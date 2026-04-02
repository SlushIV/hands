import cv2
import mediapipe as mp
import numpy as np
import math
import pyautogui
import time

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import pyaudio
import threading
from pedalboard import Pedalboard, Reverb, LowpassFilter

# shared variable for audio thread
fx_params = {"cutoff": 5000, "reverb": 0.0} 

# audio thread
def audio_processor():

    # YOUR DEVICE IDs (test with virtualcabletest.py)
    INPUT_ID = 1
    OUTPUT_ID = 10
    
    CHUNK = 1024
    RATE = 44100 # ENSURE WINDOWS AUDIO SETTINGS ARE ALSO AT 44100Hz TO AVOID ISSUES
    audio = pyaudio.PyAudio()
    
    board = Pedalboard([
        LowpassFilter(cutoff_frequency_hz=5000),
        Reverb(room_size=0.0)
    ])

    try:
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=RATE,
            input=True,
            output=True,
            input_device_index=INPUT_ID,
            output_device_index=OUTPUT_ID,
            frames_per_buffer=CHUNK
        )
        print(f"Audio Stream Active: Input({INPUT_ID}) -> Output({OUTPUT_ID})")
    except Exception as e:
        print(f"CRITICAL AUDIO ERROR: {e}")
        print("Check if your Headphones are plugged in and Sample Rate is 44100Hz")
        return # exit thread but keep main app running

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            # update the FX board with shared parameters (fx_params)
            board[0].cutoff_frequency_hz = fx_params["cutoff"]
            board[1].room_size = fx_params["reverb"]

            effected = board(audio_data, RATE)
            out_data = (effected * 32768.0).astype(np.int16).tobytes()
            stream.write(out_data)
            time.sleep(0.005) # small delay to prevent CPU overload
        except Exception as e:
            continue

# audio setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_controller = cast(interface, POINTER(IAudioEndpointVolume))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

cv2.namedWindow("hands", cv2.WINDOW_NORMAL)
cv2.resizeWindow("hands", 800, 600)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Camera failed to open")
    exit()

# start audio thread
threading.Thread(target=audio_processor, daemon=True).start()

last_action_time = 0      
action_cooldown = 1.0     
last_volume = -1             
volume = volume_controller.GetMasterVolumeLevelScalar()
ripples = []

def resize_with_aspect_ratio(frame, max_w, max_h):
    h, w = frame.shape[:2]

    scale = min(max_w / w, max_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))

    # create black canvas
    result = cv2.copyMakeBorder(
        resized,
        (max_h - new_h) // 2,
        (max_h - new_h) // 2,
        (max_w - new_w) // 2,
        (max_w - new_w) // 2,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    return result


while camera.isOpened():
    success, frame = camera.read()
    if not success: 
        print("Failed to capture video frame")
        break

    frame = cv2.flip(frame, 1) # horizontal flip so frame matches mirror view
    
    _, _, win_w, win_h = cv2.getWindowImageRect("hands")

    if win_w > 0 and win_h > 0:
        frame = resize_with_aspect_ratio(frame, win_w, win_h)

    height, width, _ = frame.shape
    
    # BGR to RGB conversion for mediapipe to work
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # volume percentage display
    cv2.putText(frame, f"Volume: {int(volume * 100)}%", (width // 2 - 70, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
    # zone boundaries
    left_bound = int(width * 0.3)
    right_bound = int(width * 0.7)

    # vertical zone lines
    cv2.line(frame, (left_bound, 0), (left_bound, height), (255, 255, 255), 1)
    cv2.line(frame, (right_bound, 0), (right_bound, height), (255, 255, 255), 1)

    # zone labels
    cv2.putText(frame, "PREV", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "PAUSE", (left_bound + 10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "NEXT", (right_bound + 10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
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
            cv2.putText(frame, f"Distance: {int(distance)}", (indexX, indexY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # handedness display
            cv2.putText(frame, f"Hand: {label}", (thumbX, thumbY + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            current_time = time.time()

            # volume control with left hand
            if label == "Left":

                y_percent = np.clip(indexY / height, 0, 1) # 0.0 at top, 1.0 at bottom

                # indexY / height gives us 0.0 (top) to 1.0 (bottom)
                # Top = 20,000Hz (Clear), Bottom = 200Hz (Muffled)
                cutoff_val = np.interp(indexY, [0, height], [20000, 200])
                fx_params["cutoff"] = float(cutoff_val)

                # 30 pixels (pinched) = 0.0 Reverb, 200 pixels (wide) = 0.8 Reverb
                reverb_val = np.interp(distance, [30, 200], [0.0, 0.8])
                fx_params["reverb"] = float(np.clip(reverb_val, 0, 1))

                # interpolate distance to volume range     
                volume = np.interp(distance, [height / 20, height / 2], [0, 1])

                # only update if volume changes
                if abs(volume - last_volume) > 0.01:
                    
                    # set volume
                    volume_controller.SetMasterVolumeLevelScalar(volume, None)
                    last_volume = volume
     
                        
            # pinch fingers
            if distance < (height / 20) and (current_time - last_action_time) > action_cooldown and label == "Right":
                
                if indexX > right_bound:  # hand is on the RIGHT side
                    pyautogui.press('nexttrack')
                    ripples.append([indexX, indexY, 5]) # spawn a ripple
                    last_action_time = current_time
                    
                elif indexX < left_bound: # hand is on the LEFT side
                    pyautogui.press('prevtrack')
                    ripples.append([indexX, indexY, 5]) 
                    last_action_time = current_time
                    
                else: # hand is in the MIDDLE
                    pyautogui.press('playpause')
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