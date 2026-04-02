import cv2
import mediapipe as mp
import numpy as np
import math
import pyautogui
import time

import pyaudio
import threading
from pedalboard import Pedalboard, PitchShift, Bitcrush

# shared variable for audio thread
fx_params = {"pitch": 0.0, "bit_depth": 16.0, "volume": 1.0}

# audio thread
def audio_processor():
    INPUT_ID = 1
    OUTPUT_ID = 10
    
    CHUNK = 4096
    RATE = 44100
    audio = pyaudio.PyAudio()
    
    board = Pedalboard([
        PitchShift(semitones=0),
        Bitcrush(bit_depth=16)
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
        print(f"✅ Audio Stream Active: Input({INPUT_ID}) -> Output({OUTPUT_ID})")
    except Exception as e:
        print(f"❌ CRITICAL AUDIO ERROR: {e}")
        return

    current_pitch = 0.0

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            
            # no idea what this does tbh thank you gemini
            # 1. Convert to float and apply our Math Volume Control
            audio_1d = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_1d = audio_1d = np.clip(audio_1d * fx_params["volume"], -1.0, 1.0) # <--- DSP Volume!

            # 2. Un-shuffle the deck (Interleaved -> Separate L/R Channels)
            # Reshapes to (1024, 2) then transposes to (2, 1024)
            audio_stereo = audio_1d.reshape(-1, 2)

            # Update FX
            target_pitch = fx_params["pitch"]
            current_pitch += (target_pitch - current_pitch) * 0.2  # smoothing

            board[0].semitones = current_pitch

            board[1].bit_depth = fx_params["bit_depth"]

            # 3. Process the audio WITHOUT giving it amnesia (reset=False)
            effected = board(audio_stereo, RATE)

            # 4. Re-shuffle the deck and convert back to raw bytes
            effected_1d = effected.flatten()
            out_data = (effected_1d * 32768.0).astype(np.int16).tobytes()
            
            stream.write(out_data)
        except Exception as e:
            print(f"Audio Stream Error: {e}")
            continue

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
volume = 1.0
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

                if indexX > right_bound:  # hand is on the RIGHT side
                    # pitch shift based on distance
                    pitch_val = np.interp(distance, [height / 20, height / 2], [-12, 12])
                    fx_params["pitch"] = float(pitch_val)
                    
                elif indexX < left_bound: # hand is on the LEFT side
                    # map distance to bit depth
                    norm = np.clip(distance / (height / 2), 0, 1)
                    bit_val = 4 + (norm ** 2) * 12  # exponential curve for more sensitivity at lower distances
                    # bit_val = np.interp(distance, [height / 20, height / 2], [2, 16])
                    fx_params["bit_depth"] = float(bit_val)
                    
                else: # hand is in the MIDDLE
                    # interpolate distance for volume (0.0 to 1.0)
                    vol_val = np.interp(distance, [height / 20, height / 2], [0.0, 1.0])
                    fx_params["volume"] = float(vol_val)
                
                    # update display variable
                    volume = vol_val

                        
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