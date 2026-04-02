# THIS SCRIPT LISTS ALL AUDIO DEVICES! FIND THE CORRECT ID AND REPLACE IN MAIN.PY

import pyaudio

audio = pyaudio.PyAudio()
for i in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(i)
    print(f"ID {i}: {info['name']}")
audio.terminate()

