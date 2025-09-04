import openwakeword
import sounddevice as sd
import time
import numpy as np
from scipy.signal import resample_poly
from openwakeword.model import Model

# --------------------------
# Settings
# --------------------------
MIC_SAMPLE_RATE = 44100   # your mic’s native rate
MODEL_SAMPLE_RATE = 16000 # OpenWakeWord expects 16kHz
BLOCK_SIZE = 1024         # ~23ms at 44.1kHz
THRESHOLD = 0.3           # detection threshold

# Load wake word model
openwakeword.utils.download_models()
model = Model(
    wakeword_models=["hey_jarvis_v0.1"],
    inference_framework="onnx"
)

def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️", status)

    audio = indata[:, 0]

    # Resample 44.1k → 16k
    audio_16k = resample_poly(audio, up=160, down=441)

    scores = model.predict(audio_16k)
    print(scores)   # <--- dump everything


# --------------------------
# Start Listening
# --------------------------
print("🎙️ Say 'Hey Jarvis'...")
print(sd.query_devices(sd.default.device['input']))

with sd.InputStream(
    channels=1,
    samplerate=MIC_SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    callback=audio_callback
):
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n👋 Stopped listening.")
