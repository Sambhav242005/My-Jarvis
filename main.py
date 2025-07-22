import os
import time
import pyaudio
import numpy as np
import whisper
import requests
import webrtcvad
from collections import deque

# Disable Torch optimizations
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import torch
import torch._dynamo
torch._dynamo.disable()
torch.set_float32_matmul_precision('medium')

# Audio and model config
RATE = 16000
CHUNK_DURATION_MS = 30  # 30ms per chunk
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)
CHANNELS = 1

# Whisper and Ollama
asr_model = whisper.load_model("turbo").to("cuda")
OLLAMA_URL = "http://localhost:11435/api/generate"
OLLAMA_MODEL = "gemma3n"

# VAD setup
vad = webrtcvad.Vad()
vad.set_mode(1)  # 0=aggressive (less sensitive), 3=very aggressive (more sensitive)

def query_ollama(prompt: str, model=OLLAMA_MODEL, max_tokens=120):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens}
    }
    response = requests.post(OLLAMA_URL, json=payload)
    if response.ok:
        return response.json().get("response", "").strip()
    else:
        return f"[Error from Ollama]: {response.status_code} {response.text}"

def record_until_silence(stream, silence_duration=0.7, max_record_duration=10):
    frames = []
    ring_buffer = deque(maxlen=int(silence_duration * 1000 / CHUNK_DURATION_MS))
    start_time = time.time()
    speaking = False

    while True:
        chunk = stream.read(CHUNK_SIZE)
        is_speech = vad.is_speech(chunk, RATE)

        ring_buffer.append((chunk, is_speech))

        if is_speech:
            speaking = True
            frames.extend([c for c, _ in ring_buffer])
            ring_buffer.clear()
        elif speaking:
            # Check if silence continued long enough to stop
            if all(not speech for _, speech in ring_buffer):
                break

        if time.time() - start_time > max_record_duration:
            break

    return b''.join(frames)

def pcm_to_float32(audio_bytes):
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    return audio_np / 32768.0

def main():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    print("Speak anytime! (Ctrl+C to stop)")
    try:
        while True:
            print("\nListening for speech...")
            audio_bytes = record_until_silence(stream, silence_duration=0.7, max_record_duration=10)

            if len(audio_bytes) == 0:
                print("...Still listening (no speech detected).")
                continue
            print("Transcribing...")
            audio_float32 = pcm_to_float32(audio_bytes)
            result = asr_model.transcribe(audio_float32, language='en', fp16=True)
            prompt = result["text"].strip()

            if prompt:
                print(f"You said: {prompt}")
                print("AI is thinking...")
                response = query_ollama(prompt)
                print(f"AI: {response}")
            else:
                print("(Could not understand anything.)")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
