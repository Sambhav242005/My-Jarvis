import os
import time
import pyaudio
import numpy as np
import whisper
import requests
import webrtcvad
from collections import deque
import torch
from tts import speak_stream

# Audio and model config
RATE = 16000
CHUNK_DURATION_MS = 30  # 30ms per chunk
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)
CHANNELS = 1

# Silence detection thresholds
MIN_AUDIO_LENGTH = 0.5  # Minimum seconds of audio to process
MIN_VOLUME_THRESHOLD = 0.005  # Minimum RMS volume threshold
MIN_SPEECH_CHUNKS = 10  # Minimum chunks that must be detected as speech
SILENCE_DURATION = 1.0  # Seconds of silence to stop recording


# Whisper and Ollama
asr_model = whisper.load_model("base").to("cuda")

# VAD setup
vad = webrtcvad.Vad()
vad.set_mode(2)  # Increased sensitivity (0=least, 3=most)

def record_until_silence(stream, silence_duration=SILENCE_DURATION, max_record_duration=10):
    frames = []
    ring_buffer = deque(maxlen=int(silence_duration * 1000 / CHUNK_DURATION_MS))
    start_time = time.time()
    speaking = False
    speech_chunk_count = 0
    total_chunks = 0

    print("🎤 Ready to record...")

    while True:
        try:
            chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        except Exception as e:
            print(f"Audio read error: {e}")
            break

        total_chunks += 1

        # Check if chunk contains speech
        try:
            is_speech = vad.is_speech(chunk, RATE)
        except Exception :
            is_speech = False

        ring_buffer.append((chunk, is_speech))

        if is_speech:
            speech_chunk_count += 1
            if not speaking:
                print("🗣️ Speech detected...")
                speaking = True
                # Add pre-speech buffered audio
                frames.extend([c for c, _ in ring_buffer])
            frames.append(chunk)  # Always add the current speech chunk
        else:
            if speaking:
                frames.append(chunk)  # Might just be a short pause
                # If the entire buffer is silence for required duration, stop
                if all(not speech for _, speech in ring_buffer):
                    print("🔇 Silence detected, stopping...")
                    break

        # Safety check for maximum duration
        if time.time() - start_time > max_record_duration:
            print("⏰ Max recording duration reached")
            break

    # Quality checks before returning audio
    if not speaking:
        print("❌ No speech detected")
        return b''

    if speech_chunk_count < MIN_SPEECH_CHUNKS:
        print(f"❌ Insufficient speech chunks ({speech_chunk_count} < {MIN_SPEECH_CHUNKS})")
        return b''

    duration = len(frames) * CHUNK_DURATION_MS / 1000
    if duration < MIN_AUDIO_LENGTH:
        print(f"❌ Audio too short ({duration:.2f}s < {MIN_AUDIO_LENGTH}s)")
        return b''

    print(f"✅ Recorded {duration:.2f}s with {speech_chunk_count} speech chunks")
    return b''.join(frames)

def pcm_to_float32(audio_bytes):
    if len(audio_bytes) == 0:
        return np.array([], dtype=np.float32)

    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    audio_float = audio_np / 32768.0

    # Check volume level
    rms_volume = np.sqrt(np.mean(audio_float**2))
    print(f"🔊 Audio RMS volume: {rms_volume:.4f}")

    if rms_volume < MIN_VOLUME_THRESHOLD:
        print(f"❌ Audio volume too low ({rms_volume:.4f} < {MIN_VOLUME_THRESHOLD})")
        return np.array([], dtype=np.float32)

    return audio_float

def main():
    p = pyaudio.PyAudio()

    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        print("🎙️  Voice Assistant Started!")
        print("📋 Settings:")
        print(f"   - Minimum audio length: {MIN_AUDIO_LENGTH}s")
        print(f"   - Silence threshold: {SILENCE_DURATION}s")
        print(f"   - Volume threshold: {MIN_VOLUME_THRESHOLD}")
        print(f"   - Speech chunks required: {MIN_SPEECH_CHUNKS}")
        print("\n💬 Speak anytime! (Ctrl+C to stop)\n")

        while True:
            print("👂 Listening for speech...")
            audio_bytes = record_until_silence(stream)

            if len(audio_bytes) == 0:
                print("⏭️  No valid audio detected, continuing to listen...\n")
                continue

            print("🔄 Converting audio...")
            audio_float32 = pcm_to_float32(audio_bytes)

            if len(audio_float32) == 0:
                print("⏭️  Audio failed volume check, continuing to listen...\n")
                continue

            print("📝 Transcribing...")
            try:
                result = asr_model.transcribe(
                    audio_float32,
                    language='en',
                    fp16=torch.cuda.is_available(),
                    condition_on_previous_text=False,  # Avoid context bleeding
                    temperature=0.0  # More deterministic output
                )
                prompt = result["text"].strip()

                print(f"🎯 Raw transcription: '{prompt}'")


            except Exception as e:
                print(f"❌ Transcription error: {e}\n")
                continue

            if prompt:
                print(f"✅ You said: {prompt}")
                print("🤖 AI is thinking...")

                try:
                    response = speak_stream(prompt)
                    print(f"💭 AI: {response}\n")
                except Exception as e:
                    print(f"❌ AI response error: {e}\n")
            else:
                print("❌ Empty transcription result\n")

    except KeyboardInterrupt:
        print("\n👋 Voice assistant stopped.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
