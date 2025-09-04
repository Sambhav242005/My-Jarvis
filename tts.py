# speech.py
import torch
import re
import sounddevice as sd
import threading
import queue
from TTS.api import TTS
from env import TTS_MODEL
from ai import query_ollama_stream

# Init TTS
tts = TTS(model_name=TTS_MODEL, progress_bar=False)
tts.to("cuda" if torch.cuda.is_available() else "cpu")

# Playback queue
audio_queue = queue.Queue()

def audio_player():
    """Background thread to play audio chunks without blocking."""
    while True:
        waveform = audio_queue.get()
        if waveform is None:  # Exit signal
            break
        sd.play(waveform, samplerate=22050)
        sd.wait()
        audio_queue.task_done()

# Start background player
threading.Thread(target=audio_player, daemon=True).start()

def clean_text(text: str) -> str:
    text = re.sub(r"[*_`~]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def enqueue_speech(text: str):
    """Generate TTS and enqueue audio for playback."""
    try:
        waveform = tts.tts(text=text, speaker="p225", return_type="np")
        audio_queue.put(waveform)
    except Exception as e:
        print(f"\n🔇 Speech error: {e}")

def speak_sentence(sentence: str):
    """Directly speak a single sentence."""
    enqueue_speech(clean_text(sentence))

def shutdown():
    """Gracefully stop audio thread after all queued speech finishes."""
    audio_queue.join()
    audio_queue.put(None)

def speak_stream(prompt):
    accumulated_text = ""
    print("🤖 AI Response:")
    print("-" * 50)

    for chunk in query_ollama_stream(prompt):
        if chunk and not chunk.startswith("[Network Error]"):
            print(chunk, end='', flush=True)
            accumulated_text += chunk

            # Detect full sentences
            import re
            sentences = re.findall(r'[^.!?]+[.!?]', accumulated_text)
            for sentence in sentences:
                speak_sentence(sentence)
                accumulated_text = accumulated_text.replace(sentence, "", 1).strip()

        elif chunk.startswith("[Network Error]"):
            print(f"\n❌ {chunk}")
            speak_sentence("Sorry, there was a network error connecting to the AI.")
            return

    if accumulated_text.strip():
        speak_sentence(accumulated_text.strip())

    print("\n" + "-" * 50)
    print("✅ Done speaking!")

