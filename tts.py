import time
import requests
import json
import re
import sounddevice as sd
from TTS.api import TTS
from ai import query_ollama_stream

tts = TTS(model_name="tts_models/en/vctk/vits", gpu=True,progress_bar=False)


def speak_text(text: str):
    """Speak a single piece of text using Coqui TTS on GPU."""
    try:
        # Load model fresh (avoids keeping VRAM occupied)

        # Generate waveform
        waveform = tts.tts(text=text,speaker="p225", return_type="np")

        # Play
        sd.play(waveform, samplerate=22050)
        sd.wait()  # Wait until playback ends

    except Exception as e:
        print(f"\n🔇 Speech error: {e}")

def clean_text(text: str) -> str:
    """Remove unwanted symbols, markdown, and extra spaces."""
    # Remove markdown-style asterisks, underscores, backticks
    text = re.sub(r"[*_`~]", "", text)

    # Remove excessive spaces
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    return text.strip()

def speak_stream(prompt):
    """Stream Ollama output, detect full sentences, and speak cleanly."""
    # Modified prompt for Ollama to produce TTS-friendly text
    tts_prompt = (
        "Respond in plain text only. "
        "Do not use markdown, asterisks, underscores, or any special formatting. "
        "Write in complete sentences suitable for speech. "
        f"{prompt}"
    )

    accumulated_text = ""

    print("🤖 AI Response:")
    print("-" * 50)

    for chunk in query_ollama_stream(tts_prompt):
        if chunk and not chunk.startswith("[Network Error]"):
            print(chunk, end='', flush=True)
            accumulated_text += chunk

            # Detect completed sentences
            sentence_pattern = r'[^.!?]+[.!?]'
            sentences = re.findall(sentence_pattern, accumulated_text)

            if sentences:
                for sentence in sentences:
                    clean_sentence = clean_text(sentence)
                    if len(clean_sentence) > 3:
                        speak_text(clean_sentence)
                        time.sleep(0.05)

                for sentence in sentences:
                    accumulated_text = accumulated_text.replace(sentence, "", 1)
                accumulated_text = accumulated_text.strip()

        elif chunk.startswith("[Network Error]"):
            print(f"\n❌ {chunk}")
            speak_text("Sorry, there was a network error connecting to the AI.")
            return

    # Speak leftover
    if accumulated_text.strip():
        clean_remaining = clean_text(accumulated_text.strip())
        if len(clean_remaining) > 2:
            speak_text(clean_remaining)

    print("\n" + "-" * 50)
    print("✅ Done speaking!")


# Example usage
if __name__ == "__main__":
    speak_stream("Tell be about the benefits of AI in healthcare.")
