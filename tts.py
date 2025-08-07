import time
import requests
import json
import re
import sounddevice as sd
from TTS.api import TTS

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"


def query_ollama_stream(prompt, model=OLLAMA_MODEL):
    """Stream responses from Ollama API."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if line:
                    try:
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        if chunk:
                            yield chunk
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
    except requests.exceptions.RequestException as e:
        yield f"[Network Error]: {e}"

def speak_text(text: str):
    """Speak a single piece of text using Coqui TTS on GPU."""
    try:
        # Load model fresh (avoids keeping VRAM occupied)
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=True,progress_bar=False)

        # Generate waveform
        waveform = tts.tts(text=text)

        # Play
        sd.play(waveform, samplerate=22050)
        sd.wait()  # Wait until playback ends

        # Free up model from memory
        del tts

    except Exception as e:
        print(f"\n🔇 Speech error: {e}")


def speak_stream(prompt):
    """Stream Ollama output and speak complete sentences naturally."""
    accumulated_text = ""
    
    print("🤖 AI Response:")
    print("-" * 50)
    
    for chunk in query_ollama_stream(prompt):
        if chunk and not chunk.startswith("[Network Error]"):
            # Print chunk to console immediately
            print(chunk, end='', flush=True)
            accumulated_text += chunk
            
            # Check for sentence endings
            if any(punct in chunk for punct in '.!?'):
                # Find complete sentences
                sentence_pattern = r'([^.!?]*[.!?]+)'
                sentences = re.findall(sentence_pattern, accumulated_text)
                
                if sentences:
                    # Speak each complete sentence
                    for sentence in sentences:
                        clean_sentence = sentence.strip()
                        if len(clean_sentence) > 3:
                            speak_text(clean_sentence)
                            time.sleep(0.1)  # Brief pause between sentences
                    
                    # Remove spoken sentences from buffer
                    for sentence in sentences:
                        accumulated_text = accumulated_text.replace(sentence, "", 1)
                    accumulated_text = accumulated_text.strip()
        
        elif chunk.startswith("[Network Error]"):
            print(f"\n❌ {chunk}")
            speak_text("Sorry, there was a network error connecting to the AI.")
            return
    
    # Speak any remaining text
    if accumulated_text.strip():
        print()
        clean_remaining = accumulated_text.strip()
        if len(clean_remaining) > 2:
            speak_text(clean_remaining)
    
    print("\n" + "-" * 50)
    print("✅ Done speaking!")


# Example usage
if __name__ == "__main__":
    speak_stream("Tell be about the benefits of AI in healthcare.")