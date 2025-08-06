import time
import pyttsx3
import requests
import json
import re

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

def speak_text(text):
    """Speak a single piece of text using a fresh engine instance."""
    try:
        # Create a fresh engine instance for each speech
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.setProperty("volume", 1.0)
        
        # Set voice if available
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)
        
        engine.say(text)
        engine.runAndWait()
        
        # Clean up the engine
        engine.stop()
        del engine
        
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

def speak_stream_alternative(prompt):
    """Alternative approach: Collect all text first, then speak by sentences."""
    print("🤖 AI Response:")
    print("-" * 50)
    
    # Collect all text first
    full_text = ""
    for chunk in query_ollama_stream(prompt):
        if chunk and not chunk.startswith("[Network Error]"):
            print(chunk, end='', flush=True)
            full_text += chunk
        elif chunk.startswith("[Network Error]"):
            print(f"\n❌ {chunk}")
            speak_text("Sorry, there was a network error.")
            return
    
    print("\n" + "-" * 30)
    
    # Now split into sentences and speak each one
    if full_text.strip():
        # Split by sentence boundaries
        sentences = re.split(r'([.!?]+\s*)', full_text)
        
        current_sentence = ""
        for i, part in enumerate(sentences):
            if re.match(r'[.!?]+\s*', part):  # This is punctuation
                current_sentence += part
                if current_sentence.strip():
                    print(f"🗣️  Speaking: {current_sentence.strip()}")
                    speak_text(current_sentence.strip())
                    time.sleep(0.2)
                current_sentence = ""
            else:  # This is text
                current_sentence += part
        
        # Speak any remaining text without punctuation
        if current_sentence.strip():
            print(f"🗣️  Speaking: {current_sentence.strip()}")
            speak_text(current_sentence.strip())
    
    print("✅ Done speaking!")

def interactive_mode():
    """Run in interactive mode for multiple conversations."""
    print("🎤 Interactive AI Voice Chat")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'alt' to use alternative speaking mode")
    print("=" * 50)
    
    use_alternative = False
    
    while True:
        try:
            user_input = input("\n💬 Your message: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'alt':
                use_alternative = not use_alternative
                mode = "alternative" if use_alternative else "streaming"
                print(f"🔄 Switched to {mode} mode")
                continue
            
            if not user_input:
                print("Please enter a message.")
                continue
            
            print()
            if use_alternative:
                speak_stream_alternative(user_input)
            else:
                speak_stream(user_input)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

# Example usage
if __name__ == "__main__":
    print("🎯 Choose your mode:")
    print("1. Streaming mode (speaks sentences as they complete)")
    print("2. Alternative mode (collects all text first, then speaks)")
    print("3. Interactive mode")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        speak_stream("Tell me a short story about a robot learning to cook.")
    elif choice == "2":
        speak_stream_alternative("Tell me a short story about a robot learning to cook.")
    elif choice == "3":
        interactive_mode()
    else:
        print("Using streaming mode by default...")
        speak_stream("Tell me a short story about a robot learning to cook.")