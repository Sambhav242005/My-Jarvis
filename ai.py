import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3n"

def query_ollama_stream(prompt, model=OLLAMA_MODEL, max_tokens=120):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,  # ✅ enable streaming
        "options": {"num_predict": max_tokens}
    }
    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=(5, 300)) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if line:  # skip keep-alive new lines
                    try:
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        if chunk:
                            yield chunk  # send partial output to caller
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
    except requests.exceptions.RequestException as e:
        yield f"[Network Error]: {e}"



# Example usage:
if __name__ == "__main__":
    for token in query_ollama_stream("Tell me a short joke about AI."):
        print(token, end="", flush=True)
    print("\n--- Done ---")
