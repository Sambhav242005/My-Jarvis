import requests
import json
from env import OLLAMA_URL, OLLAMA_MODEL

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

# Example usage:
if __name__ == "__main__":
    for token in query_ollama_stream("Tell me a short joke about AI."):
        print(token, end="", flush=True)
    print("\n--- Done ---")
