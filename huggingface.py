import os
import time
import pyaudio
import numpy as np
import whisper
import webrtcvad
import torch
from collections import deque
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    # BitsAndBytesConfig  # uncomment if you want 4-bit quant
)

# ─── TUNE TORCH FOR PERFORMANCE ───────────────────────────────────────────────
os.environ["TORCH_COMPILE_DISABLE"] = "1"
torch._dynamo.disable()  # disable torch.compile
torch.set_float32_matmul_precision('medium')  # use TF32 where possible

# ─── AUDIO CONFIG ────────────────────────────────────────────────────────────
RATE = 16000
CHUNK_MS = 30                   # 30 ms per frame
CHUNK_SIZE = int(RATE * CHUNK_MS / 1000)
SILENCE_SEC = 0.7               # stop after ~0.7 s of silence
MAX_UTTERANCE_SEC = 10          # hard limit
CHANNELS = 1

# ─── LOAD MODELS ─────────────────────────────────────────────────────────────
# Whisper ASR
asr_model = whisper.load_model("tiny").to("cuda")

# Hugging Face Gemma 3n-it (float16)
MODEL_ID = "google/gemma-3n-E2B-it"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )
# model.eval()

# ─── OPTIONAL: 4-BIT QUANTIZATION ────────────────────────────────────────────
# If you want to drop VRAM to ~5–6 GB, comment out the above AutoModelForCausalLM
# and uncomment below (requires bitsandbytes & accelerate):
#
from transformers import BitsAndBytesConfig
bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_state_dict=True       
)
model.eval()

if hasattr(model, "altup") and hasattr(model.altup, "correct"):
    model.altup.correct = lambda predictions, attn_ffw_laurel_gated: predictions

# ─── SET UP VAD ──────────────────────────────────────────────────────────────
vad = webrtcvad.Vad(1)  # 0 = less aggressive (detect more speech), 3 = most aggressive

def record_until_silence(stream):
    """Record raw PCM chunks until silence of SILENCE_SEC or MAX_UTTERANCE_SEC reached."""
    ring = deque(maxlen=int(SILENCE_SEC * 1000 / CHUNK_MS))
    frames = []
    start = time.time()
    saw_speech = False

    while True:
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        is_speech = vad.is_speech(data, RATE)
        ring.append((data, is_speech))

        if is_speech:
            saw_speech = True
            # flush ring buffer into frames
            while ring:
                frames.append(ring.popleft()[0])
        elif saw_speech:
            # if all in ring are silence, break
            if not any(s for _, s in ring):
                break

        # safety cutoff
        if time.time() - start > MAX_UTTERANCE_SEC:
            break

    return b"".join(frames)

def pcm_to_float32(pcm_bytes):
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return arr / 32768.0

def generate_response(text, max_new_tokens=120):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    input_ids = input_ids.to("cuda")
    with torch.no_grad():
        gen_tokens = model.generate(input_ids, max_new_tokens=max_new_tokens)
    return tokenizer.batch_decode(gen_tokens)[0]


def main():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    print("🎙️  Speak whenever – recording will auto-stop on silence. (Ctrl+C to quit)")
    try:
        while True:
            print("\n⏳ Listening…")
            pcm = record_until_silence(stream)

            if not pcm:
                # no speech detected in max utterance window
                continue

            audio = pcm_to_float32(pcm)
            print("📝 Transcribing…")
            result = asr_model.transcribe(audio, language='en', fp16=True)
            prompt = result["text"].strip()

            if not prompt:
                print("(Could not parse speech.)")
                continue

            print(f"You said: {prompt}")
            print("💡 Generating response…")
            reply = generate_response(prompt)
            print(f"AI: {reply}")

    except KeyboardInterrupt:
        print("\n👋 Stopped by user.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
