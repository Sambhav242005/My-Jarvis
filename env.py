import os
from dotenv import load_dotenv

load_dotenv()  # loads variables from .env

# Auto-convert into Python variables (dictionary unpacking)
vars_dict = {key: os.getenv(key) for key in ["OLLAMA_URL", "OLLAMA_MODEL", "TTS_MODEL", "WHISPER_MODEL"] if os.getenv(key) is not None}

# Assign dynamically to globals
globals().update(vars_dict)
