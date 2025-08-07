from TTS.api import TTS
import soundfile as sf
import sounddevice as sd

# Load a small, fast English model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=True)

# Generate waveform
waveform = tts.tts(text="This is a test of the TTS system.")

# Play audio
sd.play(waveform, samplerate=22050)
sd.wait()  # Wait until playback is finished
