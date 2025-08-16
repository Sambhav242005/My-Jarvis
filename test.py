from TTS.api import TTS
import sounddevice as sd

# Load a small, fast English model
tts = TTS(model_name="tts_models/en/vctk/vits", gpu=True)

# Generate waveform and get the sample rate
waveform = tts.tts(text="This is a test of the TTS system.", speaker="p225", return_type="np")

# Play audio
sd.play(waveform, samplerate=22050)
sd.wait()  # Wait until playback is finished
