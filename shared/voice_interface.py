import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
from gtts import gTTS
import os
from playsound import playsound

# Load Whisper model once
model = whisper.load_model("base")  

def record_audio(duration=5, samplerate=44100):
    print("🎙️ Listening...")
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1)
    sd.wait()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        wav.write(f.name, samplerate, np.int16(audio * 32767))
        return f.name

def transcribe_audio(file_path):
    print("🔍 Transcribing...")
    result = model.transcribe(file_path)
    return result["text"]

def speak(text):
    print(f"🗣️ Speaking: {text}")
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts.save(f.name)
        playsound(f.name)
        os.unlink(f.name)
