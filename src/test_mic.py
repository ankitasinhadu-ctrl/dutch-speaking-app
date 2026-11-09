import time
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000  # Whisper expects 16,000 sound samples per second
SECONDS = 5          # how long to record

print("Loading Whisper...")
model = WhisperModel("small", device="cpu", compute_type="int8")

input("Press Enter, then speak Dutch for 5 seconds...")
print("Recording...")
audio = sd.rec(int(SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
sd.wait()  # wait until the 5 seconds are over
print("Done. Transcribing...")

start = time.time()
segments, info = model.transcribe(audio.flatten(), language="nl")
text = " ".join(segment.text for segment in segments).strip()
print("Whisper heard:", text)
print(f"Transcription time: {time.time() - start:.1f}s")