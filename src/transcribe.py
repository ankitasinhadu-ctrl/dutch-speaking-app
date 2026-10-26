from faster_whisper import WhisperModel

# Load the "small" Whisper model (downloads once, about 500 MB)
model = WhisperModel("small", device="cpu", compute_type="int8")

# Transcribe the recording, telling Whisper the language is Dutch
segments, info = model.transcribe("data/test1.m4a", language="nl")

# Print what Whisper heard
for segment in segments:
    print(segment.text)