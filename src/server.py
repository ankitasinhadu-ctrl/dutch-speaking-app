import io
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel, decode_audio

app = FastAPI(title="Praatmaatje")

# Load Whisper once, when the server starts
WHISPER_HINT = ("Een telefoongesprek met de doktersassistente van de huisarts. "
                "Afspraak maken, koorts, hoesten, klachten, twee dagen, vandaag, morgen.")
whisper = WhisperModel("base", device="cpu", compute_type="int8")

@app.get("/api/health")
def health():
    """A simple check that the server is running."""
    return {"status": "ok"}

@app.post("/api/transcribe")
def transcribe(audio: UploadFile = File(...)):
    """Receive a recording from the browser and return what Whisper heard."""
    data = audio.file.read()                                  # the recording, in memory only
    samples = decode_audio(io.BytesIO(data), sampling_rate=16000)
    segments, info = whisper.transcribe(samples, language="nl", beam_size=5,
                                        vad_filter=True, initial_prompt=WHISPER_HINT)
    text = " ".join(segment.text for segment in segments).strip()
    return {"text": text}

# Serve the web page (web/index.html) at the main address
app.mount("/", StaticFiles(directory="web", html=True), name="web")