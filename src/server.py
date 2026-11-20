import io
import edge_tts
from fastapi import FastAPI, UploadFile, File, Response
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel, decode_audio
from pydantic import BaseModel

app = FastAPI(title="Praatmaatje")

# Load Whisper once, when the server starts
WHISPER_HINT = ("Een telefoongesprek met de doktersassistente van de huisarts. "
                "Afspraak maken, koorts, hoesten, klachten, twee dagen, vandaag, morgen.")
whisper = WhisperModel("base", device="cpu", compute_type="int8")

# Dutch voices the page may ask for
VOICES = {"nl-NL-FennaNeural", "nl-NL-ColetteNeural"}

class SpeakRequest(BaseModel):
    text: str
    voice: str = "nl-NL-FennaNeural"

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

@app.post("/api/speak")
async def speak(request: SpeakRequest):
    """Turn the assistant's Dutch text into speech (MP3) for the browser to play."""
    voice = request.voice if request.voice in VOICES else "nl-NL-FennaNeural"
    mp3 = b""
    async for chunk in edge_tts.Communicate(request.text[:500], voice).stream():
        if chunk["type"] == "audio":
            mp3 += chunk["data"]
    return Response(content=mp3, media_type="audio/mpeg")

# Serve the web page (web/index.html) at the main address
app.mount("/", StaticFiles(directory="web", html=True), name="web")