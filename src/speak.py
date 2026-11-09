import asyncio
import io
import time
import edge_tts
import sounddevice as sd
from faster_whisper import decode_audio

VOICE = "nl-NL-FennaNeural"
SAMPLE_RATE = 24000  # the edge-tts voices use 24,000 sound samples per second

async def text_to_audio(text):
    """Turn Dutch text into speech audio, without saving a file."""
    mp3_bytes = b""
    async for chunk in edge_tts.Communicate(text, VOICE).stream():
        if chunk["type"] == "audio":
            mp3_bytes += chunk["data"]
    return decode_audio(io.BytesIO(mp3_bytes), sampling_rate=SAMPLE_RATE)

def play(audio):
    """Play the audio through the speakers and wait until it's finished."""
    sd.play(audio, SAMPLE_RATE)
    sd.wait()

async def main():
    start = time.time()
    audio = await text_to_audio("Huisartsenpraktijk De Linde, goedemorgen. Waarmee kan ik u helpen?")
    print(f"Voice ready in {time.time() - start:.1f}s, now playing...")
    play(audio)

if __name__ == "__main__":
    asyncio.run(main())