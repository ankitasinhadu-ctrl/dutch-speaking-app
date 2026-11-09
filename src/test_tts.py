import asyncio
import time
import edge_tts

TEXT = "Huisartsenpraktijk De Linde, goedemorgen. Waarmee kan ik u helpen?"

# Three Dutch voices to compare
VOICES = ["nl-NL-FennaNeural", "nl-NL-ColetteNeural", "nl-NL-MaartenNeural"]

async def main():
    for voice in VOICES:
        path = f"data/tts_{voice}.mp3"
        start = time.time()
        await edge_tts.Communicate(TEXT, voice).save(path)
        print(f"{voice}: made in {time.time() - start:.1f}s -> {path}")

asyncio.run(main())