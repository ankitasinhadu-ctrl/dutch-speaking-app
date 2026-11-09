import asyncio
import json
import time
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from runware import Runware

from chat_feedback import OPENING, SYSTEM_PROMPT, RESPONSE_SCHEMA, keep_correction  # reuse the prompt and JSON structure
from speak import text_to_audio, play                              # reuse Fenna's voice

load_dotenv()

MIC_RATE = 16000  # Whisper expects 16,000 sound samples per second

# Words Whisper should expect in this scenario (helps it hear "koorts" instead of "kort")
WHISPER_HINT = ("Een telefoongesprek met de doktersassistente van de huisarts. "
                "Afspraak maken, koorts, hoesten, klachten, twee dagen, vandaag, morgen.")

def record_until_enter():
    """Record from the microphone until the learner presses Enter."""
    chunks = []
    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())
    with sd.InputStream(samplerate=MIC_RATE, channels=1, dtype="float32", callback=callback):
        input("  Opnemen... druk op Enter als je klaar bent.")
    if not chunks:
        return np.zeros(0, dtype="float32")
    return np.concatenate(chunks).flatten()

async def main():
    print("Whisper laden...")
    whisper = WhisperModel("base", device="cpu", compute_type="int8")
    history, all_corrections = [], []

    async with Runware() as client:
        print("Assistent:", OPENING)
        play(await text_to_audio(OPENING))

        while True:
            command = input("\nEnter = spreken, q + Enter = stoppen: ")
            if command.strip().lower() == "q":
                break

            audio = record_until_enter()
            if len(audio) < MIC_RATE * 0.5:  # shorter than half a second
                print("Te kort. Probeer het nog eens.")
                continue

            # 1. Speech to text
            t0 = time.time()
            segments, info = whisper.transcribe(audio, language="nl", beam_size=5,
            vad_filter=True, initial_prompt=WHISPER_HINT)
            user_text = " ".join(segment.text for segment in segments).strip()
            t1 = time.time()
            print("Jij (Whisper):", user_text)
            if not user_text:
                print("Ik heb niets gehoord. Probeer het nog eens.")
                continue

            # 2. LLM: reply in role + grammar corrections
            history.append({"role": "user", "content": user_text})
            results = await client.run({
                "model": "google:gemini@3.1-flash-lite",
                "outputFormat": "JSON",
                "jsonSchema": RESPONSE_SCHEMA,
                "settings": {
                    "systemPrompt": SYSTEM_PROMPT,
                    "thinkingLevel": "minimal",
                    "maxTokens": 400,
                },
                "messages": history,
            })
            data = json.loads(results[0]["text"])
            reply = data["reply"]
            history.append({"role": "assistant", "content": reply})
            all_corrections.extend(data["corrections"])
            t2 = time.time()

            # 3. Text to speech
            reply_audio = await text_to_audio(reply)
            t3 = time.time()

            print("Assistent:", reply)
            print(f"  [Whisper {t1 - t0:.1f}s | LLM {t2 - t1:.1f}s | Voice {t3 - t2:.1f}s | Total wait {t3 - t0:.1f}s]")
            play(reply_audio)

    # End-of-session report
    print("\n===== Sessierapport =====")
    if not all_corrections:
        print("Geen fouten gevonden. Goed gedaan!")
    for c in all_corrections:
        print(f"- You said: {c['original']}")
        print(f"  Correct:  {c['correct']}")
        print(f"  Why:      {c['explanation']}\n")

asyncio.run(main())