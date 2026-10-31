import asyncio
import time
from dotenv import load_dotenv
from runware import Runware

load_dotenv()  # reads RUNWARE_API_KEY from the .env file

SYSTEM_PROMPT = (
    "Je bent een doktersassistente bij huisartsenpraktijk De Linde. "
    "Spreek eenvoudig Nederlands op A2-niveau. "
    "Antwoord in maximaal twee korte zinnen. Spreek nooit Engels."
)

async def main():
    async with Runware() as client:
        start = time.time()
        stream = await client.stream({
            "model": "google:gemma@4-31b",
            "settings": {"systemPrompt": SYSTEM_PROMPT},
            "messages": [
                {"role": "user", "content": "Goedemorgen, ik wil graag een afspraak maken met de huisarts."},
            ],
        })
        reply = await stream.text()
        print("Assistant:", reply)
        print(f"Response time: {time.time() - start:.1f} seconds")

asyncio.run(main())