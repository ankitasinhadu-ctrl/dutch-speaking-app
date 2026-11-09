import asyncio
import time
from dotenv import load_dotenv
from runware import Runware

load_dotenv()  # reads RUNWARE_API_KEY from the .env file

OPENING = "Huisartsenpraktijk De Linde, goedemorgen. Waarmee kan ik u helpen?"

SYSTEM_PROMPT = (
    "Je bent een doktersassistente bij huisartsenpraktijk De Linde. "
    "Spreek eenvoudig Nederlands op A2-niveau. "
    "Antwoord in maximaal twee korte zinnen. Spreek nooit Engels. "
    "Help de patiënt om een afspraak te maken. "
    f"Je hebt de patiënt al begroet met: '{OPENING}'"
)

async def main():
    history = []  # everything said so far in this conversation

    async with Runware() as client:
        print("Typ je antwoord in het Nederlands. Typ 'stop' om te stoppen.\n")
        print("Assistent:", OPENING)

        while True:
            user_text = input("Jij: ")
            if user_text.strip().lower() == "stop":
                break

            history.append({"role": "user", "content": user_text})

            start = time.time()
            stream = await client.stream({
                "model": "google:gemini@3.1-flash-lite",
                "settings": {
                    "systemPrompt": SYSTEM_PROMPT,
                    "thinkingLevel": "minimal",
                    "maxTokens": 200,
                },
                "messages": history,
            })
            reply = await stream.text()

            history.append({"role": "assistant", "content": reply})
            print(f"Assistent: {reply}   ({time.time() - start:.1f}s)\n")

asyncio.run(main())