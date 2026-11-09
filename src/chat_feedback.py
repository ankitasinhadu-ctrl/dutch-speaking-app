import asyncio
import json
import time
from dotenv import load_dotenv
from runware import Runware

load_dotenv()  # reads RUNWARE_API_KEY from the .env file

OPENING = "Huisartsenpraktijk De Linde, goedemorgen. Waarmee kan ik u helpen?"

SYSTEM_PROMPT = f"""You are a receptionist (doktersassistente) at huisartsenpraktijk De Linde in the Netherlands.
You help a Dutch learner (level A2) practise a phone call to make an appointment.
You already greeted the patient with: "{OPENING}"

Rules for your reply:
- Always reply in simple Dutch at A2 level, maximum two short sentences.
- Never switch to English, even if the patient uses English.
- Stay in your role and help the patient make an appointment.
- If the patient made a mistake, naturally use the correct form in your reply, without pointing out the mistake.

Rules for corrections:
- Check only the patient's latest message.
- Note grammar mistakes, wrong word choices and English words.
- Ignore spelling, capital letters and punctuation, because the text will later come from speech recognition.
- If there are no mistakes, return an empty list.
"""

# The fixed structure the LLM must answer in
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string", "description": "Your next line in the conversation, in Dutch"},
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "original": {"type": "string"},
                    "correct": {"type": "string"},
                    "explanation": {"type": "string", "description": "Short explanation in simple English"},
                },
                "required": ["original", "correct", "explanation"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["reply", "corrections"],
    "additionalProperties": False,
}

async def main():
    history = []          # the conversation so far
    all_corrections = []  # collected for the end-of-session report

    async with Runware() as client:
        print("Typ je antwoord in het Nederlands. Typ 'stop' om te stoppen.\n")
        print("Assistent:", OPENING)

        while True:
            user_text = input("Jij: ")
            if user_text.strip().lower() == "stop":
                break

            history.append({"role": "user", "content": user_text})

            start = time.time()
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
            data = json.loads(results[0]["text"])  # turn the JSON text into a Python dictionary

            reply = data["reply"]
            history.append({"role": "assistant", "content": reply})
            all_corrections.extend(data["corrections"])
            print(f"Assistent: {reply}   ({time.time() - start:.1f}s)\n")

    # End-of-session report
    print("\n===== Sessierapport =====")
    if not all_corrections:
        print("Geen fouten gevonden. Goed gedaan!")
    for c in all_corrections:
        print(f"- You said: {c['original']}")
        print(f"  Correct:  {c['correct']}")
        print(f"  Why:      {c['explanation']}\n")

if __name__ == "__main__":
    asyncio.run(main())