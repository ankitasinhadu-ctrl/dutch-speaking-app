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
- Never ask for a date of birth, address, BSN or other personal details. The patient's name is enough.

Rules for corrections:
- Check only the patient's latest message.
- Note grammar mistakes, wrong word choices and English words.
- Give each correction a type: "grammar", "word choice", "English word" or "spelling". Use "spelling" for spelling mistakes and for words that are wrongly split or joined.
- Only correct real mistakes. If the patient's sentence is correct Dutch, do not correct it, even if you would say it differently.
- Report each mistake only once.
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
                    "type": {"type": "string", "enum": ["grammar", "word choice", "English word", "spelling"]},
                },
                "required": ["original", "correct", "explanation", "type"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["reply", "corrections"],
    "additionalProperties": False,
}
def keep_correction(c):
    """Keep only real language mistakes; drop spelling and spacing, which usually come from speech recognition."""
    if c["type"] == "spelling":
        return False
    same_without_spaces = c["original"].replace(" ", "").lower() == c["correct"].replace(" ", "").lower()
    return not same_without_spaces
async def main():
    history = []          # the conversation so far
    all_corrections = []  # every correction from the whole session

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
            all_corrections.extend(c for c in data["corrections"] if keep_correction(c))
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