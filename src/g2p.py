import espeakng_loader
from phonemizer import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer.separator import Separator

# Use the eSpeak engine bundled with espeakng-loader (no separate install needed)
EspeakWrapper.set_library(espeakng_loader.get_library_path())
EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

def text_to_phonemes(text):
    """Return the sounds a native Dutch speaker would produce.
    Sounds are separated by spaces, words by ' | '."""
    result = phonemize(text, language="nl", backend="espeak", strip=True,
                       separator=Separator(phone=" ", word=" | "))
    return result.strip()

if __name__ == "__main__":
    examples = ["huis", "goedemorgen", "koorts", "deur", "tijd", "uur",
                "Ik heb koorts sinds twee dagen."]
    for text in examples:
        print(f"{text:35} -> {text_to_phonemes(text)}")