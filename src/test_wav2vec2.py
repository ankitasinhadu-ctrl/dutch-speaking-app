import time
import torch
from transformers import AutoProcessor, AutoModelForCTC
from faster_whisper import decode_audio
from g2p import text_to_phonemes  # also sets up eSpeak, which the model's processor needs

MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"

# What you said in each recording
references = {
    "data/test1.m4a": "Goedemorgen, ik wil graag een afspraak maken met de huisarts.",
    "data/test2.m4a": "Ik heb sinds gisteren koorts en ik moet veel hoesten.",
    "data/test3.m4a": "Ik ben verhuisd en ik wil mijn nieuwe adres doorgeven.",
}

print("Loading wav2vec2 (first time: about 1.2 GB download)...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCTC.from_pretrained(MODEL_ID)

def audio_to_phonemes(audio):
    """Return the sounds the learner actually produced."""
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():                        # we only use the model, not train it
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)  # most likely sound at each moment
    return processor.batch_decode(predicted_ids)[0]

for path, sentence in references.items():
    audio = decode_audio(path, sampling_rate=16000)
    start = time.time()
    heard = audio_to_phonemes(audio)
    print(path)
    print("  Expected:", text_to_phonemes(sentence))
    print("  Heard:   ", heard)
    print(f"  ({time.time() - start:.1f}s)\n")