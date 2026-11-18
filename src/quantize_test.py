import time
import torch
from transformers import AutoProcessor, AutoModelForCTC
from faster_whisper import decode_audio
from g2p import text_to_phonemes  # also sets up eSpeak for the processor

MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"

references = {
    "data/test1.m4a": "Goedemorgen, ik wil graag een afspraak maken met de huisarts.",
    "data/test2.m4a": "Ik heb sinds gisteren koorts en ik moet veel hoesten.",
    "data/test3.m4a": "Ik ben verhuisd en ik wil mijn nieuwe adres doorgeven.",
}

print("Loading wav2vec2...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCTC.from_pretrained(MODEL_ID)
model.eval()

print("Making a quantized (8-bit) copy...")
model_q = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

def audio_to_phonemes(m, audio):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = m(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]

for path, sentence in references.items():
    audio = decode_audio(path, sampling_rate=16000)
    seconds = len(audio) / 16000

    t = time.time()
    normal = audio_to_phonemes(model, audio)
    t_normal = time.time() - t

    t = time.time()
    quant = audio_to_phonemes(model_q, audio)
    t_quant = time.time() - t

    print(f"{path} ({seconds:.1f}s of audio)")
    print("  Expected: ", text_to_phonemes(sentence))
    print(f"  Normal:    {normal}   [{t_normal:.1f}s]")
    print(f"  Quantized: {quant}   [{t_quant:.1f}s]\n")