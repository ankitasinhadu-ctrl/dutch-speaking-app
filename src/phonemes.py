import torch
from transformers import AutoProcessor, AutoModelForCTC
import g2p  # sets up eSpeak, which the wav2vec2 processor needs

MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
_processor = None
_model = None

def load():
    """Load wav2vec2 only once, the first time it is needed."""
    global _processor, _model
    if _model is None:
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForCTC.from_pretrained(MODEL_ID)
        _model.eval()

def audio_to_phonemes(audio):
    """Return the sounds the learner actually produced (16 kHz audio)."""
    load()
    inputs = _processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = _model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return _processor.batch_decode(predicted_ids)[0]