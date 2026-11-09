import time
import jiwer
from faster_whisper import WhisperModel

# What you actually said in each recording
references = {
    "data/test1.m4a": "Goedemorgen, ik wil graag een afspraak maken met de huisarts.",
    "data/test2.m4a": "Ik heb sinds gisteren koorts en ik moet veel hoesten.",
    "data/test3.m4a": "Ik ben verhuisd en ik wil mijn nieuwe adres doorgeven.",
    "data/test4.m4a": "Welke documenten moet ik meenemen naar de afspraak?",
    "data/test5.m4a": "Ik werk graag in een team en ik ben morgen beschikbaar.",
}

normalize = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])

# (model size, beam size) combinations to compare
CONFIGS = [("small", 5), ("small", 1), ("base", 5), ("base", 1)]

for model_name, beam in CONFIGS:
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    refs, hyps, times = [], [], []

    for path, reference in references.items():
        start = time.time()
        segments, info = model.transcribe(path, language="nl", beam_size=beam)
        hypothesis = " ".join(segment.text for segment in segments)
        times.append(time.time() - start)

        refs.append(normalize(reference))
        hyps.append(normalize(hypothesis))

    wer = jiwer.wer(refs, hyps)
    avg_time = sum(times) / len(times)
    print(f"{model_name:6} beam={beam}:  WER {wer:.1%}   average {avg_time:.1f}s per sentence")