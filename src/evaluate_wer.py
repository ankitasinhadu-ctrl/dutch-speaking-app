from faster_whisper import WhisperModel
import jiwer

# What you actually said in each recording (the "correct answer")
references = {
    "data/test1.m4a": "Goedemorgen, ik wil graag een afspraak maken met de huisarts.",
    "data/test2.m4a": "Ik heb sinds gisteren koorts en ik moet veel hoesten.",
    "data/test3.m4a": "Ik ben verhuisd en ik wil mijn nieuwe adres doorgeven.",
    "data/test4.m4a": "Welke documenten moet ik meenemen naar de afspraak?",
    "data/test5.m4a": "Ik werk graag in een team en ik ben morgen beschikbaar.",
}

model = WhisperModel("small", device="cpu", compute_type="int8")

# Make texts comparable: lowercase, no punctuation, no extra spaces
normalize = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])

all_refs, all_hyps = [], []

for path, reference in references.items():
    segments, info = model.transcribe(path, language="nl")
    hypothesis = " ".join(segment.text for segment in segments)

    ref = normalize(reference)
    hyp = normalize(hypothesis)
    wer = jiwer.wer(ref, hyp)

    print(path)
    print("  You said: ", ref)
    print("  Whisper:  ", hyp)
    print(f"  WER: {wer:.0%}\n")

    all_refs.append(ref)
    all_hyps.append(hyp)

print(f"Overall WER: {jiwer.wer(all_refs, all_hyps):.1%}")