from pronunciation import compare, summarise

# wav2vec2 output from your own recordings (copied from test_wav2vec2.py)
samples = {
    "Goedemorgen, ik wil graag een afspraak maken met de huisarts.":
        "x ʊ d ə m ɔ r x ə n ɪ k v ɪ l x r a x ɛ n d f s p ɹ a k m aː k ə n m ɪ t d ə h aʊ z a t s",
    "Ik heb sinds gisteren koorts en ik moet veel hoesten.":
        "ɪ k h ɛ b s ɛ n s x ɪ s t r ə n k oː t s ɛ n d ɪ g m oː t f eː l h ɔ s t ə n",
    "Ik ben verhuisd en ik wil mijn nieuwe adres doorgeven.":
        "ɪ g b ɛ n f ɛ r h aʊ z t ɛ n ɪ k w ɛ l m aɪ n n uː eː h a d r ɛ s d oː r x eː v ə n",
}

all_mistakes = []
for sentence, heard in samples.items():
    mistakes, per = compare(sentence, heard)
    all_mistakes += mistakes
    print(sentence)
    print(f"  Phoneme error rate: {per:.0%}")
    for m in mistakes:
        print(f"  - {m['word']}: expected {m['expected']}, heard {m['heard'] or '(missing)'}")
    print()

print("===== Klanken om te oefenen (sounds to practise) =====")
for tip in summarise(all_mistakes):
    print("-", tip)