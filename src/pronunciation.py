import re
from collections import defaultdict
import jiwer
from g2p import text_to_phonemes

# Differences that are normal Dutch, not mistakes
ACCEPTED = {
    ("ɣ", "x"),               # voiceless g, common in the west
    ("v", "f"),               # v said like f, common in the west
    ("ʋ", "v"), ("ʋ", "w"),   # Dutch w has several accepted variants
    ("ɑ", "a"),               # very small vowel difference
    ("ɔː", "oː"),             # 'oo' before r
    ("d", "t"), ("b", "p"),   # ends of words are voiceless in Dutch
    ("k", "g"),               # k before b/d often sounds like g
    ("r", "ɹ"), ("r", "ʁ"), ("r", "ɾ"),  # many kinds of r are fine
}
IGNORE_MISSING = {"r", "ə", "t"}  # sounds often dropped in normal speech

# The difficult Dutch sounds we give feedback on: phoneme -> spelling
FOCUS = {"œy": "ui", "u": "oe", "øː": "eu", "ɛɪ": "ij", "y": "uu", "aː": "aa"}

# How to describe what the learner said instead
HEARD_AS = {"aʊ": 'English "ow"', "ɔ": 'a short "o"', "oː": 'a long "o"',
            "a": 'a short "a"', "ʊ": 'a short "oe"', "aɪ": 'English "i" (as in "my")'}

def compare(sentence, heard):
    """Compare expected and heard sounds for one sentence.
    Returns the list of real mistakes and the phoneme error rate (PER)."""
    word_phonemes = [w.split() for w in text_to_phonemes(sentence).split("|")]
    words = re.findall(r"[^\W_]+", sentence.lower())
    if len(words) != len(word_phonemes):          # safety net
        words = ["".join(p) for p in word_phonemes]

    expected, owner = [], []                       # owner = which word each sound belongs to
    for i, phones in enumerate(word_phonemes):
        expected += phones
        owner += [i] * len(phones)
    heard_list = heard.split()

    # Line up the two sound lists (Levenshtein alignment, via jiwer)
    result = jiwer.process_words(" ".join(expected), " ".join(heard_list))

    mistakes = []
    for chunk in result.alignments[0]:
        if chunk.type in ("equal", "insert"):
            continue
        for k in range(chunk.ref_start_idx, chunk.ref_end_idx):
            exp = expected[k]
            if chunk.type == "substitute":
                got = heard_list[chunk.hyp_start_idx + (k - chunk.ref_start_idx)]
                if (exp, got) in ACCEPTED:
                    continue
            else:                                  # "delete": sound missing
                got = None
                if exp in IGNORE_MISSING:
                    continue
            mistakes.append({"word": words[owner[k]], "expected": exp, "heard": got})

    per = len(mistakes) / len(expected)
    return mistakes, per

def summarise(all_mistakes, min_count=2, top=3):
    """Turn mistakes into a few plain-language tips about the difficult sounds."""
    by_sound = defaultdict(list)
    for m in all_mistakes:
        if m["expected"] in FOCUS and m["heard"]:
            by_sound[m["expected"]].append(m)
    tips = []
    for exp, ms in sorted(by_sound.items(), key=lambda item: -len(item[1])):
        if len(ms) < min_count:
            continue
        heard = sorted({HEARD_AS.get(m["heard"], '"' + m["heard"] + '"') for m in ms})
        words = ", ".join(m["word"] for m in ms)
        tips.append(f'"{FOCUS[exp]}" sounded like {" or ".join(heard)} ({len(ms)} times: {words})')
    return tips[:top]