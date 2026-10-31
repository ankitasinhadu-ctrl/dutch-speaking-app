# dutch-speaking-app

An AI speaking partner for practising Dutch, with feedback on pronunciation and grammar.
Built as a team project for TechPros Rotterdam (Deep Learning track, Cohort 8).

## Results

### Speech to text baseline (31 October 2026)

- Model: Whisper small (faster-whisper, CPU, int8)
- Test set: 5 Dutch sentences from our scenarios, spoken by a non-native speaker
- Overall WER: 16.3%
- Errors clustered around difficult Dutch sounds: ui (verhuisd → verhaalst), oe (hoesten → hosten), v/f (veel → feil)
- Next step: separate Whisper errors from pronunciation errors with a wav2vec2 phoneme model