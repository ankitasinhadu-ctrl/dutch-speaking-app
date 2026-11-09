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

### Whisper speed vs accuracy (9 November 2026)

| Model | beam_size | WER | Avg time per sentence |
|---|---|---|---|
| small | 5 | 16.3% | 7.0s |
| small | 1 | 20.4% | 6.2s |
| base | 5 | 16.3% | 2.0s |
| base | 1 | 32.7% | 1.8s |

- Chosen setting: base model, beam_size 5. Same WER as small on our test set, 3.5x faster on a laptop CPU.
- Caveat: only 5 test sentences (about 50 words), so WER differences are indicative.