# Universal-3.5-Pro vs tiny-audio-granite-qwen-frozen

*Analysis date: 2026-09-24. Sweeps run 2026-09-22.*

Reproduce the headline tables with:

```bash
COLUMNS=260 poetry run ta analysis compare universal-3-5-pro tiny-audio-granite-qwen-frozen
```

## TL;DR

- **Headline corpus WER is a statistical tie.** tiny-audio 7.03% vs U3.5 7.16%. The paired-bootstrap Δ is +0.14 pt [−0.02, +0.30], so the two models are *not separated*.
- **The tie hides a clean split by training domain.** On the 7 test sets whose domain tiny-audio trained on, **tiny-audio is better**: Δ +0.57 [+0.35, +0.79]. On the 3 held-out domains (Earnings22, Peoples Speech, Loquacious), **U3.5 is better**: Δ −0.69 [−1.05, −0.31].
- **AMI decides the corpus number.** U3.5 produces many empty or runaway outputs on meeting audio. Drop the two AMI sets and **U3.5 is significantly better**: 6.16 vs 6.42, Δ −0.26 [−0.40, −0.10].
- **On "clean" rows U3.5 is also significantly better.** Excluding rows where either model hit ≥100% WER (n = 11,163) gives Δ −0.21 [−0.33, −0.09]. This filter conditions on the outcome, so treat it as a robustness check, not an unbiased estimate.
- **Where each model wins:**
  - tiny-audio clearly wins AMI (+4.75), ami-sdm (+6.73) and Earnings22 (+1.13).
  - U3.5 clearly wins Peoples (−2.80) and TED-LIUM (−0.45).
  - Everything else is within ±0.7 pt and mostly not significant.
- **Formatting (ITN):**
  - U3.5 reproduces 93.7% of numeric spans exactly vs 86.0% for tiny-audio.
  - tiny-audio emits **no currency symbols** (0/28) and is weak on decimals (71%).
- **Latency numbers are not comparable** (see [Latency](#latency)).

## Systems under test

| | tiny-audio-granite-qwen-frozen | universal-3-5-pro |
|---|---|---|
| Type | Open speech-LLM, local inference | AssemblyAI production API |
| Encoder | `ibm-granite/granite-speech-5.0-470m-turboctc`, **fully frozen** | proprietary |
| Adapter | 2-layer MLP projector (12.6M), 12.5 Hz audio tokens | — |
| Decoder | `Qwen/Qwen3.5-2B`, frozen base + LoRA r=64/α=64 all-linear (67.3M) | — |
| Trainable | 79.9M of 2.43B params (3.3%) | — |
| Training | One pass over ~2.85M utterances (`configs/data/multiasr.yaml`), 44,561 steps, batch 64. Sources: LibriHeavy-medium, TED-LIUM, Common Voice 17, GigaSpeech-M, SPGISpeech-M, VoxPopuli-en, AMI IHM+SDM | — |
| Decoding | Greedy; 0.25 s silence prepended (`inference_lead_in_seconds`); prompt "Transcribe the speech with proper punctuation and capitalization" | API default |
| Recipe | `configs/experiments/granite_qwen_frozen.yaml` | — |
| Sweep ID | `64b28e90e087` | `a158e0148b86` |

The checkpoint scored is `mazesmazes/tiny-audio-granite-qwen-frozen`. The eval harness does not record the Hub revision, so the checkpoint is identified by sweep time against the push log (final weights, step 44,561, pushed 2026-09-22 12:26 UTC). Training started 2026-09-21, after Earnings22 and Peoples Speech were removed from the training mix (commit `3262859d`, 2026-09-18).

## Evaluation protocol

- **12 English test sets, 1,000 utterances each.** Rows are shuffled with the same seed for both models, so the pairing is exact: the scripts assert identical references row by row. After dropping empty references there are **11,822 scored rows and 209,289 reference words**.
- **Scoring.** Both models are re-normalized from the raw text at analysis time with the harness `TextNormalizer` (Whisper-style English normalizer plus repo rules), then scored with `jiwer`. WER is corpus WER, total errors over total reference words, not the mean of per-utterance WERs.
- **Significance.** Paired utterance bootstrap with 10k resamples. Both models are resampled on the same row indices and the corpus WER is recomputed each time (`scripts/analysis.py::_paired_bootstrap_delta`).
- **Sign convention.** Δ below is **U3.5 − tiny-audio**. Positive means tiny-audio is better, negative means U3.5 is better.

## Results

### Corpus

| Slice | Rows | U3.5 WER | tiny WER | Δ (U3.5 − tiny) | 95% CI | Verdict |
|---|---:|---:|---:|---:|---|---|
| All 12 sets | 11,822 | 7.16 | 7.03 | +0.14 | [−0.02, +0.30] | not separated |
| **In training domain** (AMI, ami-sdm, GigaSpeech, SPGI, TED-LIUM, CV, VoxPopuli) | 6,825 | 7.21 | 6.64 | +0.57 | [+0.35, +0.79] | **tiny better** |
| **Held-out domain** (Earnings22, Peoples, Loquacious) | 2,997 | 10.77 | 11.46 | −0.69 | [−1.05, −0.31] | **U3.5 better** |
| Excl. AMI + ami-sdm | 9,994 | 6.16 | 6.42 | −0.26 | [−0.40, −0.10] | **U3.5 better** |
| Rows where neither model ≥100% WER | 11,163 | — | — | −0.21 | [−0.33, −0.09] | U3.5 better (conditioned) |

Total errors: U3.5 14,995 vs tiny 14,711. One corpus point is ≈ 2,090 word errors.

LibriSpeech clean and other are left out of both domain slices. tiny-audio trains on LibriHeavy, which is drawn from the same LibriVox audiobook pool, so neither label fits cleanly.

The held-out pool is not uniform:

- **Peoples Speech** (−2.80) drives it.
- **Loquacious** (−0.36) leans toward U3.5.
- **Earnings22** favors tiny-audio (+1.13). It is held out at the corpus level, but SPGISpeech (studio earnings calls) is in training, so the domain is adjacent rather than unseen.

### Per dataset

| Dataset | Rows | U3.5 | tiny | Δ | 95% CI | Rows U3.5 better / tiny better | Sig. |
|---|---:|---:|---:|---:|---|---:|---|
| ami-sdm | 909 | 30.01 | 23.29 | +6.73 | [+5.07, +8.41] | 134 / 327 | tiny |
| AMI (IHM) | 919 | 13.58 | 8.83 | +4.75 | [+3.44, +6.16] | 84 / 212 | tiny |
| Earnings22 † | 997 | 11.73 | 10.60 | +1.13 | [+0.42, +1.94] | 211 / 232 | tiny |
| VoxPopuli | 1000 | 7.07 | 6.84 | +0.23 | [−0.17, +0.69] | 220 / 233 | — |
| SPGISpeech | 1000 | 2.28 | 2.19 | +0.09 | [−0.24, +0.49] | 124 / 94 | — |
| LS other | 1000 | 3.04 | 3.08 | −0.04 | [−0.41, +0.36] | 157 / 126 | — |
| GigaSpeech | 997 | 8.94 | 8.99 | −0.05 | [−0.56, +0.53] | 216 / 174 | — |
| LS clean | 1000 | 1.66 | 1.85 | −0.18 | [−0.39, +0.02] | 97 / 64 | — |
| Loquacious † | 1000 | 5.46 | 5.82 | −0.36 | [−0.77, +0.05] | 181 / 146 | — |
| TED-LIUM | 1000 | 3.22 | 3.67 | −0.45 | [−0.70, −0.20] | 178 / 99 | U3.5 |
| Common Voice | 1000 | 5.93 | 6.57 | −0.64 | [−1.36, +0.07] | 148 / 104 | — |
| Peoples Speech † | 1000 | 15.00 | 17.80 | −2.80 | [−3.48, −2.12] | 405 / 193 | U3.5 |

† These corpora are not in this checkpoint's training mix.

**Earnings22 note:** `sanchit-gandhi/earnings22_robust_split` is not call-disjoint. Its train and test splits share every sampled call. Earlier tiny-audio runs trained on that train split, and their Earnings22 numbers are contaminated. This checkpoint was trained after the split was removed (2026-09-18), so its Earnings22 number is clean. Before comparing Earnings22 across tiny-audio checkpoints, check each one's training date.

### Robustness to catastrophic rows

Some rows go badly wrong: empty output, `!!!!!!`, a language switch, or a runaway repetition. A few such rows can decide a corpus WER.

| | U3.5 | tiny |
|---|---:|---:|
| Rows empty after normalization | 102 | 48 |
| Rows with WER ≥ 100% | 598 | 417 |
| Word errors on U3.5's empty rows | 408 | 129 (tiny on the same rows) |

AMI and ami-sdm hold most of the gap: 425 of U3.5's 598 ≥100% rows vs 285 of tiny-audio's. With those rows removed, the AMI advantage shrinks but stays significant: AMI +2.88 [+1.97, +3.79], ami-sdm +3.50 [+2.12, +4.87]. Earnings22 drops to +0.27 [−0.20, +0.76], so its win is mostly U3.5 failures. GigaSpeech flips to U3.5 at −0.48 [−0.82, −0.14].

Interpretation: part of tiny-audio's AMI lead is robustness (U3.5 failing outright on far-field or overlapped meeting audio). The rest is **domain and style match**: AMI transcripts are in tiny-audio's training mix, so it learned AMI's verbatim transcription conventions. U3.5 has not seen them.

### Insertions (hallucination proxy)

| | Corpus | Peoples | ami-sdm | Earnings22 | LS clean |
|---|---:|---:|---:|---:|---:|
| U3.5 | 1.62% | 4.55% | 6.66% | 2.61% | 0.13% |
| tiny | 1.70% | 5.89% | 4.57% | 2.17% | 0.29% |

Corpus insertion rates are similar. tiny-audio inserts more on Peoples and on clean read speech. U3.5 inserts more on far-field meetings.

**Caveat:** the tiny-audio pipeline trims repetition loops *before* scoring (`_truncate_repetitions` in `asr_pipeline.py`). Its insertion rate is therefore a lower bound, and looping behavior is unmeasured.

### WER by reference length

| Words in ref | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U3.5 | 82.7 | 58.0 | 29.8 | 23.8 | 19.3 | 14.2 | 13.3 | 10.4 | 10.4 | 9.0 |
| tiny | 57.6 | 54.8 | 24.6 | 22.4 | 14.6 | 12.8 | 12.4 | 10.9 | 10.2 | 8.8 |

tiny-audio is much better on 1–5 word utterances, mostly backchannels and short turns from AMI and Peoples. The two converge from about 8 words on. Short utterances are where U3.5's empty outputs concentrate. Audio duration is not recorded by the harness, so word count is the only length proxy available.

### Formatting / inverse text normalization

This is scored on **raw** (un-normalized) text: the percentage of reference numeric spans reproduced exactly. "Fmt-only err" means the value is right but the format is wrong (e.g. `1250` for `1,250`).

| | Exact | Fmt-only err | currency (28) | percent (93) | thousands (22) | decimal (21) | alnum ID (45) | year (77) | integer (172) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U3.5 | **93.7** | 2.1 | **85.7** | 95.7 | 95.5 | **95.2** | **91.1** | 98.7 | **96.5** |
| tiny | 86.0 | 5.7 | 0.0 | **97.8** | 95.5 | 71.4 | 84.4 | 98.7 | 92.4 |

- **Currency is a label artifact, not a recognition failure.** tiny-audio never emits `$`. Its training labels spell currency out, and the WER normalizer erases the difference, so this costs 0 WER but matters for any user-facing transcript.
- **Percent formatting works in tiny-audio.** It was broken until a label-pipeline fix on 2026-09-18.
- **Spans per class are small (21–172).** Differences of a few points are within noise.

### Named entities

Both models miss 8.47% of ORG references (n = 59). Other entity types have fewer than 15 references and are hidden. **This sweep cannot resolve entity accuracy.** Earlier manual error analysis suggests entity and rare-word errors are the main remaining gap on Common Voice, GigaSpeech, SPGISpeech and part of Peoples. A dedicated entity-dense test set is needed to measure it.

### Confidence (tiny-audio only)

Mean top-1 token log-prob is −0.096 and the mean top-1/top-2 margin is 5.69 nats. Confidence tracks difficulty: it is lowest on ami-sdm (−0.28 / 4.13) and Peoples (−0.16 / 4.70), and highest on SPGISpeech and LS clean (−0.05 / 6.1). U3.5 does not expose comparable scores here.

### Latency

| | Avg per utterance |
|---|---:|
| tiny (local) | 805 ms |
| U3.5 (API) | 4,041 ms |

**Do not read this as a speed comparison.** The two numbers measure different things:

- **tiny-audio:** in-process `pipeline()` wall time for one utterance on the eval machine. The hardware isn't recorded in the sweep, and there is no batching.
- **U3.5:** synchronous SDK `transcribe()` round trip, which covers upload, queueing, and polling of the async batch API. It is dominated by network and job overhead, not model compute.

A fair comparison needs matched hardware or a streaming time-to-first-token and RTF benchmark.

## Threats to validity

1. **Training-domain overlap favors tiny-audio.** tiny-audio trains on the *train* splits of 7 of the 12 test corpora (AMI ×2, GigaSpeech, SPGISpeech, TED-LIUM, Common Voice, VoxPopuli), plus LibriHeavy. Test splits are the canonical speaker-disjoint ones, so this is not leakage. It does mean tiny-audio learned each corpus's transcription *conventions*, such as verbatim fillers and disfluencies, while U3.5 is scored zero-shot against them. The in-domain vs held-out split above measures this effect directly. The held-out result is the better estimate of how tiny-audio performs on new customer audio.
1. **The held-out pool is small and mixed.** It covers three corpora (n = 2,997), and Peoples Speech dominates it. Peoples segment boundaries are tight. In 182 of 1,000 rows both models miss the first reference word, and on inspection it is often not audible. Another 121 rows lose a leading function word ("and", "the", "so") for tiny-audio only, which looks like a decoder prior against sentence-initial function words at a zero-silence onset.
1. **Normalization dominates small deltas.** Most per-dataset differences are under 0.5 pt, which is the same size as normalizer choices such as fillers, hyphenation and compounds. The normalizer was **not** tuned toward either model.
1. **Repetition is sanitized for tiny-audio before scoring.** U3.5 output is not post-processed, so the comparison slightly favors tiny-audio on looping.
1. **Sample size.** n = 1,000 per set gives per-dataset CIs of about ±0.2–1.7 pt. The corpus CI is about ±0.16 pt. The headline tie is a real "can't tell", not a near-win.
1. **Single checkpoint, single sweep.** tiny-audio is bit-deterministic for a fixed revision. U3.5 is an API and may change server-side, so these numbers are a 2026-09-22 snapshot.

## Takeaways for research

- **Parity on the benchmark is not parity on new audio.** An 80M-trainable-param adapter on frozen open models ties a production system on this 12-set suite. Split by training domain, though, it leads on domains it trained on (+0.57) and trails on domains it didn't (−0.69). The benchmark suite overlaps heavily with the training mix, which flatters any in-house model trained on it.
- **The remaining gap is concentrated, not diffuse.** U3.5 clearly wins on Peoples Speech, where tiny-audio drops onset and function words on an unseen domain, and on TED-LIUM. Entity and rare-word errors account for much of the rest. Cheap inference-time levers are exhausted: prompt swaps, longer lead-in, logprob gating and CTC fusion were each measured and rejected. Further gains need data or decoder capacity.
- **Robustness is tiny-audio's real edge.** It fails outright far less often on far-field, overlapped, very short audio (48 vs 102 empty rows, 417 vs 598 rows ≥100% WER).
- **Formatting is a data problem.** Currency symbols and decimals come from label conventions in the training mix, not from model capacity.
