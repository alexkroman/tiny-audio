# Turn-aware Qwen3-ASR: answers to the review questions

**Date:** 2026-09-30 · **Re:**
[`turn-aware-qwen3-asr-vs-commercial-endpointers.md`](turn-aware-qwen3-asr-vs-commercial-endpointers.md)
(the "original report") · **Branch:** `turn-aware-qwen3-asr`

## Short version

1. **Row counts:** you read the table correctly. The strict-cutoff column came
   from a different scoring than the other columns. The strict scoring also had
   a bug, and it is the one you describe in question 3. With a fixed scorer,
   every row adds up to 200. AssemblyAI `balanced` goes from 57 real cutoffs
   (old strict) or 30 (old generous) to **32–34**. Per-turn fire times and the
   scoring script are now in the repo.
1. **Competitor runs:** all six arms were captured on **2026-09-16**. AssemblyAI
   was `universal-3-5-pro` on the v3 streaming API with stock presets. Deepgram
   was Flux `flux-general-en` at its defaults (`eot_threshold` 0.7,
   `eot_timeout_ms` 5000). ElevenLabs was `scribe_v2_realtime` with default VAD.
   Audio was sent as 16 kHz PCM16 in **50 ms frames**, paced in real time.
1. **Pause window:** yes, the old strict scoring counted a late but correct fire
   as a cutoff. The new scorer credits any fire that lands while the caller is
   still silent to that pause, however late it comes.
1. **Latency with compute:** not measured on a GPU. On the one machine measured
   (an M4 Max laptop), one re-decode of a 10 s prefix takes **~1.0 s**, so the
   "+0.1–0.2 s" estimate is **not supported**. As built, the method does not run
   in real time on that hardware. About 95% of the cost is regenerating the
   transcript on every window. A score-only design would cut it to about the
   prefill cost (0.06–0.15 s), but that design is not yet built or validated.
1. **Data:** AssemblyAI's own streaming TTS, **11 English voices** (the full
   published catalog). The same 11 voices appear in train and test. Scripts were
   written by `claude-sonnet-5`. The pauses are spliced **digital silence**
   (exact zeros). Your concern about transfer is well founded: none of the
   hesitation, noise floor or number-reading variety of real callers is present.

The headline result that survives rescoring is fewer cutoffs than AssemblyAI
`balanced`, `balanced-control` and `min_latency` on this synthetic set
(paired p ≤ 0.007). The Deepgram comparison is weaker than reported (p = 0.21,
was 0.078), and `max_accuracy` is still a tie. The latency advantage is
unproven until compute is measured.

______________________________________________________________________

## 1. Row counts, per-turn fire times and the labeling script

### Why the rows added up to more than 200

The classification ran twice: *strict* (a fire is credited to a pause only
within 1.2 s of the chunk end) and *generous* (1.2 s plus the service's p90
latency). The published table put the **strict** cutoff count next to the
**generous** "Early at complete sentence" count, so for any system where the
two scorings disagree, the row double-counts. Re-running the original
classifier reproduces every published number exactly:

| System | Strict: cutoff / complete / end / never | Generous: cutoff / complete / end / never |
|---|---|---|
| Turn-aware Qwen3-ASR | 15 / 156 / 29 / 0 | 15 / 156 / 29 / 0 |
| AssemblyAI `balanced` | **57** / 103 / 40 / 0 | 30 / **130** / 40 / 0 |
| AssemblyAI `max_accuracy` | **28** / 108 / 64 / 0 | 16 / **120** / 64 / 0 |
| Deepgram | **30** / 131 / 37 / 2 | 24 / **137** / 37 / 2 |
| ElevenLabs | **103** / 0 / 97 / 0 | 11 / **92** / 97 / 0 |

(Bold marks the two numbers that were put side by side.) Each scoring adds up
to 200 on its own. Our model scores the same under both, which is why its row
looked fine.

### Why `balanced`'s strict number looked too high

You were right to flag it. 27 of `balanced`'s 57 strict cutoffs were fires
1.2 s or more after a complete-sentence chunk ended. In 23 of those 27, **the
caller was still silent**. The strict window was measured from the chunk end, not from when the
caller spoke again, and the chunk-boundary pauses in this set are long (median
1.84 s). A system that answers ~0.5 s into a pause, plus network, was being
charged for cutoffs on pauses it had correctly waited through. This is the bug
you describe in question 3.

### The fixed scorer

`scripts/turn_aware/score_endpoints.py` (new) replaces the one-off script. It
finds speech segments in the audio (20 ms frames, the same RMS threshold the
runtime uses, pauses ≥ 0.1 s) and assigns each first fire to the **pause it
was caused by**:

- A fire that lands **while the caller is silent** goes to that pause, however
  late it comes. No window is needed.
- A fire that lands **after the caller has resumed** is where lateness matters,
  and it is scored two ways:
  - **Strict:** it is a cutoff. From the caller's side, the agent talked over
    them.
  - **Generous:** it goes to the pause the caller just left, if it arrives within
    the system's own p90 reply latency.
- The pause is then labeled by the chunk it follows: complete sentence
  (`. ? !`, a correct fire by the dataset's `chunk_cut` label), mid-sentence, or
  inside a chunk. The last two, plus unexplained fires during speech, count as
  **real cutoffs**.
- Latency is now measured from the **last speech frame**, not from the end of
  the file. The file ends a median 0.12 s later because of the TTS clip's
  trailing silence.

```bash
poetry run python -m scripts.turn_aware.score_endpoints \
  outputs/turn_aware_local/replay200_v3_onset/replay.csv -o outputs/turn_aware_local/scored200_v3
```

**Per-turn fire times** for all seven systems (200 turns × 7 = 1,400 rows,
with both categories and the pause each fire was assigned to) are in
[`turn-aware-endpoints-per-turn.csv`](turn-aware-endpoints-per-turn.csv). The
services' raw fire times, including every later endpoint, are the dataset's
`endpoints` table (`endpoints_s`). The 200 turns are
`random.Random(0).sample(sorted(endpoint turn keys), 200)`.

### Rescored results (same 200 turns; every row adds up to 200)

Generous scoring:

| System | At turn end | Complete-sentence pause | Mid-sentence pause | Within-chunk pause | During speech | Never | **Real cutoffs (95% CI)** |
|---|---|---|---|---|---|---|---|
| **Turn-aware Qwen3-ASR** | 29 | 156 | 8 | 7 | 0 | 0 | **15 (4.6–12.0%)** |
| AssemblyAI `balanced` | 40 | 128 | 22 | 10 | 0 | 0 | 32 (11.6–21.7%) |
| AssemblyAI `balanced-control` | 40 | 126 | 22 | 12 | 0 | 0 | 34 (12.4–22.8%) |
| AssemblyAI `min_latency` | 32 | 140 | 20 | 8 | 0 | 0 | 28 (9.9–19.5%) |
| AssemblyAI `max_accuracy` | 64 | 116 | 16 | 4 | 0 | 0 | 20 (6.6–14.9%) |
| Deepgram Flux | 40 | 137 | 18 | 2 | 1 | 2 | 21 (7.0–15.5%) |
| ElevenLabs Scribe | 97 | 91 | 11 | 1 | 0 | 0 | 12 (3.5–10.2%) |

Real cutoffs under strict scoring (late fires after the caller resumed count
against the system): Qwen 15, `balanced` 34, `balanced-control` 35,
`min_latency` 35, `max_accuracy` 22, Deepgram 21, ElevenLabs 24.

Latency from the end of speech, on fires at the turn end. **Qwen is replay
time and excludes compute (see question 4). The services' numbers include
their compute and network.**

| System | p50 | p90 | p99 |
|---|---|---|---|
| Turn-aware Qwen3-ASR (no compute) | 0.36 s | 0.47 s | 0.50 s |
| AssemblyAI `balanced` | 0.50 s | 1.52 s | 1.60 s |
| AssemblyAI `balanced-control` | 0.60 s | 1.53 s | 1.70 s |
| AssemblyAI `min_latency` | 0.50 s | 0.91 s | 1.04 s |
| AssemblyAI `max_accuracy` | 0.87 s | 2.85 s | 2.93 s |
| Deepgram Flux | 0.76 s | 1.47 s | 2.14 s |
| ElevenLabs Scribe | 1.79 s | 1.89 s | 1.95 s |

The p99 values rest on 29–97 at-turn-end fires per system, so treat them as
rough.

Head to head, real cutoffs on the same turns:

| Qwen vs | Only Qwen cut in | Only the service | Both | Sign-test p (generous) | p (strict) |
|---|---|---|---|---|---|
| `balanced` | 5 | 22 | 10 | **0.002** | 0.0005 |
| `balanced-control` | 4 | 23 | 11 | **0.0003** | 0.0002 |
| `min_latency` | 4 | 17 | 11 | **0.007** | 0.0002 |
| `max_accuracy` | 8 | 13 | 7 | 0.38 | 0.21 |
| Deepgram Flux | 5 | 11 | 10 | 0.21 | 0.21 |
| ElevenLabs | 9 | 6 | 6 | 0.61 | 0.12 |

**What changed from the original report:** `balanced` gets slightly worse
(30 → 32), because 2 fires that had been credited to a complete pause are now
correctly placed inside a chunk. The Deepgram gap narrows (24 → 21, p 0.078 →
0.21), because 3 Deepgram fires in the TTS trailing silence are now counted as
turn-end fires. The claim "fewer cutoffs than Deepgram" should be dropped.

### Is `balanced` still high for its settings?

At 32 of 200 it is higher than you would expect from `max_turn_silence`
1,280 ms. It is **not** the fallback timer:

- `balanced`'s cutoffs fire a median **0.46 s** into the pause. Only 2 of 32
  fire past 1.28 s. They are the semantic check running at `min_turn_silence`
  (128 ms) and judging the text complete. Examples from the original report:
  `Hi, I need to`, `Confirmation number DX4958. Charge was`.
- It cuts in more than `min_latency` (32 vs 28) because of **first-fire
  scoring**. Of the 7 turns where only `balanced` cut in, `min_latency` had
  already ended the turn at an earlier complete-sentence pause in 5 (its
  640 ms fallback fires there more often: 140 vs 128). It never reached the
  later pause where `balanced` cut in. A faster endpointer is shielded from
  later cutoffs by its own early fires, so cutoff counts should always be read
  together with the "complete-sentence pause" column.
- **Reproducibility:** `balanced` and `balanced-control` are the same
  configuration captured in two separate runs a few hours apart. They agree on
  32 of the 34 cutoff turns (0 only in `balanced`, 2 only in `-control`).

## 2. Where the competitor numbers come from

The `endpoints` table was captured with the sibling repo `tiny-llm`
(`tiny_llm/asr.py`, `tiny_llm/incumbents.py`, `tl stream-capture`). The
dataset was published 2026-09-17.

| Arm | Service / model | Settings | Captured |
|---|---|---|---|
| `assemblyai:balanced` | AssemblyAI v3 streaming (`wss://…/v3/ws`), `speech_model=universal-3-5-pro`, `format_turns=true` | `mode=balanced` preset (`min_turn_silence` 128 ms, `max_turn_silence` 1,280 ms), no other overrides | 2026-09-16, 18:07–19:41 PT |
| `assemblyai:balanced-control` | same | same as `balanced`. This is the original capture, kept to check reproducibility. 300 of its rows predate the file recording `mode`, but the code sent `mode=balanced` throughout | 2026-09-16, 13:25–19:41 PT |
| `assemblyai:min_latency` | same | `mode=min_latency` (128 / 640 ms) | 2026-09-16, same run as `balanced` |
| `assemblyai:max_accuracy` | same | `mode=max_accuracy` (512 / 2,560 ms) | 2026-09-16, same run as `balanced` |
| `deepgram` | Deepgram **Flux**, `flux-general-en`, `EndOfTurn` events (not nova-3 `UtteranceEnd`) | `eot_threshold=0.7`, `eot_timeout_ms=5000` (Deepgram's defaults) | 2026-09-16, 14:21–20:57 PT |
| `elevenlabs` | ElevenLabs `scribe_v2_realtime`, VAD commit strategy | default VAD. `vad_silence_threshold_secs` is not set, so the server default applies | 2026-09-16, 15:01–19:41 PT |

How audio was sent to the services:

- **Format and pacing:** 16 kHz mono PCM16, sent in **50 ms frames** and paced
  to wall-clock real time. The endpointers' silence thresholds are wall-clock
  timers, so faster-than-real-time sending would distort them.
- **Trailing silence:** 3 s of digital silence is appended so the fallback
  timers can fire. Flux gets 6 s, because its pad is `eot_timeout_ms` + 1 s.
- **Concurrency:** AssemblyAI ran 8 concurrent sessions; Deepgram and
  ElevenLabs ran 4.
- **Fire time:** a fire's time is when the client *received* the end-of-turn
  message, measured from stream start with `time.monotonic`. It therefore
  includes server compute and network round trip. All captures ran from one
  client machine.
- **First fire:** `first_endpoint_s` is the first end-of-turn message.

On your note that our endpointing defaults have changed: these are **mid-September
presets on U3.5 Pro**. If `balanced`'s thresholds or the semantic check have
changed since then, these arms need recapturing before any comparison for U3.7.
Recapturing is cheap (1,030 turns per arm) and doesn't touch the model side.

## 3. Fires that land late in a pause

The old strict scoring did count them as cutoffs (see question 1: 27 of
`balanced`'s 57). Even the old generous scoring only covered them up to
1.2 s + p90 after the chunk end, so a very late fire in a very long pause could
still be miscounted.

The new scorer has no window for fires that land in silence. If the caller has
not started speaking again, the fire belongs to that pause, whether it came
0.2 s or 2.5 s in. A timer-driven fire at 1,280 ms, plus network, is therefore
credited to the right pause: correct if the chunk before it ended a sentence,
a cutoff if it didn't. Lateness only matters for fires that arrive after the
caller has resumed. Those are the cases where strict and generous differ, and
there are few of them (0–14 per system, mostly ElevenLabs and `min_latency`).

One remaining approximation: the scorer labels a pause by the punctuation of
the scripted chunk before it. A within-chunk pause can follow a complete
sentence, so "within-chunk" slightly overcounts real cutoffs. The overcount
applies the same way to every system.

## 4. Latency including compute

**Not measured on a GPU, and not measured with concurrent streams.** The
original report's "+0.1–0.2 s" was an estimate, and the one measurement we do
have says it is wrong for the current design.

Measured on an Apple M4 Max (MPS, bf16, eager attention), with real prefixes
from the 200 test turns, median of 3 runs × 3 turns:

| Prefix length | Full re-decode, 1 stream | Full re-decode, batch 8 (per stream) | Prefill + 1 token, 1 stream |
|---|---|---|---|
| 2 s | 0.37 s | 0.06 s | 0.06 s |
| 5 s | 0.74 s | — | — |
| 10 s | 1.02 s | 0.21 s | 0.08 s |
| 15 s | 1.58 s | — | — |
| 20 s | 2.06 s | — | — |
| 25 s | 2.45 s | 0.58 s | 0.15 s |

- **The decoder regenerates the entire transcript every window.** Cost grows
  with turn length (about 45 ms per output token here) and is **~95% of the
  total**. Encoder plus prompt prefill is only 0.06–0.15 s even at 25 s.

- **How often it decodes:** the silence gate (decode only when the last 0.1 s
  is quiet) cuts decodes from 6.25 to **2.5 per second of audio** (median 40
  per turn). The gate is exact on this data because the pauses are digital
  zeros (see question 5). On real audio it would open less cleanly.

- **What that means:** one 10 s prefix takes ~1 s to re-decode. The fire
  decision would arrive about a second late, and the stream would fall behind
  real time. That erases the p50 advantage in the table above. **On this
  hardware the method does not yet run in real time.** An H100 will be much
  faster, but with batch-1 HF `generate` the per-token overhead dominates, so
  "fast enough" has to be measured, not assumed.

- **The fix is clear but not yet built or validated:** don't regenerate text
  that is already committed. Either

  1. **teacher-force** the previous window's transcript and read the
     `<END_OF_TURN>` vs `<|im_end|>` margin from a single forward pass, or
  1. **reuse the decoder KV cache** and generate only the new tail.

  Either would bring per-window cost close to the prefill column. Neither is
  implemented, and option 1 changes what the model is scored on, so it would
  need its own replay to confirm the cutoff numbers hold.

The next step is the measurement you asked for: N concurrent streams on one
GPU, fire time = decision window + queueing + decode, for the current design
and the teacher-forced one. Until that exists, the latency comparison should be
read as "decision time on the audio clock", not "reply time".

## 5. Data, voices and checkpoint

**TTS and voices:**

- The TTS is **AssemblyAI's streaming TTS** (`streaming-tts.assemblyai.com`).
- It uses the whole published **English catalog of 11 voices**: `jane`,
  `michael`, `alba`, `eve`, `george`, `jean`, `mary`, `anna`, `charles`,
  `paul`, `vera` (US-accented first, then British).
- Each turn's voice is chosen by a hash of its content.
- Splits are by a hash of the conversation, **not** by voice, so train,
  validation and test all contain all 11 voices. No voice is held out. The
  per-turn voice isn't stored in the published dataset, so per-split counts
  can't be given exactly. With ~5,000 test turns spread by hash, every voice is
  represented.

**Scripts and pauses:**

- `claude-sonnet-5` wrote all 58,455 caller scripts (10 customer-service
  domains) in the build.
- Each turn is 1–N separately synthesized sentence chunks, joined with
  **spliced digital silence** from a fixed ladder: 120, 250, 500, 900, 1,400,
  2,200 or 3,000 ms. The ladder was placed against endpointer thresholds.
- Mid-phrase breaks ("I need to, · pause · return this") are made by splitting
  one chunk and splicing a pause in. They are not natural hesitations.
- About 21% of samples in a typical test turn are exact zeros.

**Checkpoint:** [`mazesmazes/tiny-audio-turn-aware-qwen3-asr-v3`](https://huggingface.co/mazesmazes/tiny-audio-turn-aware-qwen3-asr-v3).
This is the final step (4,504) of one epoch on the v3 recipe (`configs/turn_aware/`,
`+experiment=v3`): LoRA r=32 on the Qwen3-ASR-0.6B decoder's q/k/v/o, plus the
`<END_OF_TURN>` embedding row. It was trained on the train split only, and the
test turns were never seen. The training silence is also digital zeros.

**How much of this carries over to real callers:** probably not all of it, and
the gaps line up with your list:

- **Silence is trivially easy here.** Both the model's "wait for observed
  silence" behavior and the runtime's energy gate were learned and tuned on
  exact-zero pauses. A phone line's noise floor, breaths and background speech
  will make "has there been 0.3 s of silence?" much harder. This is likely the
  biggest risk, and it hits our model harder than the services, which were
  built for real audio.
- **Hesitations are spliced, not spoken.** Real "um… so…" pauses come with
  lengthening, filled pauses and falling intonation that TTS doesn't make. The
  model has only seen the lexical side of an unfinished thought.
- **Numbers and spellings:** this is already the model's main failure in
  domain (`my phone number is 585.` · pause · `585-813-…`). Real callers group
  digits less predictably, so expect it to get worse.
- **Train and test share the same TTS voices and generator,** so the test set
  measures in-distribution generalization to new scripts, not to new speakers.
  The services have no such advantage. The comparison favors our model for
  that reason alone.

## A fair test

We'd welcome your offer. The pieces we'd bring:

- the scorer above, which runs unchanged on any turn set with per-turn fire
  times and speech-segment timing;
- a replay harness that already takes arbitrary 16 kHz audio;
- recaptures of every service arm at current defaults.

What's missing on our side:

- **Real conversational audio**, labeled the way you label U3 Pro endpointing.
  All of our data is synthetic.
- **A reply-time measurement for our model that includes compute** (question
  4). We'd want the teacher-forced variant built first, so the test doesn't just
  confirm that re-decoding is slow.
- **Transcription accuracy and entity errors,** which our report didn't compare.
  Our model's transcripts are within 1.1% WER of base Qwen3-ASR-0.6B, but that
  base hasn't been scored against U3 on your sets.

Scoring on your metrics (endpoint latency p50/p90/p99, early cutoffs, accuracy,
entity errors) with one scorer for all systems is the right bar. If the gains
don't hold on real audio, that's the answer, and it's cheap to get.
