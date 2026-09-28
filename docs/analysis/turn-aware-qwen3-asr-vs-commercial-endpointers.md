# Turn-aware Qwen3-ASR vs commercial endpointers

**Date:** 2026-09-28 · **Branch:** `turn-aware-qwen3-asr` · **Model:**
[`mazesmazes/tiny-audio-turn-aware-qwen3-asr-v3`](https://huggingface.co/mazesmazes/tiny-audio-turn-aware-qwen3-asr-v3),
used **without** the agent-question context.

## Summary

On 200 random test turns of `mazesmazes/turn-end-detection`, streamed through
each system, Turn-aware Qwen3-ASR cut callers off mid-thought (a *real cutoff*)
on **15 turns (7.5%)**. AssemblyAI's default `balanced` endpointer did so on 30
(15.0%) and Deepgram on 24 (12.0%), even when scored generously for their
network delay. Turn-aware Qwen3-ASR also replied fastest: **0.24 s p50 /
0.32 s p90** after the caller finished, against 0.41 s / 1.43 s for AssemblyAI
`balanced` and 0.69 s / 1.32 s for Deepgram.

- **vs AssemblyAI `balanced`, `balanced-control` and `min_latency`:** about half
  the real cutoffs (paired sign test p = 0.002–0.017) at 1.7–2.2× lower median
  latency.
- **vs AssemblyAI `max_accuracy`:** the same cutoff rate (15 vs 16, p = 1.0) at
  about 3× lower median and 8.6× lower p90 latency.
- **vs Deepgram:** fewer cutoffs (15 vs 24, p = 0.078, not significant at this
  sample size) at about 3× lower median latency.
- **Where Turn-aware Qwen3-ASR is more aggressive:** it fires more often after a
  *complete* sentence that the caller then extends ("… Oh, and …"): 156 turns vs
  120–141. The dataset labels that moment a correct fire, but slower endpointers
  sometimes let the caller resume first.

Read these results as in-domain. The test audio is the same synthetic TTS style
the model was trained on, and its latency excludes model compute and network
time (see [Caveats](#caveats)). This report compares turn-taking only;
transcription accuracy is not compared.

## The model

| | |
|---|---|
| Base | `Qwen/Qwen3-ASR-0.6B-hf` (audio tower, projector and Qwen3 decoder all frozen) |
| Trained | LoRA r=32 on decoder q/k/v/o, plus one `<END_OF_TURN>` token row (9.18M params) |
| Behavior | Transcribes as usual and appends `<END_OF_TURN>` once the speaker is done **and** at least 0.3 s of silence has been heard; one greedy decode yields both the transcript and the endpoint |
| Training data | About 144k examples built from `mazesmazes/turn-end-detection` plus 12.3k mined mid-sentence pauses; see [Training data](#training-data) |
| Recipe | `configs/turn_aware/` preset `+experiment=v3`; code in `scripts/turn_aware/` (training) and `tiny_audio/turns/` (runtime) |

## Training data

All training and evaluation audio comes from one synthetic dataset. Test turns
never appear in training.

### Source: `mazesmazes/turn-end-detection`

- **What it is:** LLM-written customer-service caller turns across 10 domains
  (banking, travel, healthcare scheduling and others), spoken by a commercial
  TTS voice at 16 kHz.
- **How turns are built:** each turn is several separately synthesized chunks
  (3.25 on average in the test sample). The dataset's `agent_turn` field holds
  the agent question the caller is answering.
- **Labels:** each labeled *observation* is a cut point inside a turn, with a
  causal label (has the caller finished, judging only from what has been said so
  far?). Its `text` is another recognizer's partial transcript.
- **Splits:** assigned by a hash of the conversation, so no turn crosses train,
  validation and test.

The model trains on the **train split: 80,185 observations from 42,478 turns**
(a few with no speech left after trimming, or over 30 s, are dropped).

| Kind (share of observations) | Label | What the prefix is |
|---|---|---|
| `chunk_cut` (42.8%) | finished | A complete sentence at an interior pause; the caller may add more |
| `complete` (34.5%) | finished | The whole turn |
| `trailing_frag` (11.5%) | not finished | A complete request plus the start of a new thought |
| `word_cut` (6.7%) | not finished | Cut mid-phrase |
| `payload_cut` (4.3%) | not finished | Cut just before the detail the agent asked for (an address, an order number) |
| `mid_thought_cut` (0.1%) | not finished | An interior pause the same thought continues past |

### Added: 12,337 mined mid-sentence pauses (`mazesmazes/turn-end-detection-mined-pauses`)

Every boundary between two chunks is a real pause in the audio, but only about
46% of them carry an observation (measured on the validation split). We mined the rest where the chunk's script text
stops without terminal punctuation, which gave 13,571 train pauses. Claude
(`claude-opus-5`) then judged each one from the words so far and the agent's
question only, never from what the caller said next. Only the 12,337 judged
`incomplete` are used, as extra *not finished* examples. The 1,234 judged
`complete` or `ambiguous` were dropped: training on complete sentences as holds
contradicts the dataset's own `chunk_cut` labels.

### From labeled moments to training examples

Each observation's audio prefix is trimmed back to its last speech frame (plus
80 ms), then re-extended with a controlled amount of silence. That silence is
what the model learns to wait for:

| Example | Built from | Audio | Target |
|---|---|---|---|
| Fire | Every finished prefix | Prefix + 0.3–1.2 s silence (2–4 s for 10% of them) | Transcript + `<END_OF_TURN>` |
| No-silence twin | 60% of finished prefixes | The **same** prefix with no added silence | Transcript only |
| Hold | Every unfinished prefix; `payload_cut` ×3 and `word_cut` ×2, each copy with its own silence draw | Prefix + 0.3–2.0 s silence | Transcript only |
| Silence only | 2% of the pool | 0.5–4 s of silence | Empty |

- **Why the fire and its no-silence twin share the same speech:** observed
  silence is then the *only* difference between "fire" and "don't fire yet",
  so the model has to wait for it rather than fire on a finished-sounding
  sentence.
- **Why unfinished prefixes also get silence:** those holds teach the opposite,
  that a pause after unfinished speech is not a turn end.
- **Leading silence:** 15% of examples get 0.5–2 s of it, since streaming
  windows can open mid-silence.
- **Context:** 50% carry the agent's question as a system prompt. It isn't used
  at inference.
- **Length cap:** examples longer than 30 s are dropped.

**Transcripts.** Every target transcript is the **base Qwen3-ASR's own greedy
transcript** of the trimmed prefix, not the dataset's `text`. The only new
behavior the adapter has to learn is therefore the `<END_OF_TURN>` token, and
transcription stays within 1.1% WER of the base. On hold examples the final
`. ? !` is removed: the base model punctuates 96–100% of *unfinished* prefixes as
if they were finished, and without the removal every hold would look finished
by the time the model decides.

### Size and training run

| | |
|---|---|
| Training examples | about 144k: ~62k fires, ~37k no-silence twins, ~31k dataset holds (with copies), 12.3k mined holds, ~2.9k silence only |
| Training | One epoch, batch 32, 4,504 steps; AdamW at 2e-4 with cosine decay to 5%; bf16 on one GPU |
| Checkpoint | The final one (no early stopping or checkpoint selection) |
| Held out | The validation split (5,363 turns) for training-time evaluation and the offline tables; the test split (5,299 turns, including the 1,030 with commercial endpoint data) only for the streaming results |

The example counts are approximate; the total follows from the 4,504 steps.

## How the comparison was run

**Audio.** 200 turns drawn at random (seed 0) from the 1,030 test turns for
which the dataset's `endpoints` table records each commercial system's
behavior. Each turn is real dataset audio: TTS chunks with their natural pauses
between and within chunks, and their own leading silence.

What the pauses look like in these 200 turns (silence of 0.3 s or more, measured
from the audio):

| | Per turn | p10 | Median | p90 |
|---|---|---|---|---|
| Pauses at chunk boundaries | 2.25 | 0.72 s | **1.84 s** | 3.46 s |
| Pauses inside a chunk | 0.60 | 0.32 s | 0.38 s | 0.60 s |
| Leading silence before speech | 1 | 0.10 s | 0.32 s | 0.66 s |
| Trailing silence after the last word | 1 | 0.02 s | 0.12 s | 0.24 s |

The pauses between chunks are long (median 1.84 s),
longer than every system's reply time except ElevenLabs'. That is why every
endpointer fires early at some of them (see the "Early at complete sentence"
column), and why waiting a little longer barely changes which turns get cut
into.

**The commercial systems** (`endpoints` table). Each turn was streamed to the
service in real time, and the first moment it declared the turn over
(`first_endpoint_s`) was recorded. The first declaration is the one that ends
the turn in production.

**Turn-aware Qwen3-ASR** (`ta turn-aware replay`). The turn audio is followed by
2.0 s of appended silence. Every 0.16 s the model decodes the whole prefix heard
so far, the same way it was trained. Decoding starts only after speech onset and
only when the last 0.1 s is quiet, since a fire needs observed silence anyway.
The first window that emits `<END_OF_TURN>` is the endpoint. No agent-question
context is passed.

**Classifying each system's first fire** against the turn's chunk structure:

| Category | Meaning |
|---|---|
| At turn end | Fired after the audio ended: the intended endpoint |
| Early at a complete-sentence pause | Fired at a pause after a chunk ending in `. ? !` while the caller went on. The dataset labels this a correct fire (`chunk_cut`), so it is defensible |
| **Real cutoff** | Fired at a pause after a chunk that stops mid-sentence, or inside a chunk: the caller was talking over |
| Never | Did not fire at all |

The services report fires with real-time processing and network delay, so a fire
caused by a pause can land well after it. Two scorings bracket this:

- **Strict:** a fire is credited to a pause only within 1.2 s of it. This is
  unfavorable to slower systems.
- **Generous:** the window is 1.2 s plus each service's own p90 latency. This is
  favorable to them.

Turn-aware Qwen3-ASR's count is the same under both.

**Latency** is measured on turns where a system fired at the turn end: fire time
minus the end of the audio file. The file ends a median 0.12 s after the last
word (the TTS clip's own trailing silence), so time from the end of *speech* is
about 0.12 s longer for every system. Turn-aware Qwen3-ASR decides on a 0.16 s
grid, and by design only fires after at least 0.3 s of silence, so its latency
has a floor near 0.2 s and a resolution of 0.16 s.

**Uncertainty.** Rates carry 95% Wilson intervals. Head-to-head comparisons use
an exact sign test on the turns where exactly one of the two systems cut in; the
turns where both or neither did carry no information about which is better.

## Results

### All systems (200 test turns, same turns for every system)

| System | Real cutoffs (generous) | Real cutoffs (strict) | Early at complete sentence | At turn end | Never | Latency p50 | Latency p90 |
|---|---|---|---|---|---|---|---|
| **Turn-aware Qwen3-ASR** | **15 (7.5%; 4.6–12.0%)** | **15** | 156 | 29 | 0 | **0.24 s** | **0.32 s** |
| AssemblyAI `balanced` | 30 (15.0%; 10.7–20.6%) | 57 | 130 | 40 | 0 | 0.41 s | 1.43 s |
| AssemblyAI `balanced-control` | 32 (16.0%; 11.6–21.7%) | 59 | 128 | 40 | 0 | 0.52 s | 1.42 s |
| AssemblyAI `min_latency` | 27 (13.5%; 9.4–18.9%) | 27 | 141 | 32 | 0 | 0.40 s | 0.83 s |
| AssemblyAI `max_accuracy` | 16 (8.0%; 5.0–12.6%) | 28 | 120 | 64 | 0 | 0.78 s | 2.75 s |
| Deepgram | 24 (12.0%; 8.2–17.2%) | 30 | 137 | 37 | 2 | 0.69 s | 1.32 s |
| ElevenLabs | 11 (5.5%; 3.1–9.6%) | 103 | 92 | 97 | 0 | 1.67 s | 1.81 s |

ElevenLabs cuts in least, but waits about 1.7 s after every turn.

### Head to head: real cutoffs on the same turns (generous scoring for the service)

| Turn-aware Qwen3-ASR vs | Only ours cut in | Only the service cut in | Both | Sign-test p |
|---|---|---|---|---|
| AssemblyAI `balanced` | 6 | 21 | 9 | **0.006** |
| AssemblyAI `balanced-control` | 5 | 22 | 10 | **0.002** |
| AssemblyAI `min_latency` | 5 | 17 | 10 | **0.017** |
| AssemblyAI `max_accuracy` | 9 | 10 | 6 | 1.00 |
| Deepgram | 6 | 15 | 9 | 0.078 |

### What the cutoffs look like

The two systems fail on different kinds of pauses. Examples from the same 200
turns, showing the caller's words at the moment of the fire:

**AssemblyAI `balanced` cut in, Turn-aware Qwen3-ASR waited** (21 turns):

| Agent had asked | Caller had said |
|---|---|
| "How can I assist you today?" | `Hi, I need to` |
| "Can I get your booking confirmation number and the exact charge…" | `Confirmation number DX4958. Charge was` |
| "Can I get your renewal application reference number, please?" | `Oh, yes, let` |

These are grammatically unfinished. The words alone say the caller is not done.

**Turn-aware Qwen3-ASR cut in, AssemblyAI waited** (6 turns):

| Agent had asked | Caller was in the middle of |
|---|---|
| "…could you give me the phone number on the account?" | `Yes, my phone number is 585.` · pause · `585-813-…` |
| "How can I assist you?" | `…this is Stanislaw Skrzypczak, S-K-R-Z-…` |
| "…can you give me your order confirmation number first?" | `Yeah it's W5318K2, that's the order confirmation code.` |

Most of the model's own cutoffs land at the short pauses inside a number,
spelling or code, where the words so far can read as finished (`…my phone number is 585.`).

### Offline decision accuracy

Scored on a 2,235-example sample built from 1,500 random validation
observations. Each example is one prefix plus a controlled silence, and the
model must fire or hold.

- **Precision:** of the times it fired, the share where the caller had finished.
- **Recall:** of the times the caller had finished and gone quiet, the share
  where it fired.
- **Holds correct:** on unfinished prefixes followed by silence, the share where
  it correctly waited.
- **Unfinished pauses:** real pauses mined from the audio that Claude judged
  unfinished from the words so far.

| Metric | Turn-aware Qwen3-ASR |
|---|---|
| Fire precision | 0.990 |
| Fire recall | 0.986 |
| Holds correct | 0.969 |
| `payload_cut` correct (cut just before the requested detail) | 0.903 |
| `word_cut` correct (cut mid-phrase) | 0.961 |
| Never fires before silence has been heard | 1.000 |
| Fires at unfinished mid-sentence pauses (1,572) | 8.1% |
| Fires at unfinished within-chunk pauses (218) | 8.3% |
| Transcription drift from the base model | 1.1% WER |

## A context-conditioned variant we rejected

We also trained the same recipe with the agent's last question as context on
every example: 20% of those were distractor questions from other calls, and the
transcripts were made with that context. Offline it looked marginally better
(holds correct 0.975, unfinished mid-sentence pauses 7.4%). In streaming, it cut
callers off on **30 of 200 turns against 15** (paired: 16 turns only the
variant, 1 only Turn-aware Qwen3-ASR).

With almost no speech to go on (pure silence, or the first 0.02–0.16 s of a
word), the variant writes the agent's question as the transcript and fires. Its
training transcripts were clean (7 of 91,798 contain an echoed question); the
cause was that it saw the question on every speech example and never on silence.
The offline evaluation has no context-on-silence rows, so it could not catch
this. The streaming runtime now waits for speech onset before decoding. That
removes the pure-silence case, but not the near-silence one.

## Where Turn-aware Qwen3-ASR still errs

- **Short pauses inside numbers, spellings and codes** (most of its streaming
  cutoffs): `Yes, my phone number is 585.` · pause · `585-813-…`. The words so
  far read as a finished sentence, and the pause is long enough to fire on.
- **Complete-sounding openers the caller extends:** `Yes.`, `Yes, of course.`,
  `Good afternoon. This is Robert Rutherford.` Whether these end the turn
  depends on what the agent asked, which the model does not use. The
  context-conditioned variant above was the attempt to fix this, and it did not
  help.
- **Early fires after a complete sentence** (156 of 200 turns). This is by label
  design, and no endpointer can tell that moment apart from a true turn end
  without waiting longer. Requiring more silence (`replay --gate-s`) trades
  latency for fewer of these. In a sweep on an earlier checkpoint of this recipe
  (40-turn sample), requiring 1.0 s of silence left the real cutoffs unchanged
  and raised median latency to 0.83 s.

## Caveats

- **In-domain test.** The turns are synthetic LLM-written, TTS-spoken customer
  service calls, the same distribution the model was trained on. The commercial
  systems are general-purpose, and none of this measures real phone audio, noise
  or accents.
- **Latency is not like for like.** Turn-aware Qwen3-ASR's numbers come from an
  offline replay and exclude model compute and network. In production, add
  roughly 0.1–0.2 s (an estimate, not measured). The services were measured
  streaming in real time.
- **Sample size.** 200 of 1,030 turns: enough for the paired AssemblyAI results,
  not for the Deepgram comparison. The offline table is a local validation sample
  (about 15% of the split). Full-split numbers should come from the pod run
  below.
- **Transcription accuracy is not compared.** The services' transcripts aren't
  part of this benchmark.
- **The classification script isn't in the repo yet.** The replay and the
  commercial arms' fire times are reproducible with the commands below; the
  step that classifies each first fire against the chunk structure was run as a
  one-off analysis.
- **The cutoff taxonomy is approximate.** It uses the script's chunk
  punctuation. A pause inside a chunk can follow a complete sentence, so
  "inside a chunk" slightly overcounts real cutoffs, equally for every system.

## Reproduce

```bash
# 200-turn streaming replay (no context) + the commercial arms on the same turns
poetry run ta turn-aware replay -m mazesmazes/tiny-audio-turn-aware-qwen3-asr-v3 \
  -n 200 --seed 0 --no-context -o outputs/turn_aware_local/replay200

# Offline decision accuracy on the recipe-neutral eval pool
poetry run ta turn-aware build-pool +experiment=eval --split validation
poetry run ta turn-aware evaluate -m mazesmazes/tiny-audio-turn-aware-qwen3-asr-v3 \
  --split validation -n 0 --context never -o outputs/turn_aware_local/eval

# Full test split on a pod
ta runpod eval-turn-aware <host> <port> -m mazesmazes/tiny-audio-turn-aware-qwen3-asr-v3 --context never
```
