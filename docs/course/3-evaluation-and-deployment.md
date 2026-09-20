# Class 3: Evaluation and Deployment

*1 hour (15 min lecture + 45 min hands-on)*

**Goal**: Measure your model, find out *where* it fails, and put it in front of other people.

> **Before you start.** You need a model on the Hugging Face Hub. If your Class 2 run has
> pushed at least one checkpoint, use that. If it hasn't finished, or you skipped the cloud
> run, use the published model instead; every command below works the same way.
>
> ```bash
> export MODEL=your-username/tiny-audio-yourname   # or: mazesmazes/tiny-audio
> ```
>
> The commands below refer to `$MODEL`, and to `$NAME` for its short name (the part after the
> slash, e.g. `tiny-audio-yourname`), which is how the analysis tools identify it.

---

## Part A: Lecture (15 min)

### Word Error Rate (WER)

```
WER = (Substitutions + Insertions + Deletions) / Words in the reference
```

Align the prediction against the reference word by word, count the edits, divide by the
reference length. A WER of 10% means roughly one word in ten is wrong. It can exceed 100% if
the model hallucinates a lot of extra words.

| WER | Quality |
|-----|---------|
| < 5% | Excellent. Commercial APIs on clean speech |
| 5-10% | Very good |
| 10-20% | Good. A realistic target for the course run |
| > 30% | Poor. Something is wrong with the model or the data |

**Normalization matters.** Before scoring, both reference and prediction go through Whisper's
`EnglishTextNormalizer`: lowercase, punctuation stripped, numbers and common spellings
standardized ("twenty five dollars" and "$25" agree). Without this, a perfect transcript with
different comma placement would count as errors. It also means WER says nothing about your
model's punctuation or capitalization. The raw pair is saved alongside the normalized one so
you can inspect those by hand.

### Evaluation Datasets

Tiny Audio's registry covers 13 test sets. Each stresses something different:

| Name (`-d`) | Domain | Why it's hard |
|-------------|--------|---------------|
| `loquacious` | Mixed read and spontaneous English (default) | Broad benchmark |
| `librispeech`, `librispeech-other` | Audiobooks | Clean baseline; `other` is harder speakers |
| `tedlium` | TED talks | Presentational speech, technical vocabulary |
| `commonvoice` | Crowd-sourced read sentences | Thousands of speakers, accents, cheap mics |
| `voxpopuli` | European Parliament | Non-native accents |
| `peoples` | Public-domain speech | Varied recording quality |
| `gigaspeech` | Podcasts and YouTube | Conversational, noisy |
| `earnings22` | Earnings calls | Financial jargon, names, numbers, phone-quality audio |
| `spgispeech` | Financial presentations | Formatted numbers and entities |
| `ami`, `ami-sdm` | Meetings | Overlapping speakers; `sdm` is a single distant microphone |
| `expresso` | Expressive read speech | Emotional and whispered speech |

A model that does well on LibriSpeech and badly on AMI is normal. A model that does badly on
LibriSpeech has a problem.

### Error Analysis

An aggregate WER tells you *how much* the model fails. To improve it you need to know *how*:

- **Worst samples**: sort by per-sample WER. The top of the list is usually noisy audio,
  mislabeled references, or a mode failure (empty output, endless repetition).
- **Entity errors**: names, places, organizations, and numbers. These carry the most meaning
  and are the hardest for a small decoder to spell.
- **Comparison**: the same clips through another model. If everyone fails on a clip, blame the
  clip.

### Deployment Options

| Option | Cost | When |
|--------|------|------|
| **Hub repo** | Free | Already done: training pushed your checkpoints there |
| **Hugging Face Space** (Gradio) | Free on CPU | A public demo anyone can try in a browser |
| **Inference Endpoints** | Paid GPU | A production HTTP API; the repo ships the handler |
| **Local server** | Your hardware | Privacy, or wiring into your own app |

---

## Part B: Hands-On (45 min)

### Exercise 1: Evaluate (10 min)

```bash
# Default benchmark: LoquaciousSet test split
poetry run ta eval -m $MODEL -n 200

# A second domain
poetry run ta eval -m $MODEL -d earnings22 -n 100

# Several at once
poetry run ta eval -m $MODEL -d loquacious -d tedlium -d ami -n 100
```

Each sample prints as it's scored:

```
Sample 1: WER=8.3%, Time=1.23s
  Ref:  the quick brown fox jumps over the lazy dog
  Pred: the quick brown fox jumped over the lazy dog
```

and each dataset ends with a summary table (WER, sample count, average time per clip).

Results land in `outputs/<timestamp>_<short-name>_<dataset>/`:

- `results.txt`: every sample with its WER, the normalized reference and prediction, and the
  raw (unnormalized) pair
- `metrics.txt`: the corpus-level numbers

The analysis commands read these directories, so don't delete them.

Score the published model on the same datasets so you have a comparison point:

```bash
poetry run ta eval -m mazesmazes/tiny-audio -n 200
```

### Exercise 2: Analyze Errors (15 min)

**Worst samples.** The model pattern is the short name, matched exactly:

```bash
poetry run ta analysis high-wer $NAME --threshold 50
poetry run ta analysis high-wer $NAME --threshold 30 --latest --output-file worst.md
```

Read a dozen of the worst. Sort them into buckets: bad audio, bad reference label, rare
vocabulary, model mode failure. The bucket that dominates tells you what to fix.

**Compare models.** Any models you've evaluated, by short name:

```bash
poetry run ta analysis compare $NAME tiny-audio
```

This prints a WER table per dataset, plus breakdowns by clip length and by entity type.

**Entity errors.** Build the entity index from the references in your eval runs, then list the
samples where your model got a named entity wrong:

```bash
poetry run ta analysis extract-entities
poetry run ta analysis entity-errors $NAME
poetry run ta analysis entity-errors $NAME --entity-type PERSON
```

**Inspect the weights.** If a run misbehaved, check whether training moved the decoder a
healthy amount and whether the projector's scale drifted:

```bash
poetry run ta debug analyze-weights $MODEL
poetry run ta debug compare-to-base $MODEL     # drift from Qwen3-0.6B, layer by layer
poetry run ta debug analyze-lora $MODEL        # only if you trained with LoRA
```

### Exercise 3: Check Your Hub Repo (5 min)

Open `https://huggingface.co/$MODEL`. Training pushed each checkpoint there as it was saved,
so the repo already contains:

- `model.safetensors`: the trained weights only, meaning the projector and the fine-tuned
  decoder. The frozen encoder is not stored; `config.json` names it and it is downloaded from
  its own repo at load time
- `config.json`: the `ASRConfig`, including which encoder and decoder to load
- `asr_modeling.py`, `projectors.py`, and the other custom code files that make
  `trust_remote_code=True` work
- Tokenizer and feature-extractor files

Two things it does *not* have yet: a model card, and any code fixes you made after training
started. `ta push` uploads exactly those. It needs a Hub write token, read from `HF_TOKEN` or
passed as `--hf-token`:

```bash
poetry run ta push --repo-id $MODEL
```

It stages the `tiny_audio/*.py` files, the repo's `MODEL_CARD.md` as the Hub `README.md`, and
`requirements.txt`, then uploads them. It does **not** upload weights. Edit `MODEL_CARD.md` to
describe your model before you push it.

Test the result the way a stranger would:

```python
from transformers import pipeline
pipe = pipeline("automatic-speech-recognition", model="your-username/tiny-audio-yourname", trust_remote_code=True)
print(pipe("audio.wav")["text"])
```

### Exercise 4: Deploy a Demo to Hugging Face Spaces (10 min)

The `demo/` directory is a complete Gradio app. It reads the model ID from the `MODEL_ID`
environment variable, so the same code serves any Tiny Audio model.

1. Edit `demo/README.md`. Its front matter is the Space's card: set `title`, and change the
   `models:` list and `preload_from_hub:` to your model ID.
2. Deploy. The command creates the Space if it doesn't exist:

   ```bash
   poetry run ta deploy --repo-id your-username/tiny-audio-demo
   ```

3. In the Space's **Settings → Variables**, add `MODEL_ID` = `your-username/tiny-audio-yourname`.
   The Space restarts. (Without this it serves the published model.)

The first build takes a few minutes on the free CPU tier. Inference on CPU is slow but works.
Share the link.

To test locally before deploying:

```bash
poetry run ta demo --model $MODEL --port 7860
```

### Exercise 5: Production Endpoints (5 min)

For an HTTP API on a GPU, use Inference Endpoints. The repo's `handler.py` is uploaded with
the custom code, so the endpoint knows how to load and warm up the model:

1. On your model page, choose **Deploy → Inference Endpoints**
2. Pick a GPU and a scaling policy (scale-to-zero keeps idle cost near nothing)
3. Create it, then call it with any HTTP client, or evaluate through it by passing the
   endpoint URL as the model:

   ```bash
   poetry run ta eval -m https://<your-endpoint>.endpoints.huggingface.cloud --endpoint -n 50
   ```

---

## Advanced Evaluation

### Comparing with Commercial APIs

```bash
export ASSEMBLYAI_API_KEY='...'
poetry run ta eval -m assemblyai -d loquacious -n 200 -w 4          # universal-3-pro by default

export DEEPGRAM_API_KEY='...'
poetry run ta eval -m deepgram -d loquacious -n 200 -w 4            # nova-3

export ELEVENLABS_API_KEY='...'
poetry run ta eval -m elevenlabs -d loquacious -n 200 -w 4          # scribe-v2

# macOS only: Apple's on-device recognizer
poetry run ta eval -m apple-speech -d loquacious -n 200
```

`-w` runs API calls in parallel. Then `ta analysis compare $NAME assemblyai deepgram` puts them
side by side.

### Every Dataset at Once

```bash
poetry run ta eval -m $MODEL -d all -n 100
```

### Running Evaluation on the Pod

If you still have a RunPod instance up, evaluation is much faster there:

```bash
poetry run ta runpod eval <HOST> <PORT> -m $MODEL -d loquacious -d ami -n 500
```

---

## Debugging Poor Performance

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| High WER everywhere | Undertrained | Check the loss curve; train longer or on more data |
| High WER on one domain | Domain gap | Add that domain's data to your data config |
| High WER on accented speech | Training data bias | Add CommonVoice or VoxPopuli to the mix |
| Empty or one-word outputs | Audio too quiet, or the run collapsed | Inspect the clips; check `analyze-weights` |
| Runaway repetition | Decoder loop | The pipeline truncates repeats; check `max_new_tokens`; more training usually fixes it |
| Great eval loss, bad WER | Mismatch between eval split and test set | Compare label formats; check normalization |
| Wrong casing or punctuation but good WER | Training labels lacked them | Expected with LoquaciousSet; add cased, punctuated data |

---

## Congratulations

You now have:

- A speech recognition model you trained, on the Hub, loadable with three lines of Python
- WER numbers on several domains, and a comparison against the published model
- A list of its worst failures and what kind they are
- A public demo

**Where to go next:**

- Retrain with a change and compare: a wider projector (`model.projector_hidden_dim=2048`), a
  different stride (`model.projector_pool_stride=2`), or LoRA instead of full fine-tuning
- Add a second dataset to your data config, targeting the domain where you failed worst
- Read `configs/data/multiasr.yaml` to see how the production mix was assembled, and why
  corpora were added and removed
- Try a different encoder or decoder; `granite_qwen.yaml` shows how little changes
- Build something with the model: the `tiny_audio/integrations/` directory has a voice-agent
  integration to start from

---

[← Class 2](./2-training.md) | [Quick Reference →](./4-quick-reference.md)
