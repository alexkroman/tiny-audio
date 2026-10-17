# tiny-audio

<div align="center">
  <img
    src="https://raw.githubusercontent.com/alexkroman/tiny-audio/main/public/logo.png"
    alt="tiny-audio logo"
  />
</div>

**Build a speech-to-text LLM from scratch, and train it on one GPU.**

tiny-audio is to speech what nanoGPT is to text: a small, readable, hackable speech LLM (a
pretrained audio encoder, a tiny MLP projector, and an open-source LLM decoder) plus a course that
teaches you to build, train, evaluate, and ship it yourself. The model trained with this repo's
default recipe gets **1.8% WER on LibriSpeech test-clean** while training only ~80M parameters.

It is deliberately simple. For production speech recognition, use a mature toolkit or a hosted API.
This repo is for understanding how the pieces fit, and for running cheap experiments of your own.

*tiny-audio is a personal side project, independent of my work at AssemblyAI.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97-mazesmazes%2Ftiny--audio-yellow)](https://huggingface.co/mazesmazes/tiny-audio)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97-Live%20Demo-blue)](https://huggingface.co/spaces/mazesmazes/tiny-audio)

**Course:** [alexkroman.github.io/tiny-audio](https://alexkroman.github.io/tiny-audio/) · **Live
demo:**
[huggingface.co/spaces/mazesmazes/tiny-audio](https://huggingface.co/spaces/mazesmazes/tiny-audio) ·
**Model:** [mazesmazes/tiny-audio](https://huggingface.co/mazesmazes/tiny-audio)

## Quickstart

```bash
git clone https://github.com/alexkroman/tiny-audio.git && cd tiny-audio

# 1. The whole idea in one file: train a projector on 73 LibriSpeech clips, then transcribe.
#    Runs on a laptop (CPU, Apple Silicon, or CUDA).
pip install torch transformers "datasets[audio]<4" librosa soundfile
python model.py
```

```python
# 2. Transcribe with the published model
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition", model="mazesmazes/tiny-audio", trust_remote_code=True
)
print(pipe("audio.wav")["text"])
# The quarterly revenue grew by 12% according to Dr. Smith.
```

```bash
# 3. Train the full recipe on one rented GPU
poetry install
poetry run ta runpod plan               # VRAM and disk the config needs, before you rent anything
poetry run python scripts/train.py      # the default config: the published model's recipe
```

No install at all: the **[live demo](https://huggingface.co/spaces/mazesmazes/tiny-audio)** lets you
record yourself or upload a file.

## Read this first

The whole model is in **[model.py](model.py)**: about 250 lines, heavily commented, no clever
tricks. If you read one file in this repo, read that one. The `Projector` class near the top is the
entire idea, and it takes five minutes:

```text
Audio (16 kHz) → speech encoder (frozen) → MLP projector (trained) → LLM decoder (frozen) → Text
```

1. A pretrained **speech encoder** turns audio into frame embeddings, one every 20 ms.
1. The **projector** stacks four neighbouring frames and maps them into the LLM's embedding space.
   It is the only part trained from scratch.
1. The **LLM** reads those projected frames as if they were words and writes out the transcript.

`model.py` uses Whisper-small's encoder and Qwen3-0.6B so it fits on a laptop. Once it makes sense,
the production version is in [`tiny_audio/`](tiny_audio/): the same three parts with a Granite
Speech encoder, Qwen3.5-2B with LoRA, batching, timestamps, diarization, and streaming.

## The course

Each chapter is a section of the [course](https://alexkroman.github.io/tiny-audio/) with code you
run and something concrete to check.

| #   | Chapter                                                                                                                 | What you build or learn                                         | Check                            |
| --- | ----------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | -------------------------------- |
| 00  | [What a speech LLM is](https://alexkroman.github.io/tiny-audio/1-introduction-and-setup/#the-architecture)              | Encoder → projector → decoder, and why reusing two models works | —                                |
| 01  | [Audio in, features out](https://alexkroman.github.io/tiny-audio/1-introduction-and-setup/#from-sound-to-a-spectrogram) | Waveform → spectrogram → encoder frames, with real shapes       | Frame counts per second of audio |
| 02  | [The projector](https://alexkroman.github.io/tiny-audio/1-introduction-and-setup/#how-the-projector-works)              | Frame stacking and the two-layer MLP, in `model.py`             | `python model.py`: loss falls    |
| 03  | [The decoder](https://alexkroman.github.io/tiny-audio/1-introduction-and-setup/#how-the-decoder-sees-audio)             | How an LLM reads audio tokens next to text tokens               | —                                |
| 04  | [Run inference](https://alexkroman.github.io/tiny-audio/1-introduction-and-setup/#part-b-hands-on-50-min)               | Install, transcribe, timestamps, trace a clip through the code  | Your first transcript            |
| 05  | [Data](https://alexkroman.github.io/tiny-audio/2-training/#data-make-the-labels-agree)                                  | Why the label conventions decide the result                     | —                                |
| 06  | [Training on one GPU](https://alexkroman.github.io/tiny-audio/2-training/)                                              | Size a GPU, write a config, launch on RunPod, read the curves   | Loss and gradient curves in W&B  |
| 07  | [Evaluating](https://alexkroman.github.io/tiny-audio/3-evaluation-and-deployment/#word-error-rate-wer)                  | WER, normalization, error analysis, commercial baselines        | Your WER next to the table below |
| 08  | [Shipping it](https://alexkroman.github.io/tiny-audio/3-evaluation-and-deployment/#part-b-hands-on-45-min)              | Publish to the Hub, deploy a demo, run the HTTP server          | A live demo of your own model    |

Coming next: per-speaker transcription (diarization), what changed between releases and why, and
where to go after this (bigger encoders, streaming, post-training).

## Expected results

Word error rate (%, lower is better) of the published model, trained with the default config, on
11,822 samples (up to 1,000 per dataset), measured with this repo's `ta eval` against the `ta serve`
HTTP API on an RTX 4090. A course run trains on far less data, so expect a higher WER from yours.

| Dataset                     |      WER |
| --------------------------- | -------: |
| LibriSpeech test-clean      |     1.84 |
| SPGISpeech                  |     2.29 |
| TED-LIUM                    |     3.71 |
| LoquaciousSet †             |     6.20 |
| LibriSpeech test-other      |     6.38 |
| VoxPopuli                   |     7.11 |
| Common Voice                |     7.18 |
| AMI (IHM)                   |     8.99 |
| GigaSpeech                  |     9.06 |
| Earnings22 †                |    10.58 |
| People's Speech             |    17.59 |
| AMI (SDM)                   |    23.53 |
| **Mean (12 sets)**          | **8.71** |
| **Pooled (11,822 samples)** | **7.42** |

† Held out: no data from this source was used in training.

Check these numbers yourself, and compare against commercial APIs on the same samples:

```bash
poetry run ta eval -m mazesmazes/tiny-audio -d loquacious -n 100
# Same samples through a commercial API (also: deepgram, elevenlabs, apple-speech)
ASSEMBLYAI_API_KEY=... poetry run ta eval -m assemblyai -d loquacious -n 100
```

What each training budget costs:

| Run                                  | Data                                 | Hardware                     | Cost                |
| ------------------------------------ | ------------------------------------ | ---------------------------- | ------------------- |
| **`model.py`**                       | 73 LibriSpeech clips                 | Your laptop                  | Free                |
| **Course run**                       | LoquaciousSet `small` (~250 hours)   | One rented GPU, a few hours  | A few GPU-hours     |
| **Default config** (published model) | ~3M clips across ten corpora (>1 TB) | One 80 GB GPU, a day or more | Hundreds of dollars |

## Using the published model

```python
# Word-level timestamps (forced alignment)
pipe("audio.wav", return_timestamps=True)
# {"text": "hello world", "words": [{"word": "hello", "start": 0.0, "end": 0.5}, ...]}

# Who spoke when (speaker diarization)
pipe("meeting.wav", return_speakers=True, num_speakers=2)
```

Transcripts come punctuated, capitalized, and with numbers formatted. The input can be a file path,
a URL, or a 16 kHz numpy array; the bf16 weights need roughly 6 GB of GPU or Apple Silicon memory.
Speaker diarization needs `transformers` installed from `main`
(`pip install git+https://github.com/huggingface/transformers`) until the next release. For
token-by-token output, see [`ASRModel.generate_streaming`](tiny_audio/asr_modeling.py).

- **HTTP server:** `ta serve` batches concurrent requests on one GPU, locally or on RunPod. See
  [docs/SERVING.md](docs/SERVING.md).
- **Batching and GPU settings:** the [model card](https://huggingface.co/mazesmazes/tiny-audio).

## Repo layout

```text
model.py         the whole model in one file: read this first
tiny_audio/      the production model: projector, encoder/decoder glue, pipeline, diarization
scripts/train.py the training loop (Hydra + HF Trainer)
configs/         the default recipe and its data mix
scripts/         the `ta` CLI: eval, serve, push, deploy, runpod
docs/course/     the course
demo/            the Hugging Face Space
```

Every training setting is a [Hydra](https://hydra.cc/) override, for example
`model.projector_hidden_dim=2048`. `ta push` publishes a model to the Hub and `ta deploy` puts a
demo like the one above on a Space.

## Contributing

Want to try a new projector, add a dataset, or change the codebase? See
[CONTRIBUTING.md](CONTRIBUTING.md) for the CLI reference, config layout, and quality gates.

## Acknowledgments

- [Granite Speech](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) and
  [Whisper](https://huggingface.co/openai/whisper-small) for audio encoding
- [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B) and
  [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B) for language modeling
- [LoquaciousSet](https://huggingface.co/datasets/speechbrain/LoquaciousSet) for the default
  evaluation set
- [nanoGPT](https://github.com/karpathy/nanoGPT) and
  [micrograd](https://github.com/karpathy/micrograd), whose one-readable-file approach this repo
  borrows

## License

MIT
