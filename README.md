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

```python
# 1. Transcribe with the published model (the first call downloads ~6 GB)
#    pip install "transformers>=5.0" peft torch torchaudio librosa
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition", model="mazesmazes/tiny-audio", trust_remote_code=True
)
print(pipe("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")["text"])
# I have a dream that one day this nation will rise up and live out the true meaning of its creed.
```

Pass a path to transcribe your own recording. WAV, FLAC, OGG and MP3 work out of the box; other
formats (m4a, video) need [ffmpeg](https://ffmpeg.org/download.html) installed.

```bash
# 2. Train your own for free, on your laptop: ~30 minutes on an Apple Silicon Mac, ~3 GB download
#    Needs Python 3.12 and Poetry (pip install poetry)
git clone https://github.com/alexkroman/tiny-audio.git && cd tiny-audio
poetry install
poetry run ta train asr -e tiny
poetry run ta eval -m outputs/tiny --local-code -d librispeech -n 20   # read its transcripts

# 3. The full run on one rented GPU
poetry run ta runpod plan               # VRAM and disk the config needs, before you rent anything
poetry run python scripts/train.py      # the default config: the published model's recipe
```

The free run ([configs/experiments/tiny.yaml](configs/experiments/tiny.yaml)) is the default recipe
with the smaller Qwen3.5-0.8B decoder, trained on LibriSpeech dev-clean (5.4 hours) instead of the
full mix. Watch the loss: it sits near 4 while the projector's output is still noise to the decoder,
then drops below 1 around step 500 as the decoder starts reading the audio. The result transcribes
unseen LibriSpeech audio at about 16% WER, with mistakes like "furnishings" for "furniture". It fits
in 16 GB of memory and also runs on a CUDA GPU, including a free Colab T4.

What runs where:

| Step                         | Apple Silicon (16 GB) | Any CUDA GPU (incl. Colab T4) | CPU only  | One 80 GB GPU |
| ---------------------------- | --------------------- | ----------------------------- | --------- | ------------- |
| 1. Transcribe                | Yes                   | Yes                           | Yes, slow | Yes           |
| 2. Free run (`-e tiny`)      | Yes, ~30 min          | Yes                           | Slow      | Yes           |
| 3. Full run (default config) | No                    | Needs 80 GB                   | No        | Yes, a day+   |

No install at all: the **[live demo](https://huggingface.co/spaces/mazesmazes/tiny-audio)** lets you
record yourself or upload a file.

## Read this first

The idea fits in one class. **[tiny_audio/projectors.py](tiny_audio/projectors.py)** holds the
projector, the only part trained from scratch: frame stacking plus a two-layer MLP, about twenty
lines of code under a docstring that records what was tried and why it was rejected. Read it first.

```text
Audio (16 kHz) → speech encoder (frozen) → MLP projector (trained) → LLM decoder (LoRA) → Text
```

1. A pretrained **speech encoder** turns audio into a sequence of frame embeddings.
1. The **projector** stacks neighbouring frames and maps them into the LLM's embedding space.
1. The **LLM** reads those projected frames as if they were words and writes out the transcript.

Then follow one clip through [tiny_audio/asr_modeling.py](tiny_audio/asr_modeling.py):
`_encode_audio` runs the encoder and the projector, `forward` splices the audio embeddings into the
prompt and computes the loss, and `generate` decodes. That file is the model that was trained and
published, so it is longer: it also handles batching, LoRA, memory, and several encoder and decoder
families.

## The course

Each chapter is a section of the [course](https://alexkroman.github.io/tiny-audio/) with code you
run and something concrete to check.

| #   | Chapter                                                                                                                 | What you build or learn                                         | Check                            |
| --- | ----------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | -------------------------------- |
| 00  | [What a speech LLM is](https://alexkroman.github.io/tiny-audio/1-introduction-and-setup/#the-architecture)              | Encoder → projector → decoder, and why reusing two models works | —                                |
| 01  | [Audio in, features out](https://alexkroman.github.io/tiny-audio/1-introduction-and-setup/#from-sound-to-a-spectrogram) | Waveform → spectrogram → encoder frames, with real shapes       | Frame counts per second of audio |
| 02  | [The projector](https://alexkroman.github.io/tiny-audio/1-introduction-and-setup/#how-the-projector-works)              | Frame stacking and the two-layer MLP, with real shapes          | Shapes and parameter count       |
| 03  | [The decoder](https://alexkroman.github.io/tiny-audio/1-introduction-and-setup/#how-the-decoder-sees-audio)             | How an LLM reads audio tokens next to text tokens               | —                                |
| 04  | [Run inference](https://alexkroman.github.io/tiny-audio/1-introduction-and-setup/#part-b-hands-on-40-min)               | Install, transcribe, timestamps, explore the CLI                | Your first transcript            |
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

| Run                                  | Data                                 | Hardware                        | Cost                |
| ------------------------------------ | ------------------------------------ | ------------------------------- | ------------------- |
| **Free run** (`-e tiny`)             | LibriSpeech dev-clean (5.4 hours)    | Your laptop or a free Colab GPU | Free                |
| **Course run**                       | LoquaciousSet `small` (~250 hours)   | One rented GPU, a few hours     | A few GPU-hours     |
| **Default config** (published model) | ~3M clips across ten corpora (>1 TB) | One 80 GB GPU, a day or more    | Hundreds of dollars |

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
tiny_audio/      the model: projector, encoder/decoder glue, pipeline, diarization
scripts/train.py the training loop (Hydra + HF Trainer)
configs/         the default recipe, its data mix, and the free `tiny` preset
scripts/         the `ta` CLI: eval, serve, push, deploy, runpod
docs/course/     the course
demo/            the Hugging Face Space
```

Every training setting is a [Hydra](https://hydra.cc/) override, for example
`model.projector_hidden_dim=2048`. `ta push` publishes a model to the Hub and `ta deploy` puts a
demo like the one above on a Space.

## Stuck?

| Problem                                     | Fix                                                   |
| ------------------------------------------- | ----------------------------------------------------- |
| `poetry install` refuses the Python version | Install Python 3.12, then `poetry env use python3.12` |
| Model download fails                        | `poetry run hf auth login`, or export `HF_TOKEN`      |
| `flash_attention_2` warning                 | Expected off CUDA; the model falls back to SDPA       |
| Out of GPU memory during the free run       | Halve the batch and keep the effective batch of 4:    |

```bash
poetry run python scripts/train.py +experiments=tiny \
    training.per_device_train_batch_size=1 training.gradient_accumulation_steps=4
```

Anything else: check
[Common issues](https://alexkroman.github.io/tiny-audio/4-quick-reference/#common-issues), then
[open an issue](https://github.com/alexkroman/tiny-audio/issues) with the command you ran and its
output.

## Contributing

Want to try a new projector, add a dataset, or change the codebase? See
[CONTRIBUTING.md](CONTRIBUTING.md) for the CLI reference, config layout, and quality gates.

## Acknowledgments

- [Granite Speech](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) and
  [GLM-ASR](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) for audio encoding
- [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B) and
  [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B) for language modeling
- [LoquaciousSet](https://huggingface.co/datasets/speechbrain/LoquaciousSet) for the default
  evaluation set
- [nanoGPT](https://github.com/karpathy/nanoGPT), whose small-and-readable approach this repo
  borrows

## License

MIT
