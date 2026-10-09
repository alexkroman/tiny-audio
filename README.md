# Tiny Audio

<div align="center">
  <img
    src="https://raw.githubusercontent.com/alexkroman/tiny-audio/main/public/logo.png"
    alt="Tiny Audio Logo"
  />
</div>

**A tiny, hackable, open-source speech LLM.**

Tiny Audio connects a frozen, pretrained speech encoder to a pretrained LLM with a small trainable
projector. The model published from this repo gets **1.8% WER on LibriSpeech test-clean and 8.4%
averaged over 12 benchmarks** while training only ~80M parameters. The codebase is small enough to
read in an afternoon, and you can run a training loop on your laptop in about five minutes.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97-mazesmazes%2Ftiny--audio-yellow)](https://huggingface.co/mazesmazes/tiny-audio)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97-Live%20Demo-blue)](https://huggingface.co/spaces/mazesmazes/tiny-audio)

## Try it in 30 seconds

**No install:** open the **[live demo](https://huggingface.co/spaces/mazesmazes/tiny-audio)**,
record yourself or upload a file, and get a transcript.

**In Python:**

```bash
pip install "transformers>=5.0" peft torch torchaudio librosa
```

```python
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition", model="mazesmazes/tiny-audio", trust_remote_code=True
)
print(pipe("audio.wav")["text"])
# The quarterly revenue grew by 12% according to Dr. Smith.
```

The output is punctuated, capitalized, and has numbers formatted, with no post-processing step. The
input can be a file path, a URL, or a 16 kHz numpy array. Weights are bf16, so you need roughly 6 GB
of GPU or Apple Silicon memory.

### More than plain text

```python
# Word-level timestamps (forced alignment)
pipe("audio.wav", return_timestamps=True)
# {"text": "hello world", "words": [{"word": "hello", "start": 0.0, "end": 0.5}, ...]}

# Who spoke when (speaker diarization)
pipe("meeting.wav", return_speakers=True, num_speakers=2)
```

Each speaker Nemotron-3-Diarization finds is transcribed separately, on a copy of the audio where
everyone else is silenced, and each word belongs to the stream it came from. This is a zero-shot
port of NeMo's `masked_asr` recipe; a word two streams both heard at once is kept once. Each speaker
costs roughly their own talk time in ASR, and single-speaker audio is transcribed unmasked. Overlap
is only partly handled: another person's speech inside a speaker's turn stays in that speaker's
stream.

Speaker diarization needs `transformers` installed from `main`
(`pip install git+https://github.com/huggingface/transformers`) until the next release. For
token-by-token streaming output, see [`ASRModel.generate_streaming`](tiny_audio/asr_modeling.py).
The [model card](https://huggingface.co/mazesmazes/tiny-audio) covers batching and GPU settings.

### As an HTTP API on RunPod

`ta serve` puts the model behind a batched HTTP server: requests arriving together share GPU
batches, so throughput grows with load (about 460x real time at 128 concurrent requests on an RTX
4090). To run it on a RunPod GPU:

```bash
poetry run ta runpod up --serve                 # create an inference pod; prints <POD_ID>
poetry run ta runpod wait <POD_ID>              # prints <HOST> <PORT>
poetry run ta runpod deploy <HOST> <PORT>       # sync the project, install the fast kernels
TINY_AUDIO_API_KEY=my-secret poetry run ta runpod serve <HOST> <PORT> --no-attach
# Ready when https://<POD_ID>-8000.proxy.runpod.net/health answers (a few minutes: it compiles first)
```

Without `TINY_AUDIO_API_KEY` the server is open to anyone who has the URL. `ta serve` also runs
locally, on CUDA, Apple Silicon, or CPU.

Send the audio as the request body, with options in the query string:

```bash
curl -X POST "https://<POD_ID>-8000.proxy.runpod.net/?return_timestamps=true" \
  -H "Authorization: Bearer my-secret" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav
```

```python
import httpx

response = httpx.post(
    "https://<POD_ID>-8000.proxy.runpod.net/",
    params={"return_speakers": "true", "num_speakers": "2"},
    content=open("meeting.wav", "rb").read(),
    headers={"Authorization": "Bearer my-secret"},
    timeout=600,
)
print(response.json()["text"])
```

- **Options:** `return_timestamps`, `return_speakers`, `num_speakers` and `max_speakers`, as in the
  pipeline. The response is the same dict the pipeline returns.
- **JSON body:** to send JSON instead, use `{"inputs": "<base64 audio>", "parameters": {...}}`.
- **Audio formats:** anything FFmpeg can read.
- **Errors:** `400` with `{"error": ...}` for bad audio or options, and `401` for a wrong key.
- **Other endpoints:** `GET /health` and `GET /stats` (batch sizes and GPU time).

RunPod's HTTP proxy rejects request bodies over 500 MiB, and it drops any request that takes more
than 100 seconds. For long recordings, send 16 kHz mono FLAC:

```bash
ffmpeg -i recording.wav -ac 1 -ar 16000 recording.flac
```

That's about 1 MB per minute of audio, and it costs nothing in accuracy, because the server converts
everything to 16 kHz mono anyway. On an RTX 4090, 45 minutes of audio takes about 10 seconds, or 21
seconds with speaker labels. The [demo Space](demo/app.py) calls the server this way.

## How good is it?

Word error rate (%, lower is better) on 11,822 samples (up to 1,000 per dataset), measured with this
repo's `ta eval` against the `ta serve` HTTP API on an RTX 4090:

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

You can check these numbers yourself and compare against commercial APIs on the same samples:

```bash
poetry run ta eval -m mazesmazes/tiny-audio -d loquacious -n 100
# Same samples through a commercial API (also: deepgram, elevenlabs, apple-speech)
ASSEMBLYAI_API_KEY=... poetry run ta eval -m assemblyai -d loquacious -n 100
```

## How it works

```text
Audio (16 kHz) → speech encoder (frozen) → MLP projector (trained) → LLM decoder → Text
```

1. A pretrained **speech encoder** turns audio into a sequence of frame embeddings.
1. A small **MLP projector** stacks neighbouring frames and maps them into the LLM's embedding
   space. It is the only part trained from scratch.
1. The **LLM** reads those projected frames as if they were tokens and writes out the transcript.

Encoder, projector, and decoder are each swappable from config. Two recipes ship with the repo:

| Recipe                                  | Encoder             | Decoder                | Trained                 |
| --------------------------------------- | ------------------- | ---------------------- | ----------------------- |
| Published model (`granite_qwen_frozen`) | Granite Speech 470M | Qwen3.5, frozen + LoRA | Projector + LoRA (~80M) |
| Default / course recipe (`stage_1`)     | GLM-ASR-Nano (635M) | Qwen3-0.6B, fine-tuned | Projector + decoder     |

## Train your own

Start on your laptop for free, and rent a GPU only once you know the pipeline works.

| Tier                  | Data                                 | Hardware                     | Cost                |
| --------------------- | ------------------------------------ | ---------------------------- | ------------------- |
| **Smoke test**        | 73 LibriSpeech clips                 | Your laptop (CPU, MPS, CUDA) | Free, ~5 minutes    |
| **Course run**        | LoquaciousSet `small` (~250 hours)   | One rented GPU, a few hours  | A few GPU-hours     |
| **Production recipe** | ~3M clips across ten corpora (>1 TB) | One 80 GB GPU, a day or more | Hundreds of dollars |

```bash
git clone https://github.com/alexkroman/tiny-audio.git && cd tiny-audio
poetry install

# 1. Smoke test: a real training loop on your laptop
poetry run python scripts/train.py +experiments=mps_smoke

# 2. Before renting hardware, estimate the VRAM and disk a config needs
poetry run ta runpod plan -e stage_1

# 3. Full run
poetry run python scripts/train.py +experiments=stage_1
```

Every setting is a [Hydra](https://hydra.cc/) override, for example
`model.projector_hidden_dim=2048` or `training.use_lora=true`. When you're happy with a model,
`ta push` publishes it to the Hugging Face Hub and `ta deploy` puts a demo like the one above on a
Space.

### Training on RunPod

```bash
poetry run ta runpod up -e stage_1              # create a pod with enough GPU for the config
poetry run ta runpod wait <POD_ID>              # prints <HOST> <PORT>
poetry run ta runpod deploy <HOST> <PORT>       # sync the project and install dependencies
HF_TOKEN=hf_... poetry run ta runpod train <HOST> <PORT> -e stage_1
poetry run ta runpod attach <HOST> <PORT>       # watch the run in tmux
```

## Learn by building it

The **[free 3.5-hour course](docs/course/0-course-overview.md)** walks you through the full loop:
how the encoder, projector, and decoder fit together (with real tensor shapes), training a model,
evaluating it against commercial APIs, and publishing it with a live demo. You need Python, the
command line, and git.

Want to try a new projector architecture, add a dataset, or change the codebase? See
[CONTRIBUTING.md](CONTRIBUTING.md) for the CLI reference, config layout, and quality gates.

## Acknowledgments

- [Granite Speech](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) and
  [GLM-ASR](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) for audio encoding
- [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B) and
  [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B) for language modeling
- [LoquaciousSet](https://huggingface.co/datasets/speechbrain/LoquaciousSet) for the default
  evaluation set

## License

MIT
