---
license: cc-by-nc-sa-4.0
language:
  - en
datasets:
  - mythicinfinity/libriheavy
  - MLCommons/peoples_speech
  - fixie-ai/common_voice_17_0
  - speechcolab/gigaspeech
  - kensho/spgispeech
  - edinburghcstr/ami
  - sanchit-gandhi/tedlium-data
  - edinburghcstr/edacc
  - mazesmazes/notsofar-train
base_model:
  - ibm-granite/granite-speech-5.0-470m-turboctc-nc
  - Qwen/Qwen3-0.6B
pipeline_tag: automatic-speech-recognition
tags:
  - asr
  - speech-recognition
  - speech-to-text
  - audio
  - speech-llm
  - word-timestamps
  - speaker-diarization
  - qwen
  - granite-speech
library_name: transformers
metrics:
  - wer
---

# Tiny Audio

**A tiny, hackable, open-source speech LLM.**

It transcribes English with punctuation, capitalization, word timestamps, and speaker labels. Built
and trained with [Tiny Audio](https://github.com/alexkroman/tiny-audio).

**[Try the live demo](https://huggingface.co/spaces/mazesmazes/tiny-audio)** ·
**[Train your own](https://github.com/alexkroman/tiny-audio)** ·
**[Free 3.5-hour course](https://github.com/alexkroman/tiny-audio/blob/main/docs/course/0-course-overview.md)**

## Model Description

Tiny Audio connects a frozen, pretrained speech encoder to a pretrained LLM with a small trainable
projector:

```text
Audio (16 kHz) → speech encoder (frozen) → MLP projector (trained) → LLM decoder → Text
```

The published model pairs the non-causal Granite Speech 5.0 470M encoder (kept frozen) with the
Qwen3-0.6B decoder. The MLP projector maps encoder frames into the decoder's embedding space; it is
the only part trained from scratch, and the whole decoder is fine-tuned alongside it: ~604M trained
parameters (596M decoder, 8.4M projector). The encoder, projector, and decoder are each swappable
from config in the [training repo](https://github.com/alexkroman/tiny-audio).

## Intended Use

Tiny Audio is for transcribing English speech — meetings, calls, lectures, and other recorded audio
— when you want the finished transcript itself: punctuation, capitalization, and formatted numbers
included, plus word-level timestamps and speaker labels when you need them. It is also built to be
read and retrained: the training code, data mix, and recipe are open, so you can adapt it to your
own audio.

Out of scope: non-English transcription, and audio far from the training distribution — far-field,
noisy, or heavily overlapping speech — where word error rate rises (see Benchmarks and Limitations).

## Quick Start

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

The input can be a file path, a URL, or a 16 kHz float32 numpy array. No post-processing is needed:
the model writes punctuation, capitalization, and numbers itself.

## Benchmarks

Benchmark results for this release have not been published yet. To score it yourself with the
[Tiny Audio eval harness](https://github.com/alexkroman/tiny-audio/tree/main/scripts/eval)
(`ta eval`):

```bash
git clone https://github.com/alexkroman/tiny-audio.git && cd tiny-audio && poetry install
poetry run ta eval -m mazesmazes/tiny-audio -d loquacious -n 100
```

## More Than Plain Text

### Batches and GPU

```python
import torch

pipe = pipeline(
    "automatic-speech-recognition",
    model="mazesmazes/tiny-audio",
    trust_remote_code=True,
    device="cuda",
    torch_dtype=torch.bfloat16,
)
for r in pipe(["audio1.wav", "audio2.wav", "audio3.wav"], batch_size=4):
    print(r["text"])
```

### Word-level timestamps

`return_timestamps=True` times every word with
[Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf) and returns them
under a `words` key. Audio of any length works: the pipeline transcribes it in chunks of up to 28 s
cut at pauses (the model trained on clips up to 30 s), aligns each chunk against its own transcript
in batches, and places every word on the recording's timeline.

```python
result = pipe("audio.wav", return_timestamps=True)
print(result["text"])
for word in result["words"]:
    print(word)
#   {'word': 'Hi,', 'start': 0.0, 'end': 0.24}
```

### Speaker diarization

Pass `return_speakers=True` to label every word with who said it. It also turns on word timestamps.
It works on recordings of any length, keeps speaker labels consistent across hour-long meetings, and
handles up to 8 speakers. Each speaker is transcribed separately on audio where everyone else is
silenced, so overlapping speech is only partly handled.

Diarization needs transformers installed from `main` until the next release:

```bash
pip install git+https://github.com/huggingface/transformers
```

```python
result = pipe("meeting.wav", return_speakers=True)

print(result["text"])

# Speaker turns (may overlap)
for seg in result["speaker_segments"]:
    print(f"{seg['start']:6.2f}-{seg['end']:6.2f}  {seg['speaker']}")
#   0.00-  2.90  SPEAKER_0
#   3.36-  6.47  SPEAKER_1

# Words with timestamps and speakers
for w in result["words"]:
    print(f"{w['start']:6.2f}-{w['end']:6.2f}  {w['speaker']}  {w['word']}")
#   0.00-  0.19  SPEAKER_0  Hi,
#   0.22-  0.69  SPEAKER_0  Daniel.
```

Speakers are numbered by when they first speak. The number of speakers is detected automatically. If
you know it, pass `num_speakers` (exact) or `max_speakers` (an upper bound); voices beyond the cap
are not transcribed:

```python
result = pipe("call.wav", return_speakers=True, num_speakers=2)
```

If alignment or diarization fails (for example, on a transformers release without diarization
support), the transcript is still returned and the error is reported under
`result["timestamp_error"]` or `result["diarization_error"]`.

## Train Your Own

Everything that produced this model is open: the code, the data mix, and the recipe. Swap the
encoder, the LLM, or the projector from config and train on your own data. The recipe is the repo's
default config:

```bash
git clone https://github.com/alexkroman/tiny-audio.git && cd tiny-audio && poetry install
poetry run python scripts/train.py
```

The
[free course](https://github.com/alexkroman/tiny-audio/blob/main/docs/course/0-course-overview.md)
walks through how the pieces fit, training a model, evaluating it, and publishing it with a demo
like [this one](https://huggingface.co/spaces/mazesmazes/tiny-audio).

## Serving and Streaming

For more than one-off calls, the [training repo](https://github.com/alexkroman/tiny-audio) ships a
batched HTTP server (`ta serve`) with an async jobs API for long recordings, and the model class
exposes token-by-token streaming output via `ASRModel.generate_streaming`. See the
[repo README](https://github.com/alexkroman/tiny-audio#as-an-http-api-on-runpod) for setup and the
request format.

## Model Specifications

| Specification    | Value                                                              |
| ---------------- | ------------------------------------------------------------------ |
| Input            | Audio (16kHz mono)                                                 |
| Output           | Punctuated, capitalized text with formatted numbers                |
| Max Audio Length | Any length; audio over 28 s is transcribed in chunks cut at pauses |
| Vocabulary       | Qwen3 tokenizer                                                    |
| Languages        | English only                                                       |
| Generation       | Greedy decoding (num_beams=1, do_sample=False), max 256 new tokens |
| Optional output  | Word-level timestamps; speaker-labelled words and speaker segments |
| Weights          | ~2.1 GB in bf16 (encoder + projector + decoder)                    |

## Training Details

|                   |                                                                                                                                                     |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data**          | LibriHeavy (medium), People's Speech (clean), Common Voice 17, GigaSpeech (M), SPGISpeech (M), AMI (IHM + SDM), TED-LIUM, NOTSOFAR, EdAcc: one pass |
| **Steps**         | 16,159                                                                                                                                              |
| **Batch Size**    | 304                                                                                                                                                 |
| **Optimizer**     | AdamW (fused), cosine schedule, 1,000 warmup steps, no weight decay                                                                                 |
| **Learning Rate** | 1e-3 (projector), 2e-5 (decoder)                                                                                                                    |
| **Precision**     | bf16 mixed precision (weights held in fp32)                                                                                                         |

The training recipe is the default config,
[`configs/config.yaml`](https://github.com/alexkroman/tiny-audio/blob/main/configs/config.yaml) with
[`configs/training/production.yaml`](https://github.com/alexkroman/tiny-audio/blob/main/configs/training/production.yaml).

## Limitations

- **English only**: Not trained on other languages.
- **Sample rate**: Expects 16kHz audio (other rates are resampled automatically).
- **Audio length**: Audio longer than 28 seconds is cut into chunks at pauses and transcribed chunk
  by chunk, so context does not carry across chunk boundaries.
- **Speaker diarization**: At most 8 speakers per recording. The count is detected automatically;
  `num_speakers` and `max_speakers` can cap it, but `min_speakers` is not supported (passing it
  raises a `ValueError`). Needs transformers `main` until the next release.
- **Accuracy**: May degrade on:
  - Far-field and overlapping speech (see AMI SDM)
  - Noisy or low-quality audio
  - Rare names and domain-specific terminology

## Files

| File                                                               | Description                                              |
| ------------------------------------------------------------------ | -------------------------------------------------------- |
| `config.json`                                                      | Model configuration                                      |
| `model.safetensors`                                                | Projector and fine-tuned decoder weights (~2.4 GB, fp32) |
| `preprocessor_config.json`                                         | Audio preprocessing config                               |
| `tokenizer.json` / `tokenizer_config.json` / `chat_template.jinja` | Tokenizer                                                |
| `asr_*.py`, `projectors.py`, `alignment.py`, `diarization.py`      | Custom model code (loaded with `trust_remote_code=True`) |

The encoder (Granite Speech) is downloaded from its own Hugging Face repo. `config.json` loads the
weights in bf16, as the eval harness runs them.

Earlier models are still available by revision: `granite-qwen-frozen-4` (Granite Speech 470M +
Qwen3.5-2B with LoRA, MIT) and `glm-asr-qwen3-0.6b` (GLM-ASR + Qwen3-0.6B):

```python
pipe = pipeline(
    "automatic-speech-recognition",
    model="mazesmazes/tiny-audio",
    revision="granite-qwen-frozen-4",
    trust_remote_code=True,
)
```

## Citation

If you use this model, please cite:

```bibtex
@misc{tinyaudio2025,
  author = {Alex Kroman},
  title = {Tiny Audio: Minimal ASR Training},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/alexkroman/tiny-audio}
}
```

## Acknowledgments

- [Granite Speech](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc-nc) for the
  audio encoder
- [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B) for the language model
- [Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf) for word timestamps
- [Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) for speaker
  diarization
- The LibriHeavy, People's Speech, Common Voice, GigaSpeech, SPGISpeech, AMI, TED-LIUM, NOTSOFAR,
  and EdAcc teams for training data

## License

CC BY-NC-SA 4.0 (non-commercial, share-alike), inherited from the
[Granite Speech 5.0 470M non-causal encoder](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc-nc)
this model is built on. For commercial use, see the MIT-licensed `granite-qwen-frozen-4` revision
above. The Tiny Audio training code is MIT.
