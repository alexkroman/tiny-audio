---
license: mit
language:
  - en
datasets:
  - mythicinfinity/libriheavy
  - MLCommons/peoples_speech
  - fixie-ai/common_voice_17_0
  - speechcolab/gigaspeech
  - kensho/spgispeech
  - facebook/voxpopuli
  - edinburghcstr/ami
  - sanchit-gandhi/tedlium-data
base_model:
  - ibm-granite/granite-speech-5.0-470m-turboctc
  - Qwen/Qwen3.5-2B
pipeline_tag: automatic-speech-recognition
tags:
  - asr
  - speech-recognition
  - audio
  - qwen
  - granite-speech
  - lora
library_name: transformers
---

# Tiny Audio

**A tiny, hackable, open-source speech LLM.**

It transcribes English with punctuation, capitalization, word timestamps, and speaker labels: 1.8%
WER on LibriSpeech test-clean and 8.4% averaged over 12 benchmarks. Built and trained with
[Tiny Audio](https://github.com/alexkroman/tiny-audio).

**[Try the live demo](https://huggingface.co/spaces/mazesmazes/tiny-audio)** ·
**[Train your own](https://github.com/alexkroman/tiny-audio)** ·
**[Free 3.5-hour course](https://github.com/alexkroman/tiny-audio/blob/main/docs/course/0-course-overview.md)**

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

Word error rate (%, lower is better) on 1,000 samples per dataset, scored with the
[Tiny Audio eval harness](https://github.com/alexkroman/tiny-audio/tree/main/scripts/eval)
(`ta eval`) after text normalization.

| Dataset                |      WER |
| ---------------------- | -------: |
| LibriSpeech test-clean |     1.80 |
| SPGISpeech             |     2.24 |
| LibriSpeech test-other |     3.09 |
| TED-LIUM               |     3.74 |
| LoquaciousSet †        |     6.10 |
| Common Voice           |     6.62 |
| VoxPopuli              |     6.96 |
| AMI (IHM)              |     8.88 |
| GigaSpeech             |     9.07 |
| Earnings22 †           |    10.54 |
| People's Speech        |    17.69 |
| AMI (SDM)              |    23.59 |
| **Mean (12 sets)**     | **8.36** |

† Held out: no data from this source was used in training.

To reproduce a row, or score AssemblyAI, Deepgram, ElevenLabs, or Apple's on-device recognizer on
the same samples:

```bash
git clone https://github.com/alexkroman/tiny-audio.git && cd tiny-audio && poetry install
poetry run ta eval -m mazesmazes/tiny-audio -d loquacious -n 100
ASSEMBLYAI_API_KEY=... poetry run ta eval -m assemblyai -d loquacious -n 100
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
under a `words` key. Audio of any length works: the pipeline transcribes it in 8-18 s chunks cut at
pauses (the model trained on clips up to 19 s), aligns each chunk against its own transcript in
batches, and places every word on the recording's timeline.

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
handles overlapping speech and up to 8 speakers.

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
you know it, pass `num_speakers` (exact) or `max_speakers` (an upper bound); any extra voices' words
go to the closest kept speaker:

```python
result = pipe("call.wav", return_speakers=True, num_speakers=2)
```

If alignment or diarization fails (for example, on a transformers release without diarization
support), the transcript is still returned and the error is reported under
`result["timestamp_error"]` or `result["diarization_error"]`.

## Train Your Own

Everything that produced this model is open: the code, the data mix, and the recipe. Swap the
encoder, the LLM, or the projector from config and train on your own data. A smoke test runs on a
laptop in about five minutes:

```bash
git clone https://github.com/alexkroman/tiny-audio.git && cd tiny-audio && poetry install
poetry run python scripts/train.py +experiments=mps_smoke
```

The
[free course](https://github.com/alexkroman/tiny-audio/blob/main/docs/course/0-course-overview.md)
walks through how the pieces fit, training a model, evaluating it, and publishing it with a demo
like [this one](https://huggingface.co/spaces/mazesmazes/tiny-audio).

## Model Specifications

| Specification    | Value                                                              |
| ---------------- | ------------------------------------------------------------------ |
| Input            | Audio (16kHz mono)                                                 |
| Output           | Punctuated, capitalized text with formatted numbers                |
| Max Audio Length | ~19 s per plain call; any length with timestamps or speakers       |
| Vocabulary       | Qwen3.5 tokenizer                                                  |
| Languages        | English only                                                       |
| Generation       | Greedy decoding (num_beams=1, do_sample=False), max 256 new tokens |

## Training Details

|                   |                                                                                                                                                                 |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data**          | LibriHeavy (medium), People's Speech (clean), Common Voice 17, GigaSpeech (M), SPGISpeech (M), VoxPopuli, AMI (IHM + SDM), TED-LIUM: one pass, ~3.8M utterances |
| **Hardware**      | Single NVIDIA H100 80GB                                                                                                                                         |
| **Steps**         | 58,861                                                                                                                                                          |
| **Batch Size**    | 64                                                                                                                                                              |
| **Optimizer**     | AdamW (fused), cosine schedule, 1,000 warmup steps, no weight decay                                                                                             |
| **Learning Rate** | 1e-3 (projector), 1e-4 (LoRA)                                                                                                                                   |
| **Precision**     | bf16 (projector held in fp32)                                                                                                                                   |

The training recipe is
[`configs/experiments/granite_qwen_frozen.yaml`](https://github.com/alexkroman/tiny-audio/blob/main/configs/experiments/granite_qwen_frozen.yaml).

## Limitations

- **English only**: Not trained on other languages.
- **Sample rate**: Expects 16kHz audio (other rates are resampled automatically).
- **Audio length**: A plain `pipe(audio)` call decodes the clip in one pass and works best up to
  about 19 seconds (the training length). For longer audio pass `return_timestamps=True` or
  `return_speakers=True`, which transcribe in 8-18 s chunks automatically.
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
| `model.safetensors`                                                | Projector weights (~50MB)                                |
| `adapter_config.json` / `adapter_model.safetensors`                | LoRA adapters for the decoder (~270MB)                   |
| `preprocessor_config.json`                                         | Audio preprocessing config                               |
| `tokenizer.json` / `tokenizer_config.json` / `chat_template.jinja` | Tokenizer                                                |
| `asr_*.py`, `projectors.py`, `alignment.py`, `diarization.py`      | Custom model code (loaded with `trust_remote_code=True`) |

Only the projector and LoRA weights are stored here. The encoder (Granite Speech) and decoder
(Qwen3.5-2B) are downloaded from their own Hugging Face repos.

The previous GLM-ASR + Qwen3-0.6B model is still available at revision `glm-asr-qwen3-0.6b`:

```python
pipe = pipeline(
    "automatic-speech-recognition",
    model="mazesmazes/tiny-audio",
    revision="glm-asr-qwen3-0.6b",
    trust_remote_code=True,
)
```

## Citation

If you use this model, please cite:

```bibtex
@misc{tinyaudio2024,
  author = {Alex Kroman},
  title = {Tiny Audio: Minimal ASR Training},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/alexkroman/tiny-audio}
}
```

## Acknowledgments

- [Granite Speech](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) for the
  audio encoder
- [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B) for the language model
- [Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf) for word timestamps
- [Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) for speaker
  diarization
- The LibriHeavy, People's Speech, Common Voice, GigaSpeech, SPGISpeech, VoxPopuli, AMI, and
  TED-LIUM teams for training data

## License

MIT
