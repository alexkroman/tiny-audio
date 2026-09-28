# Inference: `tiny-audio-granite-qwen-frozen-3`

Minimal script for transcribing audio with
[`mazesmazes/tiny-audio-granite-qwen-frozen-3`](https://huggingface.co/mazesmazes/tiny-audio-granite-qwen-frozen-3).

**Model:** audio → Granite Speech 5.0 470M encoder (frozen) → MLP projector →
Qwen3.5-4B + LoRA (rank 64) → text. The weights are bf16, so you need about 10 GB
of GPU/unified memory.

## Install

```bash
pip install "transformers>=5.0" peft torch torchaudio librosa
```

## Script

```python
# transcribe.py -- usage: python transcribe.py audio.wav
import sys

import torch
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

pipe = pipeline(
    "automatic-speech-recognition",
    model="mazesmazes/tiny-audio-granite-qwen-frozen-3",
    trust_remote_code=True,  # the model, processor, and pipeline code live in the model repo
    device=device,
)

print(pipe(sys.argv[1])["text"])
```

The input can be a file path, a URL, or `{"raw": np_array, "sampling_rate": 16000}`.

## Notes

- **You don't need to set any decoding options.** The pipeline reads everything from the
  checkpoint's `config.json`: greedy decoding, `max_new_tokens=256`,
  `no_repeat_ngram_size=12`, the transcription prompt, and the 0.25 s of lead-in
  silence added before inference.
- **Word timestamps:** `pipe(path, return_timestamps=True)["words"]` (uses forced alignment).
- **Apple Silicon:** transcribe one file per call. When the MPS `sdpa` kernel gets a
  left-padded batch, it returns NaN and the output decodes as `!!!!`.
- **Speed (CUDA):** Qwen3.5 uses linear-attention layers. Unless `flash-linear-attention` and
  `causal-conv1d` are installed, transformers warns and falls back to slower reference
  PyTorch kernels. The output is correct either way.
- **Long audio:** a single call produces at most 256 tokens. For long recordings, split the
  audio into chunks of about 30 s first.
