---
license: mit
datasets:
  - mozilla-foundation/common_voice_17_0
  - speechcolab/gigaspeech
  - openslr/librispeech_asr
  - speechbrain/LoquaciousSet
language:
  - en
base_model:
  - facebook/hubert-xlarge-ls960-ft
  - HuggingFaceTB/SmolLM3-3B
pipeline_tag: automatic-speech-recognition
tags:
  - hubert
  - smollm3
  - asr
  - speech-recognition
  - audio
  - parameter-efficient
  - frozen-encoder
  - flash-attention-2
library_name: transformers
model-index:
  - name: tiny-audio
    results:
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: speechbrain/LoquaciousSet
          name: LoquaciousSet
          config: large
          split: test
        metrics:
          - type: wer
            name: Word Error Rate
            value: TBD
---

# Tiny Audio

## Efficient Speech Recognition with Frozen Pretrained Models

Tiny Audio is a lightweight automatic speech recognition (ASR) model that combines a frozen HuBERT-XLarge encoder with a SmolLM3-3B language model decoder, connected via a trainable audio projector. This architecture enables efficient training by only fine-tuning a small projection layer (~7M parameters) while leveraging the power of large pretrained models.

## Model Description

- **Developed by:** Alex Kroman
- **Model type:** Automatic Speech Recognition (Speech-to-Text)
- **Language(s):** English
- **License:** MIT
- **Architecture:** Encoder-Projector-Decoder
  - Audio Encoder: HuBERT-XLarge (1.3B params, frozen)
  - Audio Projector: SwiGLU MLP (~13M params, trainable)
  - Text Decoder: SmolLM3-3B (3B params, frozen)

## Key Features

✅ **Parameter Efficient**: Only ~13M trainable parameters
✅ **Fast Training**: Frozen encoder/decoder enable rapid fine-tuning (~6 hours on A40)
✅ **Modular Design**: Easy to swap different encoder or decoder models
✅ **Production Ready**: Includes evaluation tools and remote training scripts
✅ **HuggingFace Native**: Full integration with transformers library
✅ **Optimized Performance**: Flash Attention 2 for faster inference

## Quick Start

```python
from transformers import pipeline

# Load ASR pipeline
pipe = pipeline("automatic-speech-recognition", model="mazesmazes/tiny-audio", trust_remote_code=True)

# Transcribe audio file
result = pipe("path/to/audio.wav")
print(result["text"])

# With custom generation parameters
result = pipe(
    "path/to/audio.wav",
    max_new_tokens=200,
)
print(result["text"])
```

The model automatically handles:

- Audio resampling to 16kHz
- Various audio formats (WAV, MP3, FLAC, etc.)
- Batch processing for multiple files

## Architecture Details

### Model Components

1. **Audio Encoder (Frozen)**

   - Base Model: `facebook/hubert-xlarge-ls960-ft`
   - Parameters: 1.3B (frozen)
   - Extracts acoustic features from raw audio waveforms
   - Output: Audio embeddings at ~50Hz frame rate

1. **Audio Projector (Trainable)**

   - Architecture: SwiGLU MLP (following Llama design)
     - `gate_proj`: Linear(6400 → 2048, no bias)
     - `up_proj`: Linear(6400 → 2048, no bias)
     - `down_proj`: Linear(2048 → 2048, no bias)
     - Activation: `silu(gate) * up` → `down`
   - Parameters: ~13M (trainable)
   - Downsamples audio features by 5x (from ~50Hz to ~10Hz)
   - Maps audio embeddings to language model embedding space

1. **Language Model Decoder (Frozen)**

   - Base Model: `HuggingFaceTB/SmolLM3-3B`
   - Parameters: 3B (frozen)
   - Generates text transcriptions autoregressively with greedy decoding
   - Uses Flash Attention 2 for efficient processing

### Data Flow

```text
Raw Audio (16kHz)
    ↓
HuBERT-XLarge Encoder (frozen)
    ↓
Audio Features [batch, ~1500, 1280]
    ↓
Audio Projector (SwiGLU MLP, trainable, 5x downsample)
    gate_proj(6400→2048) & up_proj(6400→2048)
    silu(gate) * up → down_proj(2048→2048)
    ↓
Language Embeddings [batch, ~300, 2048]
    ↓
SmolLM3-3B Decoder (frozen, Flash Attention 2)
    ↓
Text Transcription
```

## Training Details

### Training Data

The model is trained on the LoquaciousSet dataset:

| Dataset | Hours | Description |
|---------|-------|-------------|
| LoquaciousSet | 25,000h | A diverse curated corpus combining CommonVoice, VoxPopuli, Libriheavy, People's Speech and YODAS. Contains hundreds of thousands of speakers with varied accents, speech types (read, spontaneous, talks), and acoustic conditions (clean to noisy with reverberation). |

This diverse training data enables the model to handle a wide range of English speech recognition scenarios.

### Training Configuration

- **Optimizer:** AdamW
- **Learning Rate:** Cosine schedule with warmup
- **Precision:** BF16 mixed precision
- **Batch Size:** Dynamic based on audio length
- **Gradient Checkpointing:** Enabled
- **Training Steps:** ~50,000 steps
- **Hardware:** Single NVIDIA A100 (40GB)
- **Training Time:** ~24 hours

### Training Strategy

Only the audio projector weights (~13M params) are trained from scratch. The HuBERT-XLarge encoder and SmolLM3-3B decoder remain frozen throughout training, which:

- Reduces memory requirements significantly
- Enables faster training convergence (~6 hours vs days/weeks)
- Preserves pretrained knowledge from both audio and language domains
- Prevents catastrophic forgetting
- Makes training affordable (~$6 on A40)

## Evaluation

The model is evaluated on the LoquaciousSet benchmark dataset using Word Error Rate (WER) as the primary metric.

### Benchmark Results

| Dataset | Split | Samples | WER |
|---------|-------|---------|-----|
| LoquaciousSet (large) | test | ~2,000 | TBD |
| LoquaciousSet (clean) | test | ~500 | TBD |

### Evaluation Script

```bash
# Run evaluation
poetry run eval mazesmazes/tiny-audio --max-samples 100

# Compare with baselines
poetry run eval --provider assemblyai --api-key YOUR_API_KEY
```

## Limitations and Bias

### Limitations

- **English Only**: Currently trained only on English speech data
- **Formal Speech**: May perform better on clear, formal speech than casual conversation
- **Background Noise**: Performance may degrade in noisy environments
- **Accents**: May have varying performance across different English accents
- **Domain Shift**: Best performance on domains similar to training data

### Potential Biases

- **Dataset Bias**: Training data may not equally represent all demographics
- **Accent Bias**: May perform differently across accents (American, British, Indian, etc.)
- **Gender Bias**: Performance may vary by speaker gender
- **Age Bias**: Primarily trained on adult speech

Users should evaluate the model on their specific use case and demographics before production deployment.

## Intended Use

### Primary Use Cases

✅ **Transcription Services**: Converting speech to text for podcasts, videos, interviews
✅ **Accessibility Tools**: Generating captions and subtitles
✅ **Voice Assistants**: Speech-to-text component in voice interfaces
✅ **Research**: ASR research and experimentation
✅ **Education**: Learning about multimodal models and parameter-efficient training

### Out-of-Scope Use

❌ Real-time critical systems (medical, legal) without thorough validation
❌ Surveillance or privacy-invasive applications
❌ Non-English languages (not trained for this)
❌ Child safety applications without age-appropriate testing

## Environmental Impact

- **Hardware:** 1× NVIDIA A100 (40GB)
- **Training Time:** ~24 hours
- **Power Consumption:** ~300W × 24h = 7.2 kWh
- **Estimated CO₂ Emissions:** ~3.6 kg CO₂e (assuming 0.5 kg CO₂/kWh)

This is significantly lower than training full ASR models from scratch thanks to frozen pretrained components.

## Citation

If you use Tiny Audio in your research, please cite:

```bibtex
@software{kroman2024tinyaudio,
  author = {Kroman, Alex},
  title = {Tiny Audio: Efficient Speech Recognition with Frozen Pretrained Models},
  year = {2024},
  url = {https://github.com/alexkroman/tiny-audio},
  note = {HuggingFace Model: https://huggingface.co/mazesmazes/tiny-audio}
}
```

## Acknowledgments

This project builds upon excellent prior work:

- **HuBERT** ([Hsu et al., 2021](https://huggingface.co/docs/transformers/model_doc/hubert)): Self-supervised speech representation learning
- **SmolLM3-3B** ([HuggingFace](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)): Efficient small language model

## Additional Resources

- 📄 [GitHub Repository](https://github.com/alexkroman/tiny-audio)
- 🎯 [Live Demo](https://huggingface.co/spaces/mazesmazes/tiny-audio)
- 📚 [Documentation](https://github.com/alexkroman/tiny-audio#readme)
- 🐛 [Issue Tracker](https://github.com/alexkroman/tiny-audio/issues)

## License

This model is released under the MIT License. See the [LICENSE](https://github.com/alexkroman/tiny-audio/blob/main/LICENSE) file for details.
