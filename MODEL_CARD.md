______________________________________________________________________

license: mit datasets:

- mozilla-foundation/common_voice_17_0
- speechcolab/gigaspeech
- openslr/librispeech_asr language:
- en metrics:
- wer base_model:
- openai/whisper-small
- HuggingFaceTB/SmolLM2-360M-Instruct pipeline_tag: automatic-speech-recognition

______________________________________________________________________

# Tiny Audio - Whisper-SmolLM2 ASR Model

This is a lightweight automatic speech recognition (ASR) model that combines
OpenAI's Whisper encoder with Hugging Face's SmolLM2 decoder, trained using
parameter-efficient LoRA fine-tuning. The model demonstrates how modern ASR
systems can be built efficiently with minimal code and computational resources.

## Model Details

### Model Description

Tiny Audio is an educational ASR model designed to demonstrate the fundamentals
of modern speech recognition. It uses a frozen Whisper encoder to extract audio
features and a SmolLM2 language model with LoRA adapters to generate
transcriptions. The entire implementation is just ~300 lines of Python code,
making it accessible for learning and experimentation.

- **Developed by:** Alex Kroman
- **Model type:** Encoder-Decoder ASR with LoRA fine-tuning
- **Language(s) (NLP):** English
- **License:** MIT
- **Finetuned from model:**
  - Encoder: `openai/whisper-small` (39M parameters, frozen)
  - Decoder: `HuggingFaceTB/SmolLM2-360M-Instruct` (360M parameters, ~2% trained
    via LoRA)

### Model Sources

- **Repository:**
  [github.com/alexkroman/tiny-audio](https://github.com/alexkroman/tiny-audio)
- **Demo:** Available in the repository via `demo/gradio_app.py`

## Uses

### Direct Use

The model is designed for automatic speech recognition of English audio. It's
particularly suited for:

- Educational purposes to understand modern ASR architectures
- Research into parameter-efficient fine-tuning for speech tasks
- Lightweight ASR applications where model size is a constraint
- Prototyping ASR systems with minimal computational requirements

### Downstream Use

The model can be:

- Fine-tuned further on domain-specific audio datasets
- Integrated into larger speech processing pipelines
- Used as a baseline for ASR research
- Adapted for other languages by continuing training on multilingual datasets

### Out-of-Scope Use

The model is not suitable for:

- Production use cases requiring high accuracy
- Non-English languages (without additional fine-tuning)
- Real-time streaming ASR (model processes full audio files)
- Audio with significant background noise or multiple speakers
- Medical or legal transcription requiring certified accuracy

## Bias, Risks, and Limitations

- **Training Data Bias:** The model inherits biases from LibriSpeech,
  GigaSpeech, and Common Voice datasets, which may not represent all English
  accents and dialects equally
- **Accuracy Limitations:** As an educational model, accuracy is lower than
  state-of-the-art commercial ASR systems
- **Language Limitation:** Currently only supports English
- **Context Length:** Limited by the SmolLM2 context window
- **Audio Quality:** Performance degrades with noisy or low-quality audio

### Recommendations

Users should:

- Evaluate the model on their specific use case before deployment
- Consider fine-tuning on domain-specific data for better performance
- Be aware of potential biases in transcriptions
- Not rely on this model for critical applications requiring high accuracy

## How to Get Started with the Model

### Using the Pre-trained Model

```python
from transformers import AutoModel

# Load the model
model = AutoModel.from_pretrained("mazesmazes/tiny-audio", trust_remote_code=True)

# Transcribe audio
transcription = model.transcribe("path/to/audio.wav")
print(transcription)
```

### Training Your Own Version

```bash
# Clone the repository
git clone https://github.com/alexkroman/tiny-audio.git
cd tiny-audio

# Install dependencies
uv sync  # or: pip install -e .

# Set up HuggingFace token for datasets
export HF_TOKEN='your-hugging-face-token'

# Quick test training (20 steps, ~2 minutes)
uv run src/train.py

# Full production training
uv run src/train.py +experiments=production
```

## Training Details

### Training Data

The model was trained on a combination of open-source English ASR datasets:

- **LibriSpeech ASR**: Clean read English speech from audiobooks
- **GigaSpeech**: Large-scale English speech recognition corpus from audiobooks,
  podcasts, and YouTube
- **Common Voice 17.0**: Crowd-sourced voice dataset with diverse speakers

All datasets were accessed via Hugging Face's streaming API to handle the large
data volumes efficiently.

### Training Procedure

#### Architecture Components

1. **Whisper Encoder**: Frozen `whisper-small` encoder (39M params) extracts
   log-mel spectrograms
1. **Audio Projector**: Custom projection layer with RMSNorm and GELU activation
   maps audio to text space
1. **SmolLM2 Decoder**: Language model with LoRA adapters (rank 32-64) generates
   transcriptions

#### Training Hyperparameters

- **Training regime:** BF16 mixed precision
- **LoRA rank:** 32 (small config) or 64 (large config)
- **LoRA alpha:** 16
- **LoRA dropout:** 0.1
- **Learning rate:** 1e-4 with cosine scheduler
- **Batch size:** 4-8 (depending on hardware)
- **Gradient accumulation:** 4 steps
- **Warmup steps:** 500
- **Max steps:** 5000-10000
- **Weight decay:** 0.01
- **Optimizer:** AdamW (β₁=0.9, β₂=0.999, ε=1e-8)

#### Key Design Choices

- **Frozen Encoder:** Preserves pre-trained audio representations
- **LoRA Fine-tuning:** Only ~7.3M parameters trained (2% of total)
- **Audio Projection:** Scaled initialization (0.01x) prevents gradient
  explosion
- **Streaming Datasets:** Handles TB-scale data without local storage
- **Cross-Entropy Loss:** Applied only to text tokens (audio tokens masked with
  -100)

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

- LibriSpeech test-clean subset
- Common Voice test split
- Custom evaluation samples from the demo application

#### Metrics

- **Word Error Rate (WER):** Primary metric for ASR performance
- **Character Error Rate (CER):** Secondary metric for fine-grained accuracy
- **Perplexity:** Language modeling quality of the decoder

### Results

Performance varies based on training configuration and dataset:

- **LibriSpeech test-clean WER:** ~15-20% (with production config)
- **Training time:** 2 minutes (minimal) to 24 hours (production) depending on
  configuration
- **Model size:** ~400M total parameters, ~7.3M trainable via LoRA

## Environmental Impact

Carbon emissions were minimized through:

- Parameter-efficient training (only 2% of parameters updated)
- Efficient data streaming (no redundant data downloads)
- Support for local training on consumer hardware

Estimated emissions for full training:

- **Hardware Type:** NVIDIA A100 40GB or Apple M2 Max
- **Hours used:** 24 hours (production config)
- **Cloud Provider:** RunPod or local
- **Compute Region:** Variable
- **Carbon Emitted:** ~5-10 kg CO2eq (estimated)

## Technical Specifications

### Model Architecture and Objective

- **Architecture:** Encoder-decoder with cross-attention
- **Encoder:** Whisper-small (39M params, frozen)
- **Projection:** 2-layer MLP with RMSNorm and GELU (1500→2048 dim)
- **Decoder:** SmolLM2-360M with LoRA adapters
- **Objective:** Cross-entropy loss on text tokens
- **Total Parameters:** ~400M
- **Trainable Parameters:** ~7.3M (via LoRA)

### Compute Infrastructure

#### Hardware

- Development: Apple M1/M2 MacBook Pro
- Production Training: NVIDIA A100 40GB GPUs on RunPod
- Inference: CPU or GPU, ~2GB memory required

#### Software

- PyTorch 2.0+
- Transformers 4.40+
- Accelerate for mixed precision training
- Hydra for configuration management
- TensorBoard for monitoring

## Citation

If you use this model in your research, please cite:

**BibTeX:**

```bibtex
@software{tiny_audio_2025,
  author = {Kroman, Alex},
  title = {Tiny Audio: Learn ASR by Building One},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/alexkroman/tiny-audio}
}
```

## Model Card Contact

For questions or issues, please open an issue on the
[GitHub repository](https://github.com/alexkroman/tiny-audio/issues).
