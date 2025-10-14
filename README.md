# Tiny Audio

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-mazesmazes%2Ftiny--audio-yellow)](https://huggingface.co/mazesmazes/tiny-audio)

A lightweight automatic speech recognition (ASR) model that combines a frozen Whisper encoder with a SmolLM3 decoder, connected via a learnable audio projector. Only the projector is trained, making it extremely efficient to fine-tune.

**Key Features:**

- **Efficient Training**: Only ~7M trainable parameters (projector) while leveraging frozen pretrained models
- **Modular Design**: Easy to swap different audio encoders or language decoders
- **Production Ready**: Includes remote training, evaluation tools, and HuggingFace integration
- **Streaming Support**: Handles large datasets with streaming data loaders

**Quick Links:**

- [Live Demo](https://huggingface.co/spaces/mazesmazes/tiny-audio)
- [Model on HuggingFace](https://huggingface.co/mazesmazes/tiny-audio)
- [GitHub Repository](https://github.com/alexkroman/tiny-audio)

## Quick Start

### Installation

```bash
pip install tiny-audio
```

Or from source:

```bash
git clone https://github.com/alexkroman/tiny-audio.git
cd tiny-audio
uv sync  # or: pip install -e .
```

### Usage

```python
from transformers import pipeline

# Load the ASR pipeline
pipe = pipeline("automatic-speech-recognition", model="mazesmazes/tiny-audio", trust_remote_code=True)

# Transcribe audio
result = pipe("path/to/audio.wav")
print(result["text"])

# Or with custom generation parameters
result = pipe(
    "path/to/audio.wav",
    max_new_tokens=200,
    num_beams=4,
)
print(result["text"])
```

The model accepts various audio formats (WAV, MP3, FLAC, etc.) and automatically handles resampling to 16kHz.

## Training

### Training Setup

```bash
git clone https://github.com/alexkroman/tiny-audio.git
cd tiny-audio
uv sync  # Install dependencies using uv (or use: pip install -e .)

# Quick test run (20 steps, small dataset)
uv run src/train.py

# Full production training
export HF_TOKEN='your-token'  # Get from https://huggingface.co/settings/tokens
uv run src/train.py +experiments=production
```

### Training Details

**Datasets** (loaded via streaming):

- LibriSpeech (960h)
- GigaSpeech (10,000h)
- Common Voice 17.0
- LoquaciousSet

**Training Configuration:**

- Mixed precision: BF16
- Optimizer: AdamW with cosine learning rate schedule
- Gradient checkpointing: Enabled for memory efficiency
- Batch size: Configurable via Hydra configs
- Only the audio projector weights are trained

**Hardware Requirements:**

- Minimum: 16GB GPU (e.g., RTX 4080)
- Recommended: 24GB+ GPU (e.g., RTX 4090, A100)
- Training time: ~24 hours on single A100 for full training

## Architecture

Tiny Audio uses a modular three-component architecture:

### Components

1. **Audio Encoder** (Frozen)
   - Default: Whisper-small (`openai/whisper-small`)
   - Alternative: HuBERT-large (`facebook/hubert-large-ls960-ft`)
   - Extracts acoustic features from raw audio waveforms
   - Pretrained weights remain frozen during training

2. **Audio Projector** (Trainable ~7M params)
   - Downsamples audio features by 5x (configurable)
   - Architecture: `Linear(encoder_dim × k, 2048) → ReLU → Linear(2048, llm_dim)`
   - Maps audio features to language model embedding space
   - **This is the only trained component**

3. **Language Model Decoder** (Frozen)
   - Default: SmolLM3-3B-Base (`HuggingFaceTB/SmolLM3-3B-Base`)
   - Generates text transcriptions autoregressively
   - Pretrained weights remain frozen during training
   - Uses BF16 precision for efficient inference

### Data Flow

```text
Audio Waveform → Encoder → Audio Features → Projector → LLM Embeddings → Decoder → Text
                 (frozen)                  (trainable)                  (frozen)
```

### Why This Architecture?

- **Parameter Efficient**: Only train the small projector (~7M params) instead of the full model (>400M params)
- **Leverages Pretrained Models**: Combines the best of audio understanding (Whisper) and language generation (SmolLM3)
- **Flexible**: Easy to swap encoders (Whisper, HuBERT) or decoders (SmolLM, Llama, etc.)
- **Fast Training**: Frozen components mean faster training and lower memory requirements

## Configuration

Tiny Audio uses [Hydra](https://hydra.cc/) for configuration management, providing flexible config composition and command-line overrides.

### Config Structure

```text
configs/
├── hydra/
│   ├── config.yaml           # Main config
│   ├── model/                # Model variants
│   │   ├── default.yaml      # Whisper-small + SmolLM3-3B
│   │   └── large.yaml        # Larger model configs
│   ├── training/             # Training hyperparameters
│   └── experiments/          # Full experiment configs
│       └── production.yaml   # Production training setup
```

### Override Examples

```bash
# Use larger model
uv run src/train.py model=large

# Adjust training parameters
uv run src/train.py training.max_steps=10000 training.learning_rate=5e-5

# Change audio encoder
uv run src/train.py model.audio_model_id=facebook/hubert-large-ls960-ft

# Use experiment preset
uv run src/train.py +experiments=production

# Combine multiple overrides
uv run src/train.py model=large training.batch_size=32 training.max_steps=50000
```

### Key Configuration Parameters

- `model.audio_model_id`: Audio encoder model (default: `openai/whisper-small`)
- `model.text_model_id`: Language model decoder (default: `HuggingFaceTB/SmolLM3-3B-Base`)
- `model.audio_downsample_rate`: Audio feature downsampling factor (default: 5)
- `model.projector_hidden_dim`: Hidden dimension in projector MLP (default: 2048)
- `training.max_steps`: Total training steps
- `training.learning_rate`: Peak learning rate for cosine schedule
- `training.batch_size`: Per-device batch size

## Remote Training

Tiny Audio includes tools for easy deployment and training on cloud GPU providers like RunPod.

### Setup Remote Instance

```bash
# Deploy code to remote GPU instance
uv run scripts/deploy_runpod.py --host <pod-id>.runpod.io --port 22

# Start training on remote instance
uv run scripts/start_remote_training.py \
  --host <pod-id>.runpod.io \
  --port 22 \
  --config production
```

### Features

- Automatic code synchronization via rsync
- Remote training script execution
- Real-time log monitoring
- Checkpoint saving to HuggingFace Hub
- Supports RunPod, Lambda Labs, Vast.ai, and other SSH-accessible GPU providers

### Monitoring

The training script automatically:

- Logs metrics to TensorBoard
- Saves checkpoints to `outputs/` directory
- Pushes final model to HuggingFace Hub (if `HF_TOKEN` is set)
- Reports training progress and WER metrics

## Evaluation

Tiny Audio includes a comprehensive evaluation framework using Word Error Rate (WER) on the LoquaciousSet benchmark dataset.

### Quick Evaluation

```bash
# Evaluate tiny-audio model on 100 samples
uv run scripts/eval.py --max-samples 100

# Evaluate on full test set
uv run scripts/eval.py

# Compare with other models
uv run scripts/eval.py --provider huggingface --model openai/whisper-small
uv run scripts/eval.py --provider assemblyai --api-key YOUR_API_KEY
```

### Supported Providers

1. **tiny-audio** (default): Your trained model
2. **huggingface**: Any HuggingFace ASR model
3. **assemblyai**: AssemblyAI commercial API

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--provider` | Model provider (`tiny-audio`, `huggingface`, `assemblyai`) | `tiny-audio` |
| `--model` | HuggingFace model ID | Required for `huggingface` |
| `--api-key` | AssemblyAI API key | Required for `assemblyai` |
| `--max-samples` | Limit evaluation samples | All samples |
| `--config` | LoquaciousSet config (`large`, `clean`) | `large` |
| `--split` | Dataset split | `test` |

### Evaluation Metrics

- **Word Error Rate (WER)**: Primary metric for ASR performance
- Automatic text normalization (lowercasing, punctuation removal)
- Detailed per-sample predictions saved to `outputs/eval_{provider}/results.txt`

### Example Output

```text
Model: mazesmazes/tiny-audio
Dataset: speechbrain/LoquaciousSet (config: large, split: test)
Samples: 2000
WER: 12.3%

Sample predictions with ground truth comparison...
```

## Development

### Code Quality

```bash
# Format code with Ruff
uv run ruff format src/

# Lint and auto-fix issues
uv run ruff check src/ --fix

# Type checking with mypy
uv run mypy src/

# Run all checks
uv run ruff format src/ && uv run ruff check src/ --fix && uv run mypy src/
```

### Testing

```bash
# Run test suite
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html
```

### Project Structure

```text
tiny-audio/
├── src/                      # Source code
│   ├── asr_modeling.py      # Model architecture
│   ├── asr_config.py        # Model configuration
│   ├── asr_pipeline.py      # HuggingFace pipeline
│   ├── asr_processing.py    # Audio/text processing
│   ├── train.py             # Training script
│   └── handler.py           # Inference handler
├── configs/                 # Hydra configurations
│   └── hydra/
│       ├── config.yaml      # Main config
│       ├── model/           # Model configs
│       ├── training/        # Training configs
│       └── experiments/     # Experiment presets
├── scripts/                 # Utility scripts
│   ├── eval.py             # Evaluation script
│   ├── deploy_runpod.py    # Remote deployment
│   └── start_remote_training.py
├── demo/                    # Gradio demo app
└── tests/                   # Test suite
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/tiny-audio.git`
3. Install development dependencies: `uv sync` or `pip install -e ".[dev]"`
4. Create a feature branch: `git checkout -b feature/amazing-feature`
5. Make your changes and ensure tests pass
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## Citation

If you use Tiny Audio in your research, please cite:

```bibtex
@software{kroman2024tinyaudio,
  author = {Kroman, Alex},
  title = {Tiny Audio: Efficient Speech Recognition with Frozen Pretrained Models},
  year = {2024},
  url = {https://github.com/alexkroman/tiny-audio},
  note = {GitHub repository}
}
```

## Acknowledgments

This project builds upon:

- [Whisper](https://github.com/openai/whisper) by OpenAI for audio encoding
- [SmolLM3](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base) by HuggingFace for language modeling
- [SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM) for architectural inspiration

## License

MIT License - see [LICENSE](LICENSE) file for details.
