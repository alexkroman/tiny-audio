# Tiny Audio - ASR Learning Project

A lightweight, efficient Automatic Speech Recognition (ASR) system that combines OpenAI's Whisper encoder with SmolLM2 decoder using LoRA adapters for parameter-efficient fine-tuning.

## Features

- **Hybrid Architecture**: Leverages frozen Whisper encoder for robust audio understanding with SmolLM2 decoder for efficient text generation
- **Parameter-Efficient**: Uses LoRA (Low-Rank Adaptation) for fine-tuning, drastically reducing trainable parameters
- **Streaming Support**: Handles large-scale datasets (LibriSpeech, GigaSpeech, Common Voice) without memory constraints
- **Flexible Configuration**: Hydra-based configuration system for easy experimentation
- **Multi-Platform**: Supports local training (including Mac M1/M2) and cloud deployment (RunPod)
- **Real-time Monitoring**: TensorBoard integration with WER tracking and prediction logging

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/alexkroman/tiny-audio.git
cd tiny-audio

# Install with uv
uv sync
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/alexkroman/tiny-audio.git
cd tiny-audio

# Install in development mode
pip install -e .

# For CUDA support
pip install -e ".[cuda]"

# For development tools
pip install -e ".[dev]"
```

## Quick Start

### Training on Sample Data

```bash
# Run with minimal configuration (great for testing)
python src/train.py +experiments=mac_minimal

# Run with default settings
python src/train.py
```

### Production Training

```bash
# Large-scale training with streaming datasets
python src/train.py +experiments=production

# Custom configuration
python src/train.py \
    model.lora_r=64 \
    model.lora_alpha=128 \
    training.batch_size=16 \
    training.learning_rate=5e-4
```

## Model Architecture

The system combines three key components:

1. **Whisper Encoder**: Frozen `whisper-small` model for audio feature extraction (39M parameters, not trained)
2. **Audio Projector**: Lightweight projection layer with RMSNorm and GELU activation to map audio features to text space
3. **SmolLM2 Decoder**: Efficient language model with LoRA adapters (360M or 1.7B parameters, ~2% trained with LoRA)

```python
Audio Input � Whisper Encoder � Audio Projector � SmolLM2 + LoRA � Text Output
```

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/hydra/config.yaml` - Base configuration
- `configs/hydra/model/` - Model architecture settings
- `configs/hydra/data/` - Dataset configurations
- `configs/hydra/training/` - Training hyperparameters
- `configs/hydra/experiments/` - Pre-configured experiment setups

### Example: Override Configuration

```bash
# Change model size
python src/train.py model=large

# Use different dataset
python src/train.py data=production_streaming

# Modify training parameters
python src/train.py training.max_steps=10000 training.eval_steps=500
```

## Datasets

Supports multiple ASR datasets through Hugging Face datasets:

- **LibriSpeech**: Clean and other subsets for English ASR
- **GigaSpeech**: Large-scale English speech recognition
- **Common Voice**: Multilingual community-collected speech data

Datasets are automatically downloaded and cached locally.

## Remote Training

Deploy and train on RunPod GPUs:

```bash
# Deploy to RunPod
python scripts/deploy_runpod.py

# Start remote training session
python scripts/start_remote_training.py

# Attach to running session
python scripts/attach_remote_session.py
```

## Monitoring

Training progress is logged to TensorBoard:

```bash
# View training metrics
tensorboard --logdir outputs/

# Metrics tracked:
# - Training/validation loss
# - Word Error Rate (WER)
# - Learning rate schedule
# - Sample predictions
```

## Performance

| Model Config | Parameters | Trainable | WER (LibriSpeech test-clean) | Training Time (A40) |
|-------------|-----------|-----------|-------------------------------|-------------------|
| Small + LoRA | 399M | 8M (2%) | ~15% | 2 hours |
| Large + LoRA | 1.7B | 34M (2%) | ~12% | 8 hours |

*Note: Results are illustrative. Actual performance depends on training configuration and dataset.*

## Development

### Code Quality

```bash
# Run linter
ruff check src/

# Format code
ruff format src/

# Type checking
mypy src/
```

### Testing

```bash
# Quick training test
python src/train.py +experiments=mac_minimal

# Verify model loading
python -c "from src.modeling import ASRModel, ASRModelConfig; model = ASRModel(ASRModelConfig())"
```

## Mac-Specific Setup

For Apple Silicon (M1/M2) devices:

```bash
# Set environment variables for MPS backend
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Run training
python src/train.py +experiments=mac_minimal
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{tiny_audio_2024,
  title = {Tiny Audio: Efficient ASR with Whisper + SmolLM2},
  author = {Kroman, Alex},
  year = {2024},
  url = {https://github.com/alexkroman/tiny-audio}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for the Whisper model
- Hugging Face for SmolLM2 and the transformers library
- The teams behind LibriSpeech, GigaSpeech, and Common Voice datasets

## Contact

Alex Kroman - alex@alexkroman.com

Project Link: [https://github.com/alexkroman/tiny-audio](https://github.com/alexkroman/tiny-audio)