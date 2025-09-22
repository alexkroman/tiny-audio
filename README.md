# Tiny Audio - Learn ASR by Building One

A minimal (~300 line) speech recognition model that combines Whisper's audio understanding with Hugging Face SmolLM2's text generation. Perfect for learning how modern ASR systems work by building and training your own.

## Why This Project

- **Actually Tiny**: The core model is just 300 lines of readable Python - small enough to understand completely
- **Modern Architecture**: Combines a frozen Whisper encoder (for audio) with Hugging Face SmolLM2 decoder (for text) using LoRA adapters and a custom projection layer
- **Trains on Consumer Hardware**: Works on your laptop (even M1/M2 Macs!) or scale up to GPUs
- **Real Datasets**: Train on actual speech data from LibriSpeech, GigaSpeech, or Common Voice
- **See Progress Live**: Watch your model improve in real-time with TensorBoard
- **Experiment Freely**: Simple config files let you try different ideas without touching code

## Installation

```bash
# Clone the repository
git clone https://github.com/alexkroman/tiny-audio.git
cd tiny-audio

# Install with uv
uv sync
```

## Quick Start

### Training

```bash
# Quick test run (20 steps, ~2 minutes)
python src/train.py

# Production training with larger datasets
python src/train.py +experiments=production
```

## Model Architecture

The system combines three key components:

1. **Whisper Encoder**: Frozen `whisper-small` model for audio feature extraction (39M parameters, not trained)
2. **Audio Projector**: Lightweight projection layer with RMSNorm and GELU activation to map audio features to text space
3. **Hugging Face SmolLM2 Decoder**: Efficient language model with LoRA adapters (360M or 1.7B parameters, ~2% trained with LoRA)

```python
Audio Input → Whisper Encoder → Audio Projector → Hugging Face SmolLM2 + LoRA → Text Output
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
- Hugging Face for the SmolLM2 model and transformers library
- The teams behind LibriSpeech, GigaSpeech, and Common Voice datasets