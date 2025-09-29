# Tiny Audio

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-mazesmazes%2Ftiny--audio-yellow)](https://huggingface.co/mazesmazes/tiny-audio)

A tiny speech recognition model

**🚀 Demo**: [Try it on Hugging Face Spaces](https://huggingface.co/spaces/mazesmazes/tiny-audio)
**📦 Pre-trained Model**: Available on
[Hugging Face Hub](https://huggingface.co/mazesmazes/tiny-audio)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/alexkroman/tiny-audio.git
   cd tiny-audio
   ```

1. **Install dependencies:** This project uses `uv` for fast dependency
   management.

   ```bash
   uv sync
   ```

## Quick Start

### Option 1: Use Pre-trained Model

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("mazesmazes/tiny-audio", trust_remote_code=True)
transcription = model.transcribe("audio.wav")
```

See the [model card](https://huggingface.co/mazesmazes/tiny-audio) for detailed
usage instructions.

### Option 2: Train Your Own Model

1. **Set up your Hugging Face token (required for some datasets):**

   ```bash
   export HF_TOKEN='your-hugging-face-token'
   ```

   You can get a token from
   [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

1. **Run a quick test training:** This will run for 20 steps and take
   approximately 2 minutes on a modern laptop.

   ```bash
   uv run src/train.py
   ```

1. **Start a full production training:** This uses larger datasets and is
   recommended for achieving better performance.

   ```bash
   uv run src/train.py +experiments=production
   ```

1. **Run the Gradio Demo:** Once you have a trained model, you can use the
   Gradio app to interact with it.

   ```bash
   # Make sure to replace the model path with your own
   uv run demo/app.py --model outputs/2025-09-22/12-51-14/outputs/mac_minimal_model
   ```

### Option 3: Deploy Demo to Hugging Face Spaces

Deploy your demo as a public Hugging Face Space for easy sharing:

1. **Create a new Space on Hugging Face:**
   - Go to [https://huggingface.co/new-space](https://huggingface.co/new-space)
   - Choose Gradio SDK
   - Name your Space (e.g., "tiny-audio-demo")

2. **Deploy using the automated script:**

   ```bash
   # Deploy to default space (mazesmazes/tiny-audio)
   uv run scripts/deploy_to_hf_space.py --force

   # Deploy to your own space
   uv run scripts/deploy_to_hf_space.py --space-url https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE --force
   ```

   The script automatically handles:
   - Copying only the necessary demo files
   - Creating a clean git repository
   - Pushing to the Hugging Face Space

3. **Access your live demo:**
   Your Space will be available at: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

   See our example Space: [https://huggingface.co/spaces/mazesmazes/tiny-audio](https://huggingface.co/spaces/mazesmazes/tiny-audio)

## Model Architecture

The model consists of three main components:

1. **W2V-BERT 2.0 Encoder**: A frozen encoder trained on 4.5M hours of audio
   across 143+ languages. (600M parameters, not trained)
2. **Audio Projector**: An attention-based projector
3. **Language Model Decoder**: SmolLM3

Audio Input → [W2V-BERT Encoder] → [Audio Projector] → [LM Decoder + LoRA] → Text Output

## Configuration

This project uses [Hydra](https://hydra.cc/) for configuration management. You
can find the configuration files in the `configs/hydra` directory.

- `config.yaml`: The base configuration.
- `model/`: Model architecture settings.
- `data/`: Dataset configurations.
- `training/`: Training hyperparameters.
- `experiments/`: Pre-configured experiment setups that combine the above.

### Overriding Configuration

You can easily override any configuration parameter from the command line:

```bash
# Use a different model
uv run src/train.py model=large

# Use a different dataset
uv run src/train.py data=production_streaming

# Change training parameters
uv run src/train.py training.max_steps=10000 training.eval_steps=500
```

## Cloud Training with RunPod

The `scripts/` directory contains utilities for deploying and managing training
on [RunPod](https://runpod.io) GPUs.

1. **Deploy your code:** This will sync your local code to the RunPod instance
   and install dependencies.

   ```bash
   uv run scripts/deploy_runpod.py --host <your-pod-id>.runpod.io --port 22
   ```

1. **Start remote training:** This will start a training session in a `tmux`
   window on the remote instance.

   ```bash
   uv run scripts/start_remote_training.py --host <your-pod-id>.runpod.io --port 22 --config production
   ```

1. **Attach to the remote session:** You can attach to the `tmux` session to
   monitor the training progress.

   ```bash
   uv run scripts/attach_remote_session.py --host <your-pod-id>.runpod.io --port 22
   ```

## Development

This project uses `ruff` for linting and formatting, and `mypy` for type
checking.

- **Linting:**

  ```bash
  uv run ruff check src/
  ```

- **Formatting:**

  ```bash
  uv run ruff format src/
  ```

- **Type Checking:**

  ```bash
  uv run mypy src/
  ```

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull
request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file
for details.
