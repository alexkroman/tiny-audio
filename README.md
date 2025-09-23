# Tiny Audio - Learn ASR by Building One

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

A minimal (~300 line) speech recognition model that combines Whisper's audio understanding with Hugging Face's SmolLM2 for text generation. This project is designed to be a hands-on guide to understanding modern ASR systems by building and training your own.

## Features

- **Minimalist Core**: The core model is just 300 lines of readable Python, making it easy to understand completely.
- **Modern Architecture**: Combines a frozen Whisper encoder with a SmolLM2 decoder using LoRA adapters and a custom projection layer.
- **Laptop-Friendly**: Train on your local machine (including M1/M2 Macs) or scale up to powerful GPUs.
- **Real-World Datasets**: Supports training on standard ASR datasets like LibriSpeech, GigaSpeech, and Common Voice.
- **Live Monitoring**: Track your model's progress in real-time with TensorBoard integration.
- **Flexible Configuration**: Easily experiment with different model sizes, datasets, and hyperparameters using Hydra.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/alexkroman/tiny-audio.git
    cd tiny-audio
    ```

2.  **Install dependencies:**
    This project uses `uv` for fast dependency management.
    ```bash
    uv sync
    ```
    Alternatively, you can use pip:
    ```bash
    pip install -e .
    ```

## Quick Start

1.  **Run a quick test training:**
    This will run for 20 steps and take approximately 2 minutes on a modern laptop.
    ```bash
    python src/train.py
    ```

2.  **Start a full production training:**
    This uses larger datasets and is recommended for achieving better performance.
    ```bash
    python src/train.py +experiments=production
    ```

3.  **Run the Gradio Demo:**
    Once you have a trained model, you can use the Gradio app to interact with it.
    ```bash
    # Make sure to replace the model path with your own
    python demo/gradio_app.py --model outputs/2025-09-22/12-51-14/outputs/mac_minimal_model
    ```

## Model Architecture

The model consists of three main components:

1.  **Whisper Encoder**: A frozen `whisper-small` model that extracts audio features. (39M parameters, not trained)
2.  **Audio Projector**: A lightweight projection layer that maps audio features to the text embedding space.
3.  **SmolLM2 Decoder**: An efficient language model from Hugging Face with LoRA adapters for parameter-efficient fine-tuning. (360M or 1.7B parameters, ~2% trained with LoRA)

```
Audio Input → [Whisper Encoder] → [Audio Projector] → [SmolLM2 + LoRA] → Text Output
```

## Configuration

This project uses [Hydra](https://hydra.cc/) for configuration management. You can find the configuration files in the `configs/hydra` directory.

-   `config.yaml`: The base configuration.
-   `model/`: Model architecture settings.
-   `data/`: Dataset configurations.
-   `training/`: Training hyperparameters.
-   `experiments/`: Pre-configured experiment setups that combine the above.

### Overriding Configuration

You can easily override any configuration parameter from the command line:

```bash
# Use a different model
python src/train.py model=large

# Use a different dataset
python src/train.py data=production_streaming

# Change training parameters
python src/train.py training.max_steps=10000 training.eval_steps=500
```

## Cloud Training with RunPod

The `scripts/` directory contains utilities for deploying and managing training on [RunPod](https://runpod.io) GPUs.

1.  **Deploy your code:**
    This will sync your local code to the RunPod instance and install dependencies.
    ```bash
    python scripts/deploy_runpod.py --host <your-pod-id>.runpod.io --port 22
    ```

2.  **Start remote training:**
    This will start a training session in a `tmux` window on the remote instance.
    ```bash
    python scripts/start_remote_training.py --host <your-pod-id>.runpod.io --port 22 --config production
    ```

3.  **Attach to the remote session:**
    You can attach to the `tmux` session to monitor the training progress.
    ```bash
    python scripts/attach_remote_session.py --host <your-pod-id>.runpod.io --port 22
    ```

## Development

This project uses `ruff` for linting and formatting, and `mypy` for type checking.

-   **Linting:**
    ```bash
    ruff check src/
    ```
-   **Formatting:**
    ```bash
    ruff format src/
    ```
-   **Type Checking:**
    ```bash
    mypy src/
    ```

## Monitoring

Training progress, including loss and Word Error Rate (WER), is logged to TensorBoard.

```bash
tensorboard --logdir outputs/
```

## Project Structure

```
├── configs/            # Hydra configuration files
├── demo/               # Gradio demo application
├── scripts/            # Scripts for cloud deployment and training
├── src/                # Source code
│   ├── modeling.py     # The core ASR model
│   └── train.py        # The training script
├── .gitignore
├── CLAUDE.md           # Notes for AI-assisted development
├── pyproject.toml
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.