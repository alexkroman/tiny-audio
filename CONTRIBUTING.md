# Contributing to Tiny Audio

Reference material for working on the codebase: the CLI, configuration, project layout, and the
quality gates every change goes through. For using the model or a first training run, start with the
[README](README.md).

## Setup

```bash
git clone https://github.com/alexkroman/tiny-audio.git && cd tiny-audio
poetry install
poetry run ta dev test   # confirm everything passes before you change anything
```

## CLI Reference

All commands available via `tiny-audio` (or `ta` for short):

```bash
poetry run ta --help  # Show all commands
```

| Command     | Description                                                                         |
| ----------- | ----------------------------------------------------------------------------------- |
| `ta train`  | Train locally: `ta train asr -e <preset>` (Hydra overrides pass through)            |
| `ta eval`   | Evaluate ASR models on datasets                                                     |
| `ta deploy` | Deploy demo to HuggingFace Space                                                    |
| `ta push`   | Push model to HuggingFace Hub                                                       |
| `ta demo`   | Launch local Gradio demo                                                            |
| `ta debug`  | Debug utilities (check-gradient-flow)                                               |
| `ta runpod` | Remote training on RunPod (plan, up, wait, deploy, train, attach, eval, checkpoint) |
| `ta dev`    | Development tools (lint, format, type-check, test, check, precommit, ...)           |

### CLI conventions

Every command follows the same rules, and `tests/test_cli_conventions.py` checks them against the
built command tree:

- Commands that *run* something (`eval`, `demo`, `runpod eval`, `dev handler`) take the model as
  `--model/-m`. Commands that *inspect* a model (`debug *`) take it as the first positional
  argument.
- `ta runpod` commands that talk to a pod take `<HOST> <PORT>` as their first two arguments.
- Every option has an explicit `--long-name` and help text. Secrets and IDs that usually come from
  the environment (`HF_TOKEN`, `ASSEMBLYAI_API_KEY`, `WANDB_RUN_ID`, `MODEL_ID`, ...) are options
  with an `[env var: ...]` fallback, so `--help` shows where each value comes from.
- Choices (`--datasets`, `--assemblyai-model`, `--component`, `--dtype`) are validated by the CLI
  and listed in `--help`; paths passed with `--output-dir`, `--demo-dir`, `--checkpoint-dir` and
  `--audio` are checked before the command runs.
- A short flag means one thing everywhere, and a long flag keeps the same short flag everywhere:

| Option          | Short | Description                                       |
| --------------- | ----- | ------------------------------------------------- |
| `--model`       | `-m`  | Model path or Hub ID                              |
| `--datasets`    | `-d`  | Datasets to evaluate (repeatable, `all` expands)  |
| `--max-samples` | `-n`  | Maximum samples per dataset                       |
| `--output-dir`  | `-o`  | Directory eval results are written to / read from |
| `--num-workers` | `-w`  | Parallel workers for API evaluations              |
| `--streaming`   | `-s`  | Streaming evaluation                              |
| `--config`      | `-c`  | Dataset config override                           |
| `--experiment`  | `-e`  | Experiment config (`ta train asr`, `ta runpod`)   |
| `--force`       | `-f`  | Kill an existing tmux session first (`ta runpod`) |
| `--repo-id`     | `-r`  | Hub repo to push or deploy to                     |
| `--branch`      | `-b`  | Hub branch (`ta push`)                            |
| `--list`        | `-l`  | List tmux sessions (`ta runpod attach`)           |
| `--port`        | `-p`  | Server port (`ta demo`)                           |
| `--audio`       | `-a`  | Audio file (`ta dev handler`)                     |

## Configuration

Configuration uses [Hydra](https://hydra.cc/). Override any value with `key=value` syntax:

```bash
# Override model settings
poetry run python scripts/train.py +experiments=stage_1 model.projector_hidden_dim=2048

# Override training settings
poetry run python scripts/train.py +experiments=stage_1 \
  training.decoder_learning_rate=1e-5 training.per_device_train_batch_size=50

# Swap the dataset config
poetry run python scripts/train.py +experiments=stage_1 data=librispeech_dummy
```

The YAML files are validated against a typed
[structured-config](https://hydra.cc/docs/tutorials/structured_config/intro/) schema in
`scripts/train_config.py`, so a misspelled key or a wrongly typed value fails at startup. To add a
new knob, declare it there; for a one-off run, append an undeclared key with
`+training.<key>=<value>`.

### Config Files

```text
configs/
├── config.yaml                  # Main config (model defaults; imports data + training)
├── experiments/                 # Training recipes
│   ├── stage_1.yaml             # Default: frozen encoder + projector + decoder trained jointly
│   ├── granite_qwen_frozen.yaml # Published model: Granite Speech encoder + LoRA on a Qwen3.5 decoder
│   └── mps_smoke.yaml           # Local CPU/MPS smoke test
├── data/
│   ├── multiasr.yaml            # Ten-corpus training mix (default)
│   └── librispeech_dummy.yaml   # 73-clip sample for smoke tests
└── training/
    └── production.yaml          # Training hyperparameters
```

### Key Config Parameters

Defaults from `configs/config.yaml` and `configs/training/production.yaml`:

```yaml
model:
  audio_model_id: zai-org/GLM-ASR-Nano-2512   # Audio encoder
  text_model_id: Qwen/Qwen3-0.6B              # Language model
  projector_type: mlp
  projector_pool_stride: 4                    # Frames stacked per projector input
  projector_hidden_dim: 1024

training:
  freeze_language_model: false                # Decoder is fine-tuned
  freeze_projector: false
  learning_rate: 1e-3                         # Projector
  decoder_learning_rate: 2e-5                 # Decoder
  per_device_train_batch_size: 100
  max_steps: -1                               # Train by epochs instead
  num_train_epochs: 2                         # stage_1 overrides this to 1
  warmup_steps: 2000
  lr_scheduler_type: cosine_with_min_lr
```

The encoder is frozen by the `freeze_audio_encoder` default in `ASRConfig`; set
`training.freeze_audio_encoder=false` (or `training.encoder_trainable_top_layers=N` for the top N
blocks) to train it.

## Project Structure

```text
tiny-audio/
├── tiny_audio/              # Core library
│   ├── asr_modeling.py      # ASRModel: encoder + projector + decoder
│   ├── asr_layers.py        # MPS-safe embeddings, encoder top-layer unfreezing
│   ├── asr_attention.py     # attn_implementation choice (FA2 / sdpa / eager)
│   ├── asr_types.py         # Typing aliases, protocols and TypedDicts
│   ├── asr_config.py        # ASRConfig: all model settings
│   ├── asr_pipeline.py      # HuggingFace pipeline for inference
│   ├── asr_processing.py    # ASRProcessor: audio/text preprocessing
│   ├── projectors.py        # Projector architectures
│   ├── alignment.py         # Forced alignment for word timestamps
│   ├── diarization.py       # Speaker diarization
│   └── handler.py           # HF Inference Endpoints handler
├── scripts/
│   ├── train.py             # Training script (Hydra)
│   ├── cli.py               # Unified CLI entry point
│   ├── dev.py               # Development utilities
│   ├── eval/                # Evaluation framework
│   │   ├── evaluators/      # ASR evaluators
│   │   └── datasets.py      # Dataset loading
│   ├── deploy/              # RunPod, run planning, HF Space deployment
│   ├── hub/                 # HF Hub integration
│   └── debug/               # Gradient-flow check
├── configs/                 # Hydra configuration
├── tests/                   # Test suite
└── docs/                    # Documentation and course
```

## Development

### Running Tests

```bash
poetry run ta dev test                    # Run all tests (enforces the coverage floor)
poetry run pytest tests/test_projectors.py -v  # Single file
poetry run pytest -k "test_forward" -v    # By name pattern
```

### Code Quality

```bash
poetry run ta dev format      # Format code (black, ruff, mdformat)
poetry run ta dev lint        # Lint + format check (poetry check --lock, ruff, black, yamllint, taplo,
                              #   actionlint, zizmor, mdformat --check)
poetry run ta dev type-check  # Type check (pyright, strict)
poetry run ta dev check       # Lint + type-check + security + dead code + duplication + ratchets
                              #   + deptry + docstrings
poetry run ta dev precommit   # Full quality gate
```

`ta dev check` and `ta dev test` also run three ratchets (`scripts/quality.py`), each against a
baseline committed under `quality/` that may only improve:

- **File length:** Python files are capped at 600 code lines; files already over it may not grow.
- **Per-file coverage:** each file keeps its recorded line coverage, and new files need 50%. Runs
  after the tests, from `coverage.json`.
- **Test assertions:** every test must assert something (an `assert`, `pytest.raises`, a mock
  `assert_*`, or a call to a guard such as `_require_*`).

After an intended improvement, refresh a baseline with `ta dev quality file-length --update` or
`ta dev quality coverage-floors --update` (record coverage floors from a run with Hub access) and
commit the diff.

### Adding a New Projector

1. Add your projector class to `tiny_audio/projectors.py`:

   ```python
   class MyProjector(nn.Module):
       def __init__(self, config):
           super().__init__()
           # config.encoder_dim, config.llm_dim, config.projector_pool_stride are available
           # Your architecture here

       def forward(self, x: torch.Tensor) -> torch.Tensor:
           # x: [batch, seq_len, encoder_dim] -> [batch, out_len, llm_dim]
           return x

       def get_output_length(self, input_length: int) -> int:
           # Return output sequence length given input length
           return input_length
   ```

1. Register it in the `PROJECTOR_CLASSES` dict in `projectors.py`:

   ```python
   PROJECTOR_CLASSES = {
       "mlp": MLPAudioProjector,
       "my_projector": MyProjector,  # Add here
   }
   ```

1. Create an experiment config `configs/experiments/my_projector.yaml`:

   ```yaml
   # @package _global_
   model:
     projector_type: my_projector
   ```

1. Train: `poetry run python scripts/train.py +experiments=my_projector`

### Adding a New Dataset

1. Add a config file `configs/data/my_dataset.yaml`. Each entry in `datasets:` is one Hub dataset
   (or one config of it); see `configs/data/librispeech_dummy.yaml` and `configs/data/multiasr.yaml`
   for the full set of fields:

   ```yaml
   datasets:
     - path: your-org/your-dataset
       name: en                  # Optional dataset config name
       audio_column: audio
       text_column: text
       task: transcribe
       text_case: cased          # or `mono` for lowercase / ALL-CAPS labels
       text_punct: true
       train_splits: [train]
       eval_splits: [validation]
       target_samples: 600000    # Optional cap on the train split

   sample_rate: 16000
   dataset_cache_dir: ${hydra:runtime.cwd}/datasets_cache
   max_eval_samples_per_dataset: 500
   ```

1. Train with your dataset:
   `poetry run python scripts/train.py +experiments=stage_1 data=my_dataset`

### Key Files to Understand

| File                | Purpose                  | When to Modify                               |
| ------------------- | ------------------------ | -------------------------------------------- |
| `asr_modeling.py`   | Core model class         | Adding model features, changing forward pass |
| `asr_config.py`     | Configuration            | Adding new config parameters                 |
| `projectors.py`     | Projector architectures  | Adding new projector types                   |
| `asr_processing.py` | Audio/text preprocessing | Changing input processing                    |
| `train.py`          | Training loop            | Modifying training behavior                  |

## Environment Variables

| Variable             | Description                                        |
| -------------------- | -------------------------------------------------- |
| `HF_TOKEN`           | HuggingFace API token (for private models/pushing) |
| `WANDB_API_KEY`      | Weights & Biases API key                           |
| `WANDB_RUN_ID`       | Resume a specific W&B run                          |
| `ASSEMBLYAI_API_KEY` | For AssemblyAI evaluation comparison               |
