# Quick Reference

The commands and numbers from the course on one page.

______________________________________________________________________

## Essential Commands

### Setup and Development

```bash
poetry install                                    # Python 3.12 required
poetry run hf auth login                          # Hugging Face credentials
poetry run ta --help                              # every command group

poetry run ta demo --model mazesmazes/tiny-audio  # Gradio demo on :7860
poetry run ta dev test                            # test suite
poetry run ta dev check                           # lint + type-check + security + docstrings
poetry run ta dev format                          # black, ruff, mdformat
```

### Training (Local)

```bash
# 10-step smoke test on 73 clips (laptop friendly)
poetry run python scripts/train.py +experiments=mps_smoke

# Production recipe (80 GB GPU, terabyte of data)
poetry run python scripts/train.py +experiments=stage_1

# Your own experiment, with overrides
poetry run python scripts/train.py +experiments=my_run training.per_device_train_batch_size=16

# Projector + LoRA instead of full decoder fine-tuning
poetry run python scripts/train.py +experiments=my_run training.use_lora=true

# Projector only (decoder frozen)
poetry run python scripts/train.py +experiments=my_run training.freeze_language_model=true

# Resume
poetry run python scripts/train.py +experiments=my_run training.resume_from_checkpoint=<dir>/checkpoint-1000
```

### Training (RunPod)

```bash
poetry run ta runpod plan --experiment my_run           # VRAM, disk, GPU pick (no download)
poetry run ta runpod deploy <HOST> <PORT>               # rsync code + install deps
poetry run ta runpod deploy <HOST> <PORT> --skip-setup --skip-deps   # sync only

export HF_TOKEN='hf_...'                                # write token
poetry run ta runpod train <HOST> <PORT> --experiment my_run [hydra overrides...]

poetry run ta runpod attach <HOST> <PORT>               # reattach to tmux
poetry run ta runpod attach <HOST> <PORT> --logs --lines 200   # print recent output
poetry run ta runpod checkpoint <HOST> <PORT>           # newest checkpoint path
poetry run ta runpod eval <HOST> <PORT> -m <model> -d loquacious -n 500

# Optional, needs runpodctl + API key
poetry run ta runpod up --experiment my_run --dry-run
poetry run ta runpod wait <pod-id>
```

### Evaluation

```bash
poetry run ta eval -m <model> -n 200                    # LoquaciousSet test (default)
poetry run ta eval -m <model> -d earnings22 -d ami -n 100
poetry run ta eval -m <model> -d all -n 100
poetry run ta eval -m assemblyai -n 200 -w 4            # needs ASSEMBLYAI_API_KEY
poetry run ta eval -m deepgram -n 200 -w 4              # needs DEEPGRAM_API_KEY
poetry run ta eval -m elevenlabs -n 200 -w 4            # needs ELEVENLABS_API_KEY
poetry run ta eval -m apple-speech -n 200               # macOS only
poetry run ta eval -m https://<endpoint>.endpoints.huggingface.cloud --endpoint -n 50
```

Results: `outputs/<timestamp>_<short-name>_<dataset>/{results.txt,metrics.txt}`

### Analysis

```bash
# Worst samples of a run
grep -h "WER:" outputs/*_<short-name>_<dataset>/results.txt | sort -t: -k2 -rn | head -20
# Corpus WER of every run on a dataset
grep -H "^wer:" outputs/*_<dataset>/metrics.txt

poetry run ta debug check-gradient-flow <model>
```

### Publishing and Deployment

```bash
# Weights: pushed automatically during training (push_to_hub + hub_model_id)

poetry run ta push --repo-id <model>                  # custom code + MODEL_CARD.md + requirements (no weights)
poetry run ta deploy --repo-id <user>/<space>         # upload demo/ to a Gradio Space
# then set MODEL_ID=<model> in the Space's Settings → Variables
```

______________________________________________________________________

## Architecture

```
Audio → GLM-ASR encoder (frozen) → MLP projector (trained) → Qwen3-0.6B (fine-tuned) → Text
```

| Component | Model | Params | Trains? | LR |
|-----------|-------|--------|---------|----|
| Audio encoder | GLM-ASR-Nano-2512, encoder only | ~635M | No | none |
| Projector | RMSNorm → Linear → GELU → Linear | ~6.3M | Yes, from scratch | 1e-3 |
| Decoder | Qwen3-0.6B | ~600M | Yes, fine-tuned | 2e-5 |

### Shapes (10 s of audio)

| Stage | Shape | Rate |
|-------|-------|------|
| Waveform | 160,000 | 16 kHz |
| Log-mel | 128 × 1000 | 100 frames/s |
| Encoder output | 500 × 1280 | 50 frames/s |
| Stacked (k = 4) | 125 × 5120 | 12.5 tokens/s |
| Projector output | 125 × 1024 | 12.5 tokens/s |

**Frame stacking**: `output_length = (input_length - k) // k + 1`, `k = 4`.

**Prompt the decoder sees**: `<audio>…<audio> Transcribe the speech to text` as the user turn,
transcript as the assistant turn, via Qwen3's chat template.

______________________________________________________________________

## Experiments (`configs/experiments/`)

| Config | Encoder | Decoder | Trains | Data |
|--------|---------|---------|--------|------|
| `stage_1` | GLM-ASR-Nano (frozen) | Qwen3-0.6B | Projector + decoder + embeddings | `multiasr` |
| `granite_qwen_frozen` | Granite Speech 470M | Qwen3.5-4B (frozen) | Projector + decoder LoRA | `multiasr` |
| `mps_smoke` | GLM-ASR-Nano | Qwen3-0.6B | 10 steps, batch 1 | `librispeech_dummy` |

### Freeze Flags

| Flag | Default | Effect |
|------|---------|--------|
| `training.freeze_audio_encoder` | `true` | Encoder fixed |
| `training.freeze_language_model` | `false` | `true` = projector-only training |
| `training.freeze_text_embed_tokens` | `true` | Embedding table fixed |
| `training.freeze_projector` | `false` | `true` = LoRA-only training |
| `training.use_lora` | `false` | LoRA adapters (rank 8, alpha 32) instead of full fine-tune |

______________________________________________________________________

## Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `training.learning_rate` | `1e-3` | Projector |
| `training.decoder_learning_rate` | `2e-5` | Decoder |
| `training.per_device_train_batch_size` | `100` | Production, 80 GB GPU |
| `training.gradient_accumulation_steps` | `1` | |
| `training.num_train_epochs` | `2` | `stage_1`: 1 |
| `training.max_steps` | `-1` | Positive value overrides epochs |
| `training.warmup_steps` | `2000` | |
| `training.lr_scheduler_type` | `cosine_with_min_lr` | floor = `min_lr_rate` × peak |
| `training.max_grad_norm` | `2.5` | `stage_1`: 1.0 |
| `training.weight_decay` | `0.01` | projector: 0.0 |
| `training.eval_steps`, `save_steps` | `2000` | |
| `model.projector_pool_stride` | `4` | |
| `model.projector_hidden_dim` | `1024` | |
| `model.label_smoothing` | `0.1` | `stage_1`: 0.0 |

______________________________________________________________________

## Evaluation Datasets (`-d`)

| Name | Domain |
|------|--------|
| `loquacious` | Mixed English (default) |
| `librispeech`, `librispeech-other` | Audiobooks |
| `tedlium` | TED talks |
| `commonvoice` | Crowd-sourced read speech |
| `voxpopuli` | European Parliament |
| `peoples` | Public-domain speech |
| `gigaspeech` | Podcasts, YouTube |
| `earnings22` | Earnings calls |
| `spgispeech` | Financial presentations |
| `ami`, `ami-sdm` | Meetings (headset / distant mic) |
| `expresso` | Expressive speech |

______________________________________________________________________

## Training Metrics

| Metric | Healthy | Warning |
|--------|---------|---------|
| `train/loss` | Below 1.0 within ~500 steps, then slowly falling | Flat, or `NaN` |
| `eval/loss` | Tracks training loss | Rising while training loss falls (overfit) |
| `train/grad_norm` | 1-3 after warmup | Spikes over 100 |

______________________________________________________________________

## Config Layout

```
configs/
├── config.yaml               # model defaults, imports data + training
├── training/production.yaml  # trainer defaults
├── data/                     # multiasr, librispeech_dummy, (your own)
└── experiments/              # recipes; use with +experiments=<name>
```

Override syntax is `key=value` (Hydra), never `--key value`. Experiment files start with
`# @package _global_`.

______________________________________________________________________

## Common Options

| Option | Short | Applies to |
|--------|-------|------------|
| `--model` | `-m` | `eval`, `demo`, `runpod eval` |
| `--datasets` | `-d` | `eval` (repeatable, or `all`) |
| `--max-samples` | `-n` | `eval` |
| `--num-workers` | `-w` | `eval` with API backends |
| `--output-dir` | `-o` | `eval` (default `outputs`) |
| `--experiment` | `-e` | `train asr`, `runpod plan`, `runpod train`, `runpod up` |
| `--repo-id` | `-r` | `push`, `deploy` |

______________________________________________________________________

## tmux (on the pod)

| Action | Keys |
|--------|------|
| Detach | `Ctrl+B`, then `D` |
| Scroll | `Ctrl+B`, then `[`; `q` to exit |
| Stop training | `Ctrl+C` |

______________________________________________________________________

## Environment Variables

Every variable below also has a flag on the command that reads it, and `--help` shows the
pairing as `[env var: ...]`. Pass the flag to override the environment for one run.

| Variable | Purpose | Flag |
|----------|---------|------|
| `HF_TOKEN` | Hub downloads and checkpoint uploads (write token for training) | `ta push --hf-token`, `ta runpod train/eval --hf-token` |
| `WANDB_API_KEY` | Weights & Biases login | — |
| `WANDB_RUN_ID`, `WANDB_RESUME` | Resume a W&B run | `ta runpod train --wandb-run-id / --wandb-resume` |
| `MODEL_ID` | Model served by the Gradio demo / Space | `ta demo --model` |
| `ASSEMBLYAI_API_KEY`, `DEEPGRAM_API_KEY`, `ELEVENLABS_API_KEY` | Commercial API baselines | `ta eval --assemblyai-api-key / --deepgram-api-key / --elevenlabs-api-key` |

______________________________________________________________________

## Common Issues

| Problem | Fix |
|---------|-----|
| Poetry refuses the Python version | Install 3.12; `poetry env use python3.12` |
| CUDA out of memory | Lower `per_device_train_batch_size`, raise `gradient_accumulation_steps`, or add `training.use_lora=true` |
| Pod out of disk | `ta runpod plan` before renting; data caches at ~2× download size |
| `HF_TOKEN` warning at launch | Export a write token (or pass `--hf-token`) before `ta runpod train` |
| `not logged in` from `ta push` | Run `hf auth login`, export `HF_TOKEN`, or pass `--hf-token`; the push needs a write token |
| W&B prompts for login | Paste your key, or pass `training.report_to=none` |
| Hydra "could not override" | Use `key=value`; check the key exists in `scripts/train_config.py` (the config schema), or add it with `+training.<key>=` |
| `analysis` finds no results | Use the short model name (after the last `/`); it must match exactly |
| Space serves the wrong model | Set `MODEL_ID` in the Space's variables |

______________________________________________________________________

## Formulas

**Frame stacking**: `output_length = (input_length - k) // k + 1`

**WER**: `(Substitutions + Insertions + Deletions) / Reference words`, after Whisper text
normalization on both sides.

______________________________________________________________________

[← Class 3: Evaluation](./3-evaluation-and-deployment.md) | [Glossary →](./5-glossary.md)
