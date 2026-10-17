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
# Default recipe: Granite + Qwen3.5-2B + LoRA (80 GB GPU, terabyte of data). No laptop recipe.
poetry run python scripts/train.py

# Your own experiment (configs/experiments/my_run.yaml), with overrides
poetry run python scripts/train.py +experiments=my_run \
    training.per_device_train_batch_size=32 training.gradient_accumulation_steps=2

# Projector only (no LoRA adapters, decoder fully frozen)
poetry run python scripts/train.py +experiments=my_run training.use_lora=false

# Full decoder fine-tune instead of LoRA (much more memory; plan first)
poetry run python scripts/train.py +experiments=my_run training.freeze_language_model=false training.use_lora=false

# Resume
poetry run python scripts/train.py +experiments=my_run training.resume_from_checkpoint=<dir>/checkpoint-1000
```

### Training (RunPod)

```bash
poetry run ta runpod plan                               # size the default config (no download)
poetry run ta runpod plan --experiment my_run           # VRAM, disk, GPU pick for your config
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
poetry run ta eval -m https://<pod-id>-8000.proxy.runpod.net --endpoint -n 50 -w 8
```

Results: `outputs/<timestamp>_<short-name>_<dataset>/{results.txt,metrics.txt}`

### Analysis

```bash
# Worst samples of a run
grep -h "WER:" outputs/*_<short-name>_<dataset>/results.txt | sort -t: -k2 -rn | head -20
# Corpus WER of every run on a dataset
grep -H "^wer:" outputs/*_<dataset>/metrics.txt
```

### Publishing and Deployment

```bash
# Weights: pushed automatically during training (push_to_hub + hub_model_id)

# custom code + MODEL_CARD.md + requirements (no weights)
poetry run ta push --repo-id <model>
poetry run ta deploy --repo-id <user>/<space>         # upload demo/ to a Gradio Space
# then set MODEL_ID=<model> in the Space's Settings → Variables
```

______________________________________________________________________

## Architecture

```text
Audio → Granite Speech encoder (frozen) → MLP projector (trained) → Qwen3.5-2B (frozen + LoRA) → Text
```

| Component     | Role                | Model                                             | Params       | Trains?                | LR   |
| ------------- | ------------------- | ------------------------------------------------- | ------------ | ---------------------- | ---- |
| Audio encoder | The ear             | `ibm-granite/granite-speech-5.0-470m-turboctc-nc` | ~473M        | No                     | none |
| Projector     | The translator      | RMSNorm → Linear → GELU → Linear (hidden 4096)    | ~12.6M       | Yes, from scratch      | 1e-3 |
| Decoder       | The language expert | `Qwen/Qwen3.5-2B`                                 | ~1.9B + ~67M | Base no; LoRA r=64 yes | 1e-4 |

### Shapes (10 s of audio, default config)

| Stage            | Shape      | Rate          |
| ---------------- | ---------- | ------------- |
| Waveform         | 160,000    | 16 kHz        |
| Encoder output   | 125 × 1024 | 12.5 frames/s |
| Projector output | 125 × 2048 | 12.5 tokens/s |

**Frame stacking**: `output_length = (input_length - k) // k + 1`, with `k` =
`model.projector_pool_stride` (default `1`, no stacking; Class 1 walks through `k = 4` on the
earlier GLM-ASR + Qwen3-0.6B pairing).

**Prompt the decoder sees**: `<audio>…<audio>` plus a transcription prompt as the user turn,
transcript as the assistant turn, via the decoder's chat template.

______________________________________________________________________

## Recipes

No experiment presets ship in the repo: `configs/config.yaml` + `configs/training/production.yaml`
is the recipe, and `poetry run python scripts/train.py` trains it. To layer your own overrides,
create `configs/experiments/<name>.yaml` (starting with `# @package _global_`) and pass
`+experiments=<name>` (or `--experiment <name>` to the `ta` commands).

| Default               | Encoder                      | Decoder             | Trains                          | Data       |
| --------------------- | ---------------------------- | ------------------- | ------------------------------- | ---------- |
| `configs/config.yaml` | Granite Speech 470M (frozen) | Qwen3.5-2B (frozen) | Projector + decoder LoRA (~80M) | `multiasr` |

### Freeze Flags

| Flag                                | Default | Effect                                                                       |
| ----------------------------------- | ------- | ---------------------------------------------------------------------------- |
| `training.freeze_audio_encoder`     | `true`  | Encoder fixed                                                                |
| `training.freeze_language_model`    | `true`  | Decoder base fixed (LoRA adapters still train)                               |
| `training.freeze_text_embed_tokens` | `true`  | Embedding table fixed                                                        |
| `training.freeze_projector`         | `false` | `true` = LoRA-only training                                                  |
| `model.use_lora`                    | `true`  | LoRA adapters (rank 64, alpha 64, all-linear); `training.use_lora` overrides |

______________________________________________________________________

## Key Hyperparameters

| Parameter                              | Default                  | Notes                           |
| -------------------------------------- | ------------------------ | ------------------------------- |
| `training.learning_rate`               | `1e-3`                   | Projector                       |
| `training.decoder_learning_rate`       | `1e-4`                   | Decoder group (LoRA adapters)   |
| `training.per_device_train_batch_size` | `64`                     | Sized for an 80 GB GPU          |
| `training.gradient_accumulation_steps` | `1`                      | Effective batch 64              |
| `training.per_device_eval_batch_size`  | `32`                     |                                 |
| `training.num_train_epochs`            | `1`                      |                                 |
| `training.max_steps`                   | `-1`                     | Positive value overrides epochs |
| `training.warmup_steps`                | `1000`                   |                                 |
| `training.lr_scheduler_type`           | `cosine`                 | Decays to zero (no minimum LR)  |
| `training.max_grad_norm`               | `1.0`                    |                                 |
| `training.weight_decay`                | `0.0`                    | Decoder (LoRA); projector: 0.01 |
| `training.seed`                        | `44`                     |                                 |
| `training.eval_steps`, `save_steps`    | `2000`                   |                                 |
| `training.attn_implementation`         | `sdpa`                   |                                 |
| `model.projector_pool_stride`          | `1`                      | No frame stacking               |
| `model.projector_hidden_dim`           | `4096`                   |                                 |
| `model.encoder_conv_layers`            | `[[0, 2, 2], [0, 2, 2]]` | Granite's ~4x subsampling       |
| `model.model_dtype`                    | `bfloat16`               | Frozen encoder/decoder storage  |
| `model.projector_dtype`                | `float32`                |                                 |

______________________________________________________________________

## Evaluation Datasets (`-d`)

| Name                               | Domain                           |
| ---------------------------------- | -------------------------------- |
| `loquacious`                       | Mixed English (default)          |
| `librispeech`, `librispeech-other` | Audiobooks                       |
| `tedlium`                          | TED talks                        |
| `commonvoice`                      | Crowd-sourced read speech        |
| `voxpopuli`                        | European Parliament              |
| `peoples`                          | Public-domain speech             |
| `gigaspeech`                       | Podcasts, YouTube                |
| `earnings22`                       | Earnings calls                   |
| `spgispeech`                       | Financial presentations          |
| `ami`, `ami-sdm`                   | Meetings (headset / distant mic) |
| `expresso`                         | Expressive speech                |

______________________________________________________________________

## Training Metrics

| Metric            | Healthy                               | Warning                                    |
| ----------------- | ------------------------------------- | ------------------------------------------ |
| `train/loss`      | Falls steeply early, then slowly      | Flat, or `NaN`                             |
| `eval/loss`       | Tracks training loss                  | Rising while training loss falls (overfit) |
| `train/grad_norm` | Settles after warmup (clipped at 1.0) | Spikes over 100                            |

______________________________________________________________________

## Config Layout

```text
configs/
├── config.yaml               # the default recipe's model settings, imports data + training
├── training/production.yaml  # the default recipe's trainer settings
├── data/                     # multiasr, librispeech_dummy, (your own)
└── experiments/              # not shipped; your own override files, +experiments=<name>
```

Override syntax is `key=value` (Hydra), never `--key value`. Experiment files you create start with
`# @package _global_`.

______________________________________________________________________

## Common Options

| Option          | Short | Applies to                                     |
| --------------- | ----- | ---------------------------------------------- |
| `--model`       | `-m`  | `eval`, `demo`, `runpod eval`                  |
| `--datasets`    | `-d`  | `eval` (repeatable, or `all`)                  |
| `--max-samples` | `-n`  | `eval`                                         |
| `--num-workers` | `-w`  | `eval` with API backends                       |
| `--output-dir`  | `-o`  | `eval` (default `outputs`)                     |
| `--experiment`  | `-e`  | `train asr`, `runpod plan/train/up` (optional) |
| `--repo-id`     | `-r`  | `push`, `deploy`                               |

______________________________________________________________________

## tmux (on the pod)

| Action        | Keys                            |
| ------------- | ------------------------------- |
| Detach        | `Ctrl+B`, then `D`              |
| Scroll        | `Ctrl+B`, then `[`; `q` to exit |
| Stop training | `Ctrl+C`                        |

______________________________________________________________________

## Environment Variables

Every variable below also has a flag on the command that reads it, and `--help` shows the pairing as
`[env var: ...]`. Pass the flag to override the environment for one run.

| Variable                                                       | Purpose                                                         | Flag                                                                       |
| -------------------------------------------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `HF_TOKEN`                                                     | Hub downloads and checkpoint uploads (write token for training) | `ta push --hf-token`, `ta runpod train/eval --hf-token`                    |
| `WANDB_API_KEY`                                                | Weights & Biases login                                          | —                                                                          |
| `WANDB_RUN_ID`, `WANDB_RESUME`                                 | Resume a W&B run                                                | `ta runpod train --wandb-run-id / --wandb-resume`                          |
| `MODEL_ID`                                                     | Model served by the Gradio demo / Space                         | `ta demo --model`                                                          |
| `ASSEMBLYAI_API_KEY`, `DEEPGRAM_API_KEY`, `ELEVENLABS_API_KEY` | Commercial API baselines                                        | `ta eval --assemblyai-api-key / --deepgram-api-key / --elevenlabs-api-key` |

______________________________________________________________________

## Which Part Failed?

| Error looks like                                    | Kind             | Look at                   |
| --------------------------------------------------- | ---------------- | ------------------------- |
| An unrelated word where the audio was unclear       | Hearing mistake  | Encoder; audio in the mix |
| Right sound, wrong word or spelling ("their/there") | Language mistake | Decoder                   |
| Digits vs words, punctuation, casing                | Formatting gap   | References and normalizer |
| Empty output or endless repetition                  | Mode failure     | Loss curve, gradient flow |

______________________________________________________________________

## Common Issues

| Problem                           | Fix                                                                                                                       |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Poetry refuses the Python version | Install 3.12; `poetry env use python3.12`                                                                                 |
| CUDA out of memory                | Lower `per_device_train_batch_size` and raise `gradient_accumulation_steps` to keep the effective batch                   |
| Pod out of disk                   | `ta runpod plan` before renting; data caches at ~2× download size                                                         |
| `HF_TOKEN` warning at launch      | Export a write token (or pass `--hf-token`) before `ta runpod train`                                                      |
| `not logged in` from `ta push`    | Run `hf auth login`, export `HF_TOKEN`, or pass `--hf-token`; the push needs a write token                                |
| W&B prompts for login             | Paste your key, or pass `training.report_to=none`                                                                         |
| Hydra "could not override"        | Use `key=value`; check the key exists in `scripts/train_config.py` (the config schema), or add it with `+training.<key>=` |
| `analysis` finds no results       | Use the short model name (after the last `/`); it must match exactly                                                      |
| Space serves the wrong model      | Set `MODEL_ID` in the Space's variables                                                                                   |

______________________________________________________________________

## Formulas

**Frame stacking**: `output_length = (input_length - k) // k + 1`

**WER**: `(Substitutions + Insertions + Deletions) / Reference words`, after Whisper text
normalization on both sides.

______________________________________________________________________

[← Class 3: Evaluation](./3-evaluation-and-deployment.md) | [Glossary →](./5-glossary.md)
