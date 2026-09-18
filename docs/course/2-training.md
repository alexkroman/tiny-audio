# Class 2: Training

*1 hour of class time (20 min lecture + 40 min hands-on), then the run continues on its own*

**Goal**: Run the training loop locally, size a cloud GPU, write your own experiment config,
and launch a real training run.

> **Before you start.** You need the install from Class 1, a Hugging Face **write** token, a
> W&B account, and a RunPod account with credit. Have `configs/experiments/stage_1.yaml` open;
> the lecture refers to it.

---

## Part A: Lecture (20 min)

### What `scripts/train.py` Does

1. **Loads the two pretrained models** from the Hub: the GLM-ASR encoder and Qwen3-0.6B.
   The decoder's master weights are kept in float32 so tiny updates aren't rounded away;
   compute runs in bfloat16.
2. **Builds a fresh projector** and calibrates its output scale against the decoder's
   embedding table.
3. **Loads the datasets** listed in the data config. This is a normal `load_dataset()` in map
   mode, not streaming: everything is downloaded and cached to disk, and `datasets` keeps two
   copies (the Hub download plus its own Arrow tables). Budget about twice the download size.
4. **Normalizes labels.** Corpora that ship all-lowercase or all-uppercase transcripts are
   re-cased with a statistical truecaser so the decoder sees one consistent format. Unicode is
   cleaned up. Clips shorter than 0.8 s or longer than 19 s are dropped, as are rows whose
   label normalizes to nothing.
5. **Collates batches** into chat-template conversations (see Class 1): audio placeholders
   plus a prompt in the user turn, the transcript in the assistant turn. Loss is computed on
   the assistant turn only.
6. **Trains** with the Hugging Face `Trainer` and AdamW. Three parameter groups get three
   learning rates: projector 1e-3, decoder 2e-5, encoder none (frozen). Cosine schedule with
   warmup, gradient clipping, bf16.
7. **Evaluates and checkpoints** every 2,000 steps. With `push_to_hub: true` every checkpoint
   is uploaded to your Hub repo as it is saved, so you can evaluate mid-run from any machine.

### Key Metrics

| Metric | W&B name | What to look for |
|--------|----------|------------------|
| **Training loss** | `train/loss` | Cross-entropy on the assistant turn. Falls fast, then slowly |
| **Eval loss** | `train/eval_loss` or `eval/loss` | Same quantity on held-out clips. Should track training loss |
| **Gradient norm** | `train/grad_norm` | Starts around 5-10, settles into 1-3 after warmup |
| **Learning rate** | `train/learning_rate` | Ramps up over warmup, then decays along a cosine |

Signs of trouble:

- Training loss flat after a few hundred steps: learning rate too low, or the data isn't
  reaching the model (check label normalization output).
- Eval loss rising while training loss falls: overfitting. Shorten the run or add data.
- Gradient norm spiking above 100: instability. Check for bad samples or lower the LR.
- `NaN`: one corrupt clip is enough. The collator filters the common cases; if it happens,
  find the batch.

### What Convergence Looks Like

The projector starts as random noise, so early on the decoder receives garbage and the loss is
high. In the published `stage_1` runs on the production mix:

- **Steps 0 to ~500**: loss falls from ~3.5 to under 1.0 as the projector finds the basic
  audio-token-to-word mapping. Transcripts go from gibberish to recognizable within this
  window.
- **Steps ~500 to ~1,500**: loss settles toward ~0.4 and the gradient norm stabilizes.
- **After that**: slow, steady improvement for the rest of the run. The decoder fine-tune is
  doing most of the work here.

Earlier versions of this course described a "cliff" around step 1,500 where the model suddenly
started working. That was the projector-only recipe. With the decoder unfrozen the transition
comes much earlier and is less abrupt. On a smaller dataset with a smaller batch, expect the
same shape stretched or compressed along the step axis.

### The Recipes

| Experiment | Encoder | Decoder | What trains | Data |
|------------|---------|---------|-------------|------|
| `stage_1` | GLM-ASR-Nano (frozen) | Qwen3-0.6B | Projector + decoder + embeddings | `multiasr` |
| `encoder_train` | Whisper-medium.en (**trained**) | Qwen3-0.6B (frozen) | Projector + encoder | `multiasr` |
| `granite_qwen` | Granite Speech 470M (frozen) | Qwen3.5-2B | Projector + decoder | `multiasr` |
| `granite_gemma` | Granite Speech 470M (frozen) | Gemma 4 E2B (frozen) | Projector only | `loquacious_medium` |
| `mps_smoke` | GLM-ASR-Nano (frozen) | Qwen3-0.6B | Projector + decoder, 10 steps | `librispeech_dummy` |

Every recipe is the same code with different freeze flags and model IDs. The flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `training.freeze_audio_encoder` | `true` | Encoder weights never change |
| `training.freeze_language_model` | `false` | Set `true` for a projector-only run |
| `training.freeze_text_embed_tokens` | `true` | Keep Qwen3's embedding table fixed (protects rare tokens) |
| `training.freeze_projector` | `false` | Set `true` to train LoRA adapters alone |
| `training.use_lora` | `false` | Add LoRA adapters to the decoder instead of full fine-tuning |

**LoRA** (Low-Rank Adaptation) adds small trainable matrices next to the decoder's linear
layers and leaves the original weights untouched. It is the cheaper alternative to full
fine-tuning: ~1-2M trainable parameters instead of ~600M. Use it when GPU memory is tight or
when you want to preserve the decoder exactly:

```bash
# Projector + LoRA adapters, original decoder weights frozen
poetry run python scripts/train.py +experiments=stage_1 training.use_lora=true
```

---

## Part B: Hands-On (40 min)

### Exercise 1: Local Smoke Test (10 min, free)

Run the complete training loop on your laptop with 73 LibriSpeech clips. This downloads both
base models (about 6 GB on first run) and trains for 10 steps at batch size 1. It won't
produce a useful model; it proves that your environment, the data pipeline, and the trainer
all work.

```bash
poetry run python scripts/train.py +experiments=mps_smoke
```

Watch for:

- The parameter summary: encoder frozen, projector and decoder trainable.
- The label-normalization log lines.
- Ten `loss` values. They should be finite and roughly decreasing.

It needs about 12 GB of memory (fp32 master weights plus AdamW state for the decoder). If it
dies with an out-of-memory error, your machine is still fine for Classes 1 and 3; move on to
Exercise 2 and do your training in the cloud.

### Exercise 2: Size Your Cloud Run (5 min)

Never guess at GPU size. `ta runpod plan` reads a config, measures the parameter counts and
download sizes from the Hub without downloading weights, and reports the VRAM and disk you
need:

```bash
poetry run ta runpod plan --experiment stage_1
```

Read the output. For the production recipe it will tell you that `multiasr` is over a terabyte
on disk and that full-decoder fine-tuning at batch 100 wants an 80 GB GPU. That's the run
behind the published model. It is not the run you'll do in class.

### Exercise 3: Write Your Experiment (10 min)

You'll train on LoquaciousSet `small`: 250 hours, ~107K clips, 23 GB of parquet (about 50 GB
on disk after caching). It is a balanced mix of read and spontaneous, clean and noisy English,
and its `test` split is the default `ta eval` benchmark, so you can score your model fairly
afterward.

**Data config** `configs/data/loquacious_small.yaml`:

```yaml
# LoquaciousSet `small`: 250 h / 107K clips, 22.9 GB of parquet.
datasets:
  - path: speechbrain/LoquaciousSet
    name: small
    audio_column: wav        # this repo stores audio in a column called `wav`
    text_column: text
    task: transcribe
    train_splits: [train]
    eval_splits: [dev]       # `test` is reserved for `ta eval`

sample_rate: 16000
dataset_cache_dir: ${hydra:runtime.cwd}/datasets_cache
max_eval_samples_per_dataset: 200
```

**Experiment config** `configs/experiments/my_run.yaml`:

```yaml
# @package _global_
defaults:
  - override /data: loquacious_small

model:
  label_smoothing: 0.0

training:
  hub_model_id: "your-username/tiny-audio-yourname"   # CHANGE THIS
  wandb_project: tiny-audio-course
  num_train_epochs: 1
  warmup_steps: 300
  eval_steps: 500
  save_steps: 500
  per_device_train_batch_size: 32
  per_device_eval_batch_size: 32
  gradient_accumulation_steps: 2
```

Everything you don't override comes from `configs/config.yaml` and
`configs/training/production.yaml`: the GLM-ASR encoder, Qwen3-0.6B, the two learning rates,
the cosine schedule, bf16, and `push_to_hub: true`. One epoch at an effective batch of 64 is
roughly 1,700 steps, so checkpoints every 500 steps gives you four to compare.

LoquaciousSet's transcripts are uppercase without punctuation. The truecaser will restore
casing, but nothing restores punctuation, so expect your model's output to be lightly
punctuated. WER is unaffected because the scorer strips punctuation from both sides.

Now size *your* run:

```bash
poetry run ta runpod plan --experiment my_run
```

Note the recommended GPU and the disk figure. Both go into the next step.

### Exercise 4: Set Up RunPod (10 min)

1. Sign up at [runpod.io](https://runpod.io) and add credit.
2. Add your SSH key under Settings → SSH Public Keys:
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   cat ~/.ssh/id_ed25519.pub
   ```
3. Deploy a pod:
   - GPU: what `ta runpod plan` recommended (an A40 or A6000 class 48 GB card is typical for
     this run; drop batch size or add `training.use_lora=true` for a 24 GB card)
   - Template: **RunPod PyTorch** (the code expects the image's CUDA-enabled PyTorch)
   - Container disk: at least the disk figure from the plan, with headroom
4. Once it's running, copy the **SSH host** and **port** from the pod's Connect panel.

Optional shortcut: if you install `runpodctl` and set your RunPod API key,
`ta runpod up --experiment my_run` sizes the config and creates the pod for you, and
`ta runpod wait <pod-id>` prints the host and port once SSH is up. Try `--dry-run` first.

**Deploy your code** (rsyncs the repo and installs dependencies, 5-10 minutes):

```bash
poetry run ta runpod deploy <HOST> <PORT>
```

Re-run with `--skip-setup --skip-deps` after local edits to sync only the files.

### Exercise 5: Train (5 min to launch)

```bash
export HF_TOKEN='hf_...'        # a WRITE token: checkpoints are pushed to your Hub repo
poetry run ta runpod train <HOST> <PORT> --experiment my_run
```

This starts training inside a `tmux` session on the pod and attaches you to it. The trainer
reports to Weights & Biases by default. If the pod isn't logged in, `wandb` will prompt in the
terminal: choose "use an existing account" and paste the key from
[wandb.ai/authorize](https://wandb.ai/authorize). To skip W&B entirely, append
`training.report_to=none` to the command above.

Hydra overrides pass straight through, so you can adjust without editing the file:

```bash
poetry run ta runpod train <HOST> <PORT> --experiment my_run training.per_device_train_batch_size=16
```

**Detach** with `Ctrl+B` then `D`. The run keeps going. **Reattach** later:

```bash
poetry run ta runpod attach <HOST> <PORT>
poetry run ta runpod attach <HOST> <PORT> --logs -n 200   # just print recent output
poetry run ta runpod checkpoint <HOST> <PORT>              # path of the newest checkpoint
```

**Monitor in W&B**. Within the first few hundred steps `train/loss` should drop below 1.0. If
you set `eval_steps: 500`, the first eval loss arrives at step 500.

**When to stop early** (`Ctrl+C` inside tmux):

- Loss hasn't moved after 500 steps
- Eval loss climbing while training loss falls
- You've seen enough and want to save money

### Exercise 6: Terminate the Pod

**RunPod bills by the hour whether or not the GPU is busy.**

| Action | Effect |
|--------|--------|
| **Stop** | Pauses GPU billing. Disk is kept and still billed |
| **Terminate** | Ends all billing. Disk is deleted |

Your checkpoints are already on the Hub (`push_to_hub`), so there is nothing to lose by
terminating. Re-deploying takes 5-10 minutes if you want to train again.

---

## Local Training (Optional)

The same commands work on a local CUDA GPU. The production recipe needs an 80 GB card; the
course run from Exercise 3 fits on 48 GB, or on 24 GB with LoRA:

```bash
# Your experiment
poetry run python scripts/train.py +experiments=my_run

# Same, but LoRA instead of full decoder fine-tuning
poetry run python scripts/train.py +experiments=my_run training.use_lora=true

# Resume after an interruption
poetry run python scripts/train.py +experiments=my_run \
    training.resume_from_checkpoint=outputs/<date>/<time>/outputs/production_model/checkpoint-1000
```

Hydra changes into a fresh `outputs/<date>/<time>/` directory for each run, so checkpoints land
under that directory rather than in the repo root.

---

## Configuration Reference

### Training (`configs/training/production.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.learning_rate` | `1e-3` | Projector learning rate |
| `training.decoder_learning_rate` | `2e-5` | Decoder learning rate (50x lower) |
| `training.num_train_epochs` | `2` | Epochs over the data config (`stage_1` uses 1) |
| `training.max_steps` | `-1` | Set a positive number to cap the run regardless of epochs |
| `training.per_device_train_batch_size` | `100` | Clips per step. Size for your GPU |
| `training.gradient_accumulation_steps` | `1` | Effective batch = batch size × this |
| `training.warmup_steps` | `2000` | Linear LR ramp before the cosine decay |
| `training.lr_scheduler_type` | `cosine_with_min_lr` | Decays to `min_lr_rate` × peak (0.1) |
| `training.max_grad_norm` | `2.5` | Gradient clipping |
| `training.weight_decay` | `0.01` | Decoder weight decay (projector uses `projector_weight_decay: 0.0`) |
| `training.eval_steps` / `save_steps` | `2000` | Evaluate and checkpoint every N steps |
| `training.push_to_hub` | `true` | Upload each checkpoint to `hub_model_id` (public by default) |
| `training.bf16` | `true` | Mixed precision. Master weights stay fp32 |
| `training.attn_implementation` | `flash_attention_2` | Falls back to SDPA where unavailable |

### Model (`configs/config.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.audio_model_id` | `zai-org/GLM-ASR-Nano-2512` | Audio encoder (only its encoder is used) |
| `model.text_model_id` | `Qwen/Qwen3-0.6B` | Decoder |
| `model.projector_type` | `mlp` | The only registered projector type |
| `model.projector_pool_stride` | `4` | Frame-stacking factor `k` |
| `model.projector_hidden_dim` | `1024` | Width of the projector's hidden layer |
| `model.label_smoothing` | `0.1` | Applied inside the loss during training (`stage_1` sets 0.0) |
| `model.model_dtype` | `float32` | Master weight precision |

---

## Key Takeaways

1. Training is a normal `Trainer` run with a frozen encoder and two learning rates: hot for the
   fresh projector, cool for the pretrained decoder.
2. Data is downloaded and cached, not streamed. Plan disk at twice the download size.
3. `ta runpod plan` replaces guesswork about GPU and disk.
4. The loss falls hardest in the first 500 steps; the rest is refinement.
5. **Terminate the pod when you're done.** Your checkpoints are already on the Hub.

## Before Class 3

- [ ] Your run has pushed at least one checkpoint: check `https://huggingface.co/<hub_model_id>`
      for `model.safetensors`
- [ ] You've glanced at the W&B curves and can say whether the loss dropped in the first 500
      steps
- [ ] The pod is stopped or terminated (or you know exactly why it's still running)

If the run isn't finished by class time, that's fine. Class 3 works on any checkpoint that has
been pushed, and falls back to the published model if you have none.

---

[← Class 1](./1-introduction-and-setup.md) | [Class 3: Evaluation →](./3-evaluation-and-deployment.md)
