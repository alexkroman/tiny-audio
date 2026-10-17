# Build Your Own Speech Recognition Model

A hands-on course: build, train, evaluate, and deploy your own ASR model with Tiny Audio.

|                   |                                                                                      |
| ----------------- | ------------------------------------------------------------------------------------ |
| **Duration**      | 3 hours 20 min of class time, plus an unattended training run                        |
| **Cost**          | Classes 1 and 3 are free. Class 2's cloud run costs GPU time (see [Budget](#budget)) |
| **Prerequisites** | Python, the command line, git                                                        |

## What You'll Build

```text
Audio → Granite Speech encoder (frozen) → MLP projector (trained)
      → Qwen3.5-2B decoder (frozen + LoRA) → Text
```

| Component     | Model                                             | Params                 | During training                           |
| ------------- | ------------------------------------------------- | ---------------------- | ----------------------------------------- |
| Audio encoder | `ibm-granite/granite-speech-5.0-470m-turboctc-nc` | ~473M                  | Frozen                                    |
| Projector     | 2-layer MLP (hidden 4096)                         | ~12.6M                 | Trained from scratch (lr 1e-3)            |
| Decoder       | `Qwen/Qwen3.5-2B`                                 | ~1.9B base + ~67M LoRA | Base frozen; LoRA r=64 adapters (lr 1e-4) |

You reuse two pretrained models and teach them to talk to each other. The projector is the only part
built from nothing; the decoder's own weights never change, and small LoRA adapters on its linear
layers learn to read audio tokens. About 3% of the 4.8B parameters train. Then you publish the
result to the Hugging Face Hub and put a live demo on the web.

> **This is the default recipe, on less data.** The course run inherits `configs/config.yaml` and
> `configs/training/production.yaml` (the same Granite + Qwen3.5 + LoRA recipe family the published
> `mazesmazes/tiny-audio` comes from) and changes the data, ~250 hours instead of the full
> `multiasr` mix, plus a few schedule settings. Expect a higher WER from your course run.

### Four Ideas to Leave With

The commands change; these don't. Each class comes back to them.

1. **Two skills, two parts.** Speech recognition needs a good ear and a good command of language.
   The encoder is the ear, the decoder is the language model, and almost every error is a hearing
   mistake or a language mistake. Knowing which tells you what to fix. (Class 1)
1. **Weights stay, vectors flow.** A model is a stack of learned grids of numbers. Audio becomes
   vectors that pass through them; training only changes the grids. (Class 1)
1. **Whatever the labels lack, the model will lack.** The training loop is simple; the data mix and
   how consistently its transcripts are written decide the result. (Class 2)
1. **One benchmark is not an evaluation.** A WER depends on the domain and on how the reference
   transcripts were written and normalized, as much as on the model. (Class 3)

______________________________________________________________________

## Course Structure

### [Class 1: Introduction and Setup](./1-introduction-and-setup.md)

**Time:** 1 hour 20 min

- How an encoder, a projector, and a decoder turn audio into text
- Exactly what the projector computes, with real shapes and sizes
- Environment setup
- Running inference with the published model and exploring the `ta` CLI

### [Class 2: Training](./2-training.md)

**Time:** 1 hour, then training runs on its own

- What `scripts/train.py` does, step by step
- Sizing a cloud GPU with `ta runpod plan`
- Writing your own experiment config and launching on RunPod
- Reading the loss and gradient curves

### [Class 3: Evaluation and Deployment](./3-evaluation-and-deployment.md)

**Time:** 1 hour

- Word Error Rate (WER) and text normalization
- Evaluating across 13 datasets and against commercial APIs
- Finding your model's worst samples and entity errors
- Publishing the model and deploying a Gradio demo to Hugging Face Spaces

### How the Classes Fit Together

Class 1 gives you a working install and the mental model. Class 2 uses both to write a config and
launch a run. The run keeps going after class, so leave a few hours (or a day) before Class 3, which
evaluates and deploys the checkpoints the run pushed to the Hub. If your run isn't ready, Class 3
works unchanged with the published model `mazesmazes/tiny-audio`; nothing in it is blocked on your
own training.

### [Quick Reference](./4-quick-reference.md)

Commands, hyperparameters, troubleshooting.

### [Glossary](./5-glossary.md)

Key terms defined.

______________________________________________________________________

## Budget

The cost of Class 2 is entirely yours to choose. Three tiers:

| Tier                                        | Data                                                   | Hardware                          | Cost                                    |
| ------------------------------------------- | ------------------------------------------------------ | --------------------------------- | --------------------------------------- |
| **Free run** (`ta train asr -e tiny`)       | LibriSpeech dev-clean (5.4 hours), Qwen3.5-0.8B        | Your laptop or a free Colab GPU   | Free                                    |
| **Course run** (what Class 2 walks through) | LoquaciousSet `small` (~250 hours)                     | One rented 80 GB GPU, a few hours | A few GPU-hours at RunPod's hourly rate |
| **Production recipe** (the default config)  | The `multiasr` mix, ~3M clips, over a terabyte on disk | An 80 GB GPU for a day or more    | Hundreds of dollars                     |

`ta runpod plan` measures the GPU memory and disk any config needs before you rent anything. Class 2
shows you how to use it.

______________________________________________________________________

## Requirements

**Development (Classes 1 and 3):** any modern laptop with 8 GB RAM and 20 GB of free disk. Loading
the published model downloads about 6 GB (its own weights plus the encoder's repo).

**Training (Class 2):** a RunPod account and a GPU. There is no laptop-sized training recipe: the
default config is a 2B decoder sized for one 80 GB GPU, so Class 2 trains in the cloud. The class
shows you how to size the GPU with `ta runpod plan`.

______________________________________________________________________

## Accounts Needed

1. [GitHub](https://github.com)
1. [Hugging Face](https://huggingface.co) (create a write token for pushing checkpoints)
1. [Weights & Biases](https://wandb.ai) (training curves)
1. [RunPod](https://runpod.io) (Class 2 only)

______________________________________________________________________

[Start Class 1 →](./1-introduction-and-setup.md)
