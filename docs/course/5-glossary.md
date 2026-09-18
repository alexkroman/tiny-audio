# Glossary

Terms used in the course, grouped by topic.

---

## The Model

**ASR (Automatic Speech Recognition)**
Converting spoken audio into written text.

**Encoder**
The audio model that turns a spectrogram into a sequence of feature vectors. Tiny Audio uses
the encoder half of GLM-ASR-Nano-2512: 32 Transformer layers, 1280-dimensional output, one
vector every 20 ms. Frozen during training. In code it is `model.audio_tower`.

**Projector**
The trainable bridge between encoder features and the decoder's embedding space. A two-layer
MLP with frame stacking, about 6.3M parameters, trained from scratch. In code, `model.projector`.

**Decoder**
The language model that writes the transcript, conditioned on the projected audio tokens and a
text prompt. Tiny Audio uses Qwen3-0.6B, fine-tuned at a low learning rate. In code,
`model.language_model`. "Decoder", "language model", and "LLM" are used interchangeably.

**GLM-ASR**
An open speech recognition model from Z.ai (Zhipu). Tiny Audio borrows only its audio encoder
and discards its own decoder.

**Qwen3**
Alibaba's open language-model family. Qwen3-0.6B is the smallest member and ships with a chat
template, which the training collator requires.

**Frame Stacking**
Concatenating `k` adjacent encoder frames into one wider vector, cutting the sequence length by
`k`. With `k = 4`, 50 encoder frames per second become 12.5 audio tokens per second.
`output_length = (input_length - k) // k + 1`.

**Audio Token**
One row of the projector's output: a 1024-dimensional vector standing in for about 80 ms of
speech, inserted into the decoder's input where an `<audio>` placeholder token sits.

**Chat Template**
The formatting a chat model expects: `<|im_start|>user … <|im_end|>` and so on. Tiny Audio
builds a user turn of `<audio>` placeholders followed by "Transcribe the speech to text", and
an assistant turn containing the transcript. Loss is computed on the assistant turn only.

**Output Scale**
A fixed multiplier at the end of the projector, calibrated once so audio tokens have the same
magnitude as Qwen3's word embeddings. Not learned.

**RMSNorm**
A normalization layer that rescales a vector to unit root-mean-square. Used at the projector's
input and throughout Qwen3.

**MLP (Multi-Layer Perceptron)**
Stacked linear layers with a nonlinearity (here GELU) between them.

**Attention / Transformer**
The mechanism and architecture underlying both the encoder and the decoder: every position can
attend to every other, weighted by relevance.

**Flash Attention / SDPA**
Two implementations of attention. Flash Attention 2 is fastest on recent NVIDIA GPUs; the
model falls back to PyTorch's scaled-dot-product attention (SDPA) elsewhere.

---

## Audio

**Waveform**
The raw audio signal: amplitude over time. Tiny Audio expects 16 kHz mono.

**Sample Rate**
Measurements per second. 16 kHz means 16,000 samples per second of audio.

**Log-Mel Spectrogram**
Audio represented as energy per frequency band over time, with bands spaced on the mel scale
(roughly how humans hear pitch) and energies on a log scale. GLM-ASR uses 128 bands at 100
frames per second.

**SpecAugment**
Data augmentation that masks random time spans of the spectrogram during training. Available
via `model.apply_spec_augment`; most useful when the encoder is trainable.

---

## Training

**Frozen**
A module whose weights are excluded from the optimizer and never change. The encoder is frozen
in every recipe; the decoder can be frozen with `training.freeze_language_model=true`.

**Fine-tuning**
Continuing to train a pretrained model on a new task, usually at a much lower learning rate
than it was pretrained with. The decoder is fine-tuned at 2e-5 while the projector trains at
1e-3.

**LoRA (Low-Rank Adaptation)**
Instead of updating a large weight matrix, learn a small low-rank correction beside it. Cuts
trainable parameters from hundreds of millions to a few million and leaves the original weights
intact. `training.use_lora=true`.

**Loss (Cross-Entropy)**
How surprised the decoder is by the correct next token, averaged over the transcript. Training
minimizes it.

**Label Smoothing**
Spreading a little of the target probability over wrong tokens so the model isn't pushed to
absolute certainty. On by default (0.1) in `config.yaml`, off in the `stage_1` recipe.

**Step**
One optimizer update after one (accumulated) batch.

**Epoch**
One pass over the training data. With a fixed dataset, steps per epoch = rows ÷ effective
batch size.

**Batch Size / Gradient Accumulation**
Clips processed per forward pass, and how many passes to sum before updating. Effective batch
= batch size × accumulation steps. Accumulation trades speed for memory.

**Learning Rate (LR)**
Step size of each weight update. Too high diverges, too low crawls.

**Warmup**
Ramping the LR linearly from zero over the first N steps so early, noisy gradients don't wreck
the pretrained weights. 2000 steps in production.

**Cosine Schedule with Minimum LR**
After warmup, the LR follows a cosine curve down to a floor (`min_lr_rate` × peak) rather than
to zero, so late training still makes progress.

**Gradient**
The direction that would increase the loss; the optimizer moves the other way.

**Gradient Norm**
The overall size of the gradient. Healthy runs settle into a narrow band (1-3 here) after
warmup. Spikes signal instability.

**Gradient Clipping**
Capping the gradient norm (`max_grad_norm`) so one bad batch can't take a huge step.

**Weight Decay**
A gentle pull of weights toward zero each step. Applied to the decoder (0.01), never to the
projector.

**AdamW**
The optimizer. Keeps a running mean and variance of each parameter's gradient, which is why it
needs two extra copies of every trainable parameter in memory.

**Mixed Precision (bf16)**
Computing in 16-bit bfloat16 for speed while keeping fp32 master weights so small updates
aren't rounded away.

**Overfitting**
The model memorizes training clips instead of generalizing. Visible as eval loss rising while
training loss keeps falling.

**Checkpoint**
A saved snapshot of the trainable weights plus optimizer state. Written every `save_steps` and
pushed to the Hub when `push_to_hub` is on.

**Truecasing**
Restoring capitalization to text that arrived all-lowercase or all-uppercase. Tiny Audio runs a
statistical truecaser over such training labels so the decoder sees one consistent format.

**Data Collator**
The function that turns a list of raw rows into one padded batch: extracts features, filters
bad clips, builds chat-template conversations, tokenizes.

---

## Evaluation

**WER (Word Error Rate)**
`(Substitutions + Insertions + Deletions) / Reference words`. The standard ASR metric. Lower is
better; can exceed 100%.

**Text Normalization**
Transforming reference and prediction into a canonical form before scoring: lowercase, no
punctuation, standardized numbers and spellings. Tiny Audio uses Whisper's
`EnglishTextNormalizer`. Casing and punctuation are therefore invisible to WER.

**Homophones**
Words that sound alike but are spelled differently ("to / too / two"). The decoder's language
knowledge is what resolves them.

**Entity**
A name, place, organization, number, or date in a transcript. `ta analysis extract-entities`
tags them; `entity-errors` reports the ones your model got wrong.

**Corpus WER vs. Per-Sample WER**
Corpus WER pools all edits and all reference words across a dataset. Per-sample WER is
computed per clip and is what `high-wer` sorts by.

---

## Tools

**Hydra**
The configuration framework. YAML files under `configs/`, composed with `defaults:` lists,
overridable from the command line with `key=value`.

**Hugging Face Hub**
Where models, datasets, and Spaces live. Training checkpoints are pushed here; `ta push`
uploads the custom code and model card.

**Hugging Face Space**
A hosted web app. `ta deploy` uploads the `demo/` directory to one.

**Inference Endpoints**
Hugging Face's paid, dedicated model hosting. The repo's `handler.py` makes Tiny Audio models
deployable there.

**Gradio**
The Python library behind the demo UI.

**Poetry**
The dependency manager. `poetry install`, then `poetry run <command>`.

**RunPod**
A GPU rental service. `ta runpod plan / deploy / train / attach / checkpoint` wrap the workflow.

**tmux**
A terminal multiplexer. Training runs inside a tmux session on the pod so it survives your SSH
connection dropping.

**W&B (Weights & Biases)**
Experiment tracking. The trainer logs loss, eval loss, gradient norm, and learning rate there
every 50 steps.

**CUDA / MPS**
NVIDIA's and Apple's GPU compute platforms. Production training targets CUDA; the smoke test
runs on MPS or CPU.

**VRAM**
GPU memory. Determines the largest batch (and model) you can train. `ta runpod plan` estimates
what a config needs.

---

[← Quick Reference](./4-quick-reference.md) | [Course Overview](./0-course-overview.md)
