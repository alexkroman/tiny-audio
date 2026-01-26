# Glossary of Terms

Key terminology used throughout the course.

______________________________________________________________________

## Machine Learning Fundamentals

**Batch Size**
The number of samples processed together in one training step. Larger batches train faster but use more GPU memory.

**Embedding**
A dense vector representation of data (audio, text, images) in a high-dimensional space where similar items are close together.

**Epoch**
One complete pass through the entire training dataset. With streaming data, we use "steps" instead.

**Frozen (Model)**
A model whose weights don't update during training. We freeze the encoder and decoder, only training the projector.

**Gradient**
The direction and magnitude of change needed to reduce the loss. Computed via backpropagation.

**Gradient Norm**
The size of all gradients combined. High values (>100) indicate instability; low values (<10) indicate stable training.

**Inference**
Using a trained model to make predictions, as opposed to training it.

**Learning Rate**
How large each update step is. Too high = unstable; too low = slow convergence.

**Loss**
A number measuring how wrong the model's predictions are. Training tries to minimize this.

**Overfitting**
When a model memorizes training data instead of learning general patterns. Detected when eval loss rises while training loss falls.

**Step**
One update of the model weights after processing a batch of data.

______________________________________________________________________

## Audio Processing

**Log-Mel Spectrogram**
A visual representation of audio showing frequency content over time, with frequency bands scaled to match human hearing (mel scale).

**Sample Rate**
How many audio measurements per second. 16kHz means 16,000 samples per second.

**SpecAugment**
A data augmentation technique that randomly masks portions of the spectrogram (time and frequency bands) during training to improve robustness.

**Waveform**
The raw audio signal showing amplitude (loudness) over time.

______________________________________________________________________

## ASR-Specific Terms

**ASR (Automatic Speech Recognition)**
Converting spoken audio into written text.

**Decoder**
The language model that generates text from projector embeddings. In our case, Qwen3-0.6B.

**Encoder**
The audio model that converts spectrograms to embeddings. In our case, GLM-ASR-Nano-2512.

**Frame Stacking**
A technique that combines adjacent audio frames to reduce sequence length while preserving information. Formula: `output_len = (input_len - k) // k + 1` where k is the stride.

**Homophones**
Words that sound identical but have different spellings: "to/too/two", "there/their/they're".

**Projector**
The trainable module that bridges audio embeddings to text embeddings. The only component we train.

**WER (Word Error Rate)**
The standard metric for ASR accuracy. Lower is better. Calculated as: (Substitutions + Insertions + Deletions) / Total Reference Words.

______________________________________________________________________

## Model Architecture

**Attention**
A mechanism that lets models focus on relevant parts of the input. Used in both encoder and decoder.

**Chat Template**
A formatting system for structuring conversations with language models. Tiny Audio uses the format: `"Transcribe: " + audio_tokens` as the user message.

**GLM-ASR**
The audio encoder model from Zhipu AI. Converts spectrograms to embeddings. We use GLM-ASR-Nano-2512.

**LoRA (Low-Rank Adaptation)**
A parameter-efficient fine-tuning technique that adds small trainable matrices to frozen model layers. Used in Stage 2/3 training to lightly fine-tune the language model.

**MLP (Multi-Layer Perceptron)**
A simple neural network with stacked linear layers and activation functions.

**MoE (Mixture of Experts)**
An architecture where multiple "expert" networks specialize in different inputs, with a router selecting which experts to use.

**QFormer**
A query-based transformer that uses learnable queries to compress and project sequences.

**Qwen3**
The language model from Alibaba used as the decoder. We use Qwen3-0.6B.

**Transformer**
The dominant architecture in modern AI, using self-attention to process sequences.

______________________________________________________________________

## Projector Types

**MLP Projector**
The simplest projector: frame stacking followed by two linear layers. Fast to train, good baseline. (~12M parameters)

**MOSA Projector**
Dense mixture of experts with frame stacking. All experts contribute to every prediction via softmax routing. Uses LoRA by default.

**MoE Projector**
A shared expert plus sparse routed experts (top-k). Balance between MLP simplicity and MoE capacity. Uses LoRA by default.

**QFormer Projector**
Uses learnable query tokens and cross-attention to compress audio sequences. Based on BLIP-2.

______________________________________________________________________

## Tools and Frameworks

**Gradio**
A Python library for creating web demos of ML models.

**Hugging Face Hub**
A platform for sharing models, datasets, and demos.

**Hydra**
A configuration framework that lets you override YAML config values from the command line. Use `key=value` syntax (not `--key value`).

**Poetry**
A Python dependency manager and packaging tool.

**RunPod**
A cloud GPU provider for running training and inference.

**tmux**
A terminal multiplexer that lets sessions persist after disconnecting.

**W&B (Weights & Biases)**
A platform for tracking ML experiments, visualizing metrics, and comparing runs.

______________________________________________________________________

## Training Infrastructure

**Checkpoint**
A saved snapshot of model weights during training. Used for resuming and evaluation.

**CUDA**
NVIDIA's parallel computing platform for GPU acceleration.

**Flash Attention**
An optimized attention implementation that reduces memory usage and increases speed.

**MPS (Metal Performance Shaders)**
Apple's GPU acceleration framework for M1/M2/M3 chips.

**VRAM**
Video RAM—the GPU's memory. Limits batch size and model size.

______________________________________________________________________

## Multi-Stage Training

**Stage 1**
Initial training where only the projector learns. Config: `+experiments=transcription`

**Stage 2**
Fine-tuning where LoRA adapters are trained while the projector is frozen. Config: `+experiments=mlp_lora`

**Stage 3**
Joint fine-tuning where both projector and LoRA adapters train together. Config: `+experiments=mlp_fine_tune`

______________________________________________________________________

## The "Cliff" Phenomenon

A common pattern in ASR training where:
- Steps 0-1500: Model outputs gibberish, loss is high
- Steps 1500-1600: Loss suddenly drops ("the cliff")
- Steps 1600+: Gradual improvement

This is normal behavior. Don't panic if the model seems broken for the first hour.

______________________________________________________________________

[← Quick Reference](./4-quick-reference.md) | [Course Overview](./0-course-overview.md)
