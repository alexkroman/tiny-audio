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

**Waveform**
The raw audio signal showing amplitude (loudness) over time.

______________________________________________________________________

## ASR-Specific Terms

**ASR (Automatic Speech Recognition)**
Converting spoken audio into written text.

**Decoder**
The language model that generates text from projector embeddings. In our case, SmolLM3-3B.

**Encoder**
The audio model that converts spectrograms to embeddings. In our case, Whisper.

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

**MLP (Multi-Layer Perceptron)**
A simple neural network with stacked linear layers and activation functions.

**MoE (Mixture of Experts)**
An architecture where multiple "expert" networks specialize in different inputs, with a router selecting which experts to use.

**QFormer**
A query-based transformer that uses learnable queries to compress and project sequences.

**Transformer**
The dominant architecture in modern AI, using self-attention to process sequences.

______________________________________________________________________

## Tools and Frameworks

**Gradio**
A Python library for creating web demos of ML models.

**Hugging Face Hub**
A platform for sharing models, datasets, and demos.

**Hydra**
A configuration framework that lets you override YAML config values from the command line.

**Poetry**
A Python dependency manager and packaging tool.

**RunPod**
A cloud GPU provider for running training and inference.

**tmux**
A terminal multiplexer that lets sessions persist after disconnecting.

**W&B (Weights & Biases)**
A platform for tracking ML experiments, visualizing metrics, and comparing runs.

______________________________________________________________________

## Projector Types

**MLP Projector**
The simplest projector: frame stacking followed by two linear layers. Fast to train, good baseline.

**MOSA Projector**
Dense mixture of experts with convolutional downsampling. All experts contribute to every prediction.

**Shared MoE Projector**
A shared expert plus sparse routed experts. Balance between MLP simplicity and MoE capacity.

**QFormer Projector**
Uses learnable query tokens and cross-attention to compress audio sequences. Based on BLIP-2.

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

[← Quick Reference](./4-quick-reference.md) | [Course Overview](./0-course-overview.md)
