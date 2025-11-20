# Build Your Own Speech Recognition Model

## Course Overview

Welcome to a hands-on journey into the heart of modern AI. This course isn't just about learning how speech recognition works; it's about building, training, and deploying your own powerful ASR model from scratch. By the end of this course, you will have a real, working model with your name on it, and you'll understand exactly how it works.

By the end of this course, you will:

- Understand the architecture of modern multimodal ASR systems (encoder-projector-decoder)

- Train your own model efficiently by keeping the large models frozen and only training the projector

- Work with ~13M trainable parameters (just the projector) instead of 9.3+ billion

- Publish your model to HuggingFace Hub

- Add your results to the community leaderboard

**Time Commitment**: 6 hours (6 one-hour sessions)
**Cost**: ~$12 for GPU training (or free with local GPU), plus optional deployment costs
**Prerequisites**: Basic Python knowledge, some ML familiarity helpful but not required
**Model Architecture**: Whisper encoder + Linear projector + SmolLM3 decoder

**Format**: Can be used for:

- 📚 **Self-paced learning** - Work through materials on your own

- 👥 **Study group** - Learn with friends

- 🎓 **Workshop/course** - Great for teaching ASR and ML engineering

## Course Structure

### [Class 1: Introduction and Setup](./1-introduction-and-setup.md) (1 hour)

- What is automatic speech recognition?

- Understanding the Tiny Audio architecture

- Setting up your development environment

- Running your first inference

- **Experiments**: Test different audio samples, explore model outputs, adjust confidence thresholds

### [Class 2: Audio Processing and Encoders](./2-audio-processing-and-encoders.md) (1 hour)

- How audio becomes data

- Feature extraction with Wav2Vec2

- Understanding the HuBERT encoder

- Exploring audio embeddings

- **Experiments**: Compare Wav2Vec2 vs HuBERT, test different audio preprocessing, visualize embeddings

### [Class 3: Language Models and Projectors](./3-language-models-and-projectors.md) (1 hour)

- What are language models?

- The SmolLM3 decoder

- Bridging audio and text: the AudioProjector

- Linear projector architecture explained

- **Experiments**: Try different decoder models, modify projection dimensions, test activation functions

### [Class 4: Training](./4-training.md) (1 hour)

- Why parameter-efficient training?

- Understanding the projector-only training approach

- Configuring training with Hydra

- Starting your first training run

- **Experiments**: Test learning rates, experiment with batch sizes, try different datasets

### [Class 5: Evaluation and Debugging](./5-evaluation-and-debugging.md) (1 hour)

- Understanding Word Error Rate (WER)

- Evaluating your model

- Debugging common issues

- Improving model performance

- **Experiments**: Compare metrics across datasets, test data augmentation, analyze error patterns

### [Class 6: Publishing and Deployment](./6-publishing-and-deployment.md) (1 hour)

- Pushing to HuggingFace Hub

- Creating a professional model card

- Deploying to HuggingFace Inference Endpoints (production APIs)

- Creating interactive demos on HuggingFace Spaces

- Testing with multiple interfaces (transformers, API, web UI)

- Adding your results to the leaderboard

- **Experiments**: Test different deployment configurations, benchmark inference speeds, compare serving options

## Learning Goals

By completing this course, you will be able to:

1. **Explain** how multimodal ASR systems work end-to-end
1. **Implement** custom audio-language model architectures
1. **Apply** efficient training techniques with frozen encoders/decoders
1. **Train** a speech recognition model on real datasets
1. **Evaluate** model performance using industry-standard metrics
1. **Deploy** models to production-ready environments
1. **Publish** models and contribute to open-source ML communities

## Prerequisites

### Required Knowledge

- Python programming basics

- Command line/terminal usage

- Git basics

### Helpful But Not Required

- PyTorch fundamentals

- Transformers library experience

- Machine learning concepts (embeddings, attention, etc.)

Don't worry if you're missing some prerequisites! The course is designed to teach you what you need as you go.

## Hardware Requirements

### For Training (Class 4-5)

- **Cloud GPU**: NVIDIA A40 40GB (~$0.50/hour, ~$12 total) - Recommended

- **Local GPU**: NVIDIA RTX 3090/4090 24GB or better

- **Apple Silicon**: M1/M2/M3 Max/Ultra with 32GB+ RAM (slower but works)

### For Development (Class 1-3, 6)

- Any modern laptop

- 8GB RAM minimum

- 20GB free disk space

## Course Materials

Each class includes:

- **Lecture (20 min)**: Core concepts and theory presented by instructor

- **Workshop (40 min)**: Hands-on exercises following step-by-step instructions

- **Summary**: Key takeaways and homework

- **Self-Check Questions**: Verify your understanding

- **Further Reading**: Resources to deepen your knowledge

## Support and Community

- **GitHub Issues**: Report bugs or ask questions

- **Discussions**: Share your results and learnings

- **Leaderboard**: See how your model compares to others

## License

This course and all materials are released under the MIT License.

______________________________________________________________________

[Next: Class 1: Introduction and Setup](./1-introduction-and-setup.md)
