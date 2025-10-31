# Build Your Own Speech Recognition Model

## Course Overview

Welcome to a hands-on journey into the heart of modern AI. This course isn't just about learning how speech recognition works; it's about building, training, and deploying your own powerful ASR model from scratch. By the end of this course, you will have a real, working model with your name on it, and you'll understand exactly how it works.

By the end of this course, you will:

- Understand the architecture of modern multimodal ASR systems
- Train your own model using parameter-efficient techniques (LoRA)
- Publish your model to HuggingFace Hub
- Add your results to the community leaderboard

**Time Commitment**: 6 hours (6 one-hour sessions)
**Cost**: ~$12 for GPU training (or free with local GPU), plus optional deployment costs
**Prerequisites**: Basic Python knowledge, some ML familiarity helpful but not required

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

### [Class 2: Audio Processing and Encoders](./2-audio-processing-and-encoders.md) (1 hour)

- How audio becomes data
- Feature extraction with Wav2Vec2
- Understanding the HuBERT encoder
- Exploring audio embeddings

### [Class 3: Language Models and Projectors](./3-language-models-and-projectors.md) (1 hour)

- What are language models?
- The Qwen-3 8B decoder
- Bridging audio and text: the AudioProjector
- SwiGLU architecture explained

### [Class 4: Training](./4-training.md) (1 hour)

- Why parameter-efficient training?
- Understanding LoRA (Low-Rank Adaptation)
- Configuring training with Hydra
- Starting your first training run

### [Class 5: Evaluation and Debugging](./5-evaluation-and-debugging.md) (1 hour)

- Understanding Word Error Rate (WER)
- Evaluating your model
- Debugging common issues
- Improving model performance

### [Class 6: Publishing and Deployment](./6-publishing-and-deployment.md) (1 hour)

- Pushing to HuggingFace Hub
- Creating a professional model card
- Deploying to HuggingFace Inference Endpoints (production APIs)
- Creating interactive demos on HuggingFace Spaces
- Testing with multiple interfaces (transformers, API, web UI)
- Adding your results to the leaderboard

## Learning Goals

By completing this course, you will be able to:

1. **Explain** how multimodal ASR systems work end-to-end
2. **Implement** custom audio-language model architectures
3. **Apply** parameter-efficient training techniques like LoRA
4. **Train** a speech recognition model on real datasets
5. **Evaluate** model performance using industry-standard metrics
6. **Deploy** models to production-ready environments
7. **Publish** models and contribute to open-source ML communities

## Course Philosophy

This course emphasizes:

- **Hands-on learning**: You'll write and modify real code
- **Understanding over memorization**: We explain the "why" behind every concept
- **Minimal complexity**: ~1200 lines of hackable code
- **Real results**: Your model will actually work and be deployable
- **Community contribution**: Share your results and learn from others

### The Training Compass: Why → What → How

Before we dive into the technical details, it's important to think strategically about our project. A valuable framework for this is the "Training Compass":

1.  **Why are we building this?** What is our goal? Are we trying to achieve state-of-the-art performance, build a model for a specific niche, or simply learn?
2.  **What should we build?** Based on our "why," what kind of model should we build? What are the architectural choices and data considerations?
3.  **How will we build it?** What are the practical steps, tools, and techniques we'll use to train, evaluate, and deploy our model?

Throughout this course, we'll return to this compass to guide our decisions.

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

## Getting Started

**For the fastest path to a working environment, see the [5-Minute Quick Start Guide](../../README.md#quick-start) in the main README.**

1. **Clone the repository**

   ```bash
   git clone https://github.com/alexkroman/tiny-audio.git
   cd tiny-audio
   ```

2. **Install dependencies**

   ```bash
   poetry install
   ```

3. **Download sample audio files**

   ```bash
   poetry run download-samples
   ```

4. **Start with Class 1**
   - [Class 1: Introduction and Setup](./1-introduction-and-setup.md)

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

---

[Next: Class 1: Introduction and Setup](./1-introduction-and-setup.md)
