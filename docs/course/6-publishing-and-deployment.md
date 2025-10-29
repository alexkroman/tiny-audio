# Class 6: Publishing and Deployment

**Duration**: 1 hour (20 min lecture + 40 min hands-on)
**Goal**: Deploy your model and add your results to the community leaderboard

## Learning Objectives

By the end of this class, you will:

- Push your model to HuggingFace Hub
- Create a professional model card
- Test your deployed model via the transformers pipeline
- Add your results to the community leaderboard
- Have a publicly accessible, working ASR model!

---

# PART A: LECTURE (20 minutes)

> **Instructor**: Present these concepts. Students should just listen.

## 1. HuggingFace Hub Overview (5 min)

### What is HuggingFace Hub?

**The GitHub for Machine Learning Models**

- Host models, datasets, and demos
- Version control for ML artifacts
- Easy sharing and collaboration
- 500K+ models hosted
- Industry standard for ML deployment

### Why Host on Hub?

**Benefits:**

- **Free hosting**: No cost for public models
- **Easy deployment**: One line of code to use
- **Version control**: Track model iterations
- **Community visibility**: Others can find and use your model
- **Portfolio piece**: Show off your work!

### Model Repository Structure

```
your-username/your-model-name/
├── config.json                  # Model architecture config
├── model.safetensors           # Trained weights (projector + LoRA)
├── tokenizer_config.json       # Text tokenizer settings
├── preprocessor_config.json    # Audio preprocessing settings
├── README.md                   # Model card (documentation)
├── encoder_lora_config.json    # Encoder LoRA config
├── decoder_lora_config.json    # Decoder LoRA config
└── asr_*.py                    # Model code files
```

### Authentication

You need a **HuggingFace token** to push models:

1. Create account at [huggingface.co](https://huggingface.co)
2. Go to Settings → Access Tokens
3. Create token with "write" permissions
4. Keep it secret!

---

## 2. Writing a Model Card (5 min)

### What is a Model Card?

**Documentation for your model**

- What it does
- How it was trained
- Performance metrics
- Usage examples
- Limitations and biases

**Why important?**

- Helps others understand and use your model
- Professional presentation
- Reproducibility
- Ethical AI practice

### Model Card Structure

**Essential sections:**

1. **Model Description**
   - What the model does
   - Architecture summary
   - Key features

2. **Training Details**
   - Dataset used
   - Training time
   - Hardware used
   - Cost

3. **Performance**
   - WER score
   - Test set used
   - Comparison with baselines

4. **Usage**
   - Code examples
   - How to install dependencies
   - Expected input/output

5. **Limitations**
   - What doesn't work well
   - Known issues
   - Ethical considerations

---

## 3. Testing and Validation (5 min)

### Pre-Deployment Checklist

Before pushing to Hub:

- [ ] Training completed successfully
- [ ] Model evaluation done (WER calculated)
- [ ] Tested inference locally
- [ ] Model card prepared
- [ ] HuggingFace token ready

### Post-Deployment Testing

After pushing:

1. **Load from Hub**: Verify model downloads correctly
2. **Test inference**: Run transcription on sample audio
3. **Check model card**: Renders properly on Hub
4. **Test pipeline**: Use transformers pipeline integration

### Common Deployment Issues

**Problem**: Model fails to load

- Check all required files are uploaded
- Verify config.json is valid
- Ensure asr_*.py files are present

**Problem**: Inference errors

- Test locally first
- Check transformers version compatibility
- Verify audio format support

---

## 4. Community Contribution (5 min)

### The Leaderboard

**Location**: Main README.md in the repo

**Purpose**:

- Track community progress
- Healthy competition
- Share knowledge
- Build community

**What to include**:

- Your name/GitHub username
- WER score
- Link to your model
- Git commit hash
- Training date

### Beyond the Leaderboard

**Ways to contribute:**

- Share training tips
- Report issues
- Improve documentation
- Help other students
- Experiment with architecture changes

---

# PART B: HANDS-ON WORKSHOP (40 minutes)

> **Students**: Follow these instructions step-by-step.
>
> **Instructor**: Circulate and help students.

---

## Workshop Exercise 1: Prepare for Publishing (10 min)

### Goal

Set up HuggingFace account and prepare model files.

### Your Task

Create account, get token, and verify model is ready.

### Instructions

**Step 1: Create HuggingFace account**

- Go to [huggingface.co/join](https://huggingface.co/join)
- Sign up (free)
- Verify your email
- Choose a good username (e.g., your name or GitHub handle)

**Step 2: Get access token**

- Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Click "New token"
- Name it: "tiny-audio-upload"
- Select role: "Write"
- Click "Generate"
- **Copy token** - you'll need it soon!

**Step 3: Install huggingface_hub**

```bash
poetry add huggingface_hub
```

**Step 4: Login to HuggingFace**

```bash
poetry run huggingface-cli login
```

Paste your token when prompted.

**Step 5: Verify model files**

```bash
ls outputs/stage1/  # or your training directory
```

Make sure you have:

- [ ] config.json
- [ ] model.safetensors
- [ ] tokenizer files
- [ ] asr_*.py files

### Success Checkpoint

- [ ] HuggingFace account created
- [ ] Access token obtained
- [ ] Logged in via CLI
- [ ] Model files verified

---

## Workshop Exercise 2: Push Model to Hub (15 min)

### Goal

Upload your trained model to HuggingFace Hub.

### Your Task

Create repository and push all model files.

### Instructions

**Step 1: Choose model name**

Pick a good name for your model:

- Descriptive: "tiny-audio-my-name"
- No spaces: Use hyphens
- Lowercase recommended
- Your full model ID: "your-username/your-model-name"

**Step 2: Create push script**

Create `push_to_hub.py`:

```python
import argparse
from src.asr_modeling import ASRModel
from src.asr_config import ASRConfig

# --- Argument Parser ---
def main():
    parser = argparse.ArgumentParser(description="Push a Tiny Audio model to HuggingFace Hub.")
    parser.add_argument("model_path", type=str, help="Path to the local model directory.")
    parser.add_argument("hub_model_id", type=str, help="The desired model ID on the Hub (e.g., your-username/your-model-name).")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    config = ASRConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = ASRModel.from_pretrained(args.model_path, config=config)
    print("✓ Model loaded")

    print(f"\nPushing to HuggingFace Hub: {args.hub_model_id}")
    print("This may take a few minutes...")

    # Push to Hub
    model.save_pretrained(
        args.hub_model_id,
        push_to_hub=True,
        commit_message="Initial model upload from Tiny Audio course"
    )

    print(f"\n✓ Model successfully pushed to: https://huggingface.co/{args.hub_model_id}")
    print("\nNext steps:")
    print("1. Visit your model page")
    print("2. Edit the README.md (model card)")
    print("3. Test the model via the transformers library")

if __name__ == "__main__":
    main()
```

**Step 3: Push to Hub**

```bash
poetry run python push_to_hub.py outputs/stage1 your-username/your-model-name
```

**What happens:**

1. Loads your trained model
2. Creates repository on Hub (if doesn't exist)
3. Uploads all necessary files:
   - Model weights
   - Configs
   - Tokenizer files
   - Code files (asr_*.py)
4. Creates initial README

**Step 5: Verify upload**

Visit: `https://huggingface.co/your-username/your-model-name`

You should see:

- Model files listed
- Auto-generated README
- Model card (needs editing!)

### Success Checkpoint

- [ ] Script ran successfully
- [ ] Model uploaded to Hub
- [ ] Can see model page on HuggingFace
- [ ] All files present

---

## Workshop Exercise 3: Create Model Card (10 min)

### Goal

Write professional documentation for your model.

### Your Task

Edit the README.md on HuggingFace to create a model card.

### Instructions

**Step 1: Go to your model page**

Navigate to: `https://huggingface.co/your-username/your-model-name`

**Step 2: Click "Edit model card"**

**Step 3: Replace content with your model card**

Use this template (customize with your info):

```markdown
---
language: en
license: mit
tags:
- audio
- speech
- speech-recognition
- automatic-speech-recognition
- tiny-audio
datasets:
- speechbrain/LoquaciousSet
metrics:
- wer
pipeline_tag: automatic-speech-recognition
---

# Tiny Audio - [Your Name]

## Model Description

This is a speech recognition model trained using the Tiny Audio framework. It combines:
- **Audio Encoder**: HuBERT-XLarge (1.3B params) with LoRA adapters (r=8, ~2M trainable)
- **Audio Projector**: SwiGLU MLP (~122M params, fully trainable)
- **Text Decoder**: SmolLM3-3B (3B params) with LoRA adapters (r=64, ~15M trainable)

**Total**: 139M trainable parameters out of 4.3B total (3.2%)

## Training Details

- **Dataset**: LoquaciousSet (25,000 hours of diverse speech)
- **Training Time**: ~24 hours
- **Hardware**: NVIDIA A40 40GB
- **Cost**: ~$12
- **Framework**: Tiny Audio course project

## Performance

**Word Error Rate (WER)**: **XX.XX%** on LoquaciousSet test set

| Metric | Value |
|--------|-------|
| WER | XX.XX% |
| Test Samples | 500 |
| Dataset | LoquaciousSet |

## Usage

```python
from transformers import pipeline

# Load the model
pipe = pipeline(
    "automatic-speech-recognition",
    model="your-username/your-model-name",
    trust_remote_code=True
)

# Transcribe audio
result = pipe("path/to/audio.wav")
print(result["text"])
```

## Limitations

- Trained primarily on English speech
- Performance may degrade with heavy background noise
- Accented speech may have higher error rates
- Best suited for conversational speech

## Citation

Trained as part of the Tiny Audio course:

```bibtex
@software{your-name-2025-tiny-audio,
  author = {Your Name},
  title = {Tiny Audio ASR Model},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/your-username/your-model-name}
}
```

## Acknowledgments

Built with [Tiny Audio](https://github.com/alexkroman/tiny-audio) by Alex Kroman.

Based on:

- HuBERT (Facebook AI)
- SmolLM3 (HuggingFace)
- LoquaciousSet (SpeechBrain)

```

**Step 4: Customize your model card**

Update these sections:
- Replace `[Your Name]` with your actual name
- Fill in your actual WER score (XX.XX%)
- Update the model ID in code examples
- Add any custom training details
- Mention any experiments you tried

**Step 5: Save and commit**

Click "Commit changes to main"

### Success Checkpoint

- [ ] Model card edited
- [ ] WER score added
- [ ] Usage example works
- [ ] Professional presentation

---

## Workshop Exercise 4: Test and Add to Leaderboard (5 min)

### Goal
Test your deployed model and add results to the community leaderboard.

### Your Task
Verify model works and submit PR with your results.

### Instructions

**Step 1: Test deployed model**

Create `test_deployed.py`:

```python
import argparse
from transformers import pipeline

# --- Argument Parser ---
def main():
    parser = argparse.ArgumentParser(description="Test a deployed Tiny Audio model from HuggingFace Hub.")
    parser.add_argument("model_id", type=str, help="The model ID on the Hub (e.g., your-username/your-model-name).")
    parser.add_argument("audio_path", type=str, help="Path to the audio file to transcribe.")
    args = parser.parse_args()

    # Load YOUR model from Hub
    print("Loading model from HuggingFace Hub...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model_id,
        trust_remote_code=True
    )
    print("✓ Model loaded\n")

    # Test on sample audio
    print(f"Transcribing {args.audio_path}...")
    result = pipe(args.audio_path)

    print("\nResult:")
    print("="*50)
    print(result["text"])
    print("="*50)
    print("\n✓ Your model is working and publicly accessible!")

if __name__ == "__main__":
    main()
```

**Step 2: Run test**

```bash
poetry run python test_deployed.py your-username/your-model-name test.wav
```

**Step 3: Fork the Tiny Audio repository**

- Go to [github.com/alexkroman/tiny-audio](https://github.com/alexkroman/tiny-audio)
- Click "Fork" (top right)
- Clone your fork locally

**Step 4: Add your entry to the leaderboard**

Edit `README.md` in your fork, find the Leaderboard section:

```markdown
| Rank | Contributor | WER | Model | Date |
|------|------------|-----|-------|------|
| 🥇 | [@alexkroman](https://github.com/alexkroman) | **12.14** | [mazesmazes/tiny-audio](https://huggingface.co/mazesmazes/tiny-audio) | 2025-10-23 |
| 🥈 | [@your-username](https://github.com/your-username) | **XX.XX** | [your-username/your-model-name](https://huggingface.co/your-username/your-model-name) | 2025-XX-XX |
```

Add your row with:

- Your GitHub username
- Your WER score
- Link to your HuggingFace model
- Today's date

**Step 5: Create Pull Request**

```bash
git add README.md
git commit -m "Add my model to leaderboard: XX.XX% WER"
git push origin main
```

Then on GitHub:

- Go to your fork
- Click "Contribute" → "Open pull request"
- Title: "Add [your-username] to leaderboard"
- Submit!

### Success Checkpoint

- [ ] Model tested from Hub
- [ ] Inference works correctly
- [ ] Added to leaderboard
- [ ] Pull request submitted

---

# CLASS SUMMARY & COURSE COMPLETION

## What We Covered Today

**Lecture (20 min):**

- HuggingFace Hub overview
- Model card best practices
- Deployment testing
- Community contribution

**Workshop (40 min):**

- Set up HuggingFace account
- Pushed model to Hub
- Created professional model card
- Tested deployed model
- Added to community leaderboard

## Course Journey Complete! 🎉

### What You've Accomplished

Over 6 classes, you've:

✅ **Class 1**: Set up environment and ran first inference
✅ **Class 2**: Understood audio processing and encoders
✅ **Class 3**: Learned about projectors and language models
✅ **Class 4**: Configured and ran training with LoRA
✅ **Class 5**: Evaluated model and analyzed errors
✅ **Class 6**: Deployed model and joined the community

**You now have:**

- A working ASR model trained by you
- Publicly deployed on HuggingFace Hub
- Professional documentation
- Evaluation metrics
- Community recognition
- Real ML engineering experience!

## Key Takeaways from Entire Course

🎯 **Technical Skills:**

- Parameter-efficient fine-tuning with LoRA
- Multimodal architecture (audio + text)
- Training pipeline configuration
- Model evaluation and debugging
- ML model deployment

🎯 **Practical Skills:**

- Cloud GPU management
- Git/GitHub workflow
- Technical writing (model cards)
- Community contribution
- End-to-end ML project

## Your Model Stats

**Fill in your achievements:**

- Model name: ______________________________
- HuggingFace URL: _________________________
- WER Score: ________%
- Training cost: $______
- Leaderboard rank: _______

## Next Steps

### Continue Improving

**Experiment with:**

- Different LoRA ranks
- Various learning rates
- More training data
- Different encoder/decoder models
- Data augmentation
- Domain-specific fine-tuning

### Share Your Work

- Tweet about your model
- Write a blog post
- Help other students
- Contribute to the codebase
- Present at meetups/conferences

### Advanced Projects

**Ideas:**

- Multi-language support
- Real-time streaming ASR
- Speaker diarization
- Emotion detection
- Custom vocabulary/domain adaptation
- Mobile deployment

## Resources for Continued Learning

### Papers to Read

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Whisper: Robust Speech Recognition](https://arxiv.org/abs/2212.04356)

### Communities

- HuggingFace Forums
- Reddit: r/MachineLearning
- Discord: Tiny Audio community
- Twitter: Follow #TinyAudio

### Tools to Explore

- [PEFT library](https://github.com/huggingface/peft)
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- [Unsloth](https://github.com/unslothai/unsloth)

## Final Check Your Understanding

1. **What's the full training pipeline?**
   - Data → Encoder → Projector → Decoder → Loss
   - LoRA adapters on encoder and decoder
   - Projector fully trained

2. **Why is Tiny Audio "tiny"?**
   - Only 3.2% of parameters trainable
   - Fast, cheap, accessible training
   - Yet produces working models!

3. **What makes a model "production-ready"?**
   - Good evaluation metrics
   - Professional documentation
   - Easy to use (one-line API)
   - Publicly accessible

4. **How do you improve a model?**
   - More/better data
   - Longer training
   - Hyperparameter tuning
   - Architecture changes
   - Error analysis → targeted fixes

---

## Congratulations! 🎊

You've completed the Tiny Audio course and joined the community of ML practitioners who have trained and deployed their own speech recognition models.

**Your model is now:**

- ✅ Publicly accessible
- ✅ Documented
- ✅ Evaluated
- ✅ Part of the community

**You are now:**

- ✅ An ML engineer
- ✅ A contributor to open source
- ✅ Part of the Tiny Audio community
- ✅ Ready for advanced projects

## Thank You

Thank you for taking this course. We can't wait to see what you build next!

**Questions or want to share your success?**

- Open an issue on GitHub
- Join the community discussions
- Tag @alexkroman on Twitter
- Share your model card!

**Keep building, keep learning, keep shipping! 🚀**

---

## Certificate of Completion

**This certifies that**

**_______________________________**
*(your name)*

**has successfully completed the Tiny Audio course**

- Trained a speech recognition model
- Achieved **_____**% WER
- Published to HuggingFace Hub
- Contributed to the community

**Date**: _____________

**Model**: [your-username/your-model-name](https://huggingface.co/your-username/your-model-name)

---

[Previous: Class 5: Evaluation and Debugging](./5-evaluation-and-debugging.md)

*Course materials available at: <https://github.com/alexkroman/tiny-audio>*
