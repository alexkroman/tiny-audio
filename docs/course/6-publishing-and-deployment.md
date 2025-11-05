# Class 6: Publishing and Deployment

**Duration**: 1 hour (15 min lecture + 45 min hands-on)

**Goal**: Deploy your model to HuggingFace Hub with production endpoints and web demo

## Learning Objectives

By the end of this class, you will:

- Push your model to HuggingFace Hub

- Create a professional model card

- Deploy your model to a production inference endpoint

- Create an interactive web demo on HuggingFace Spaces

- Test your deployed model via multiple interfaces

- Add your results to the community leaderboard

- Have a fully deployed, publicly accessible ASR system!

---

# PART A: LECTURE (15 minutes)

## 1. Why Share Your Work? (3 min)

Before we get into the technical details of deployment, let's talk about why it's so important to share your work. The open-source community is the engine that drives progress in machine learning. By sharing your model, you are not just publishing a file; you are contributing to a global collaboration.

**Benefits of Sharing:**

- **Community Contribution**: You are giving back to the community that created the tools and models you used.

- **Feedback and Improvement**: Others can test your model, find its weaknesses, and suggest improvements.

- **Build Your Portfolio**: A public model on the Hugging Face Hub is a powerful demonstration of your skills as an ML engineer.

- **Advance the Field**: Every shared model, no matter how small, adds to our collective understanding of what works and what doesn't.

In this chapter, you'll learn how to share your work in a way that is professional, responsible, and valuable to the community.

---

## 2. HuggingFace Hub & Deployment Options (5 min)


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


### Deployment Options Overview

Once your model is on the Hub, you have **three deployment options**:

1. **Direct usage via transformers** - Users download and run locally (free)
2. **Inference Endpoints** - Serverless API for production (pay-per-use, scales to zero)
3. **Spaces (Gradio demo)** - Interactive web UI for demos (free tier available)

**Quick Experiment**: Compare deployment options:

```python
# Compare deployment characteristics
options = {
    "Local": {"cost": 0, "latency": "low", "scale": "limited"},
    "Endpoints": {"cost": "$$$", "latency": "medium", "scale": "unlimited"},
    "Spaces": {"cost": "$", "latency": "high", "scale": "moderate"},
}

for name, specs in options.items():
    print(f"{name}: Cost={specs['cost']}, Latency={specs['latency']}, Scale={specs['scale']}")


```

**Authentication**

You need a **HuggingFace token** with "write" permissions to push models and create deployments.

---

## 3. Writing a Model Card: Your Contribution to the Community (3 min)


### What is a Model Card?

A model card is more than just documentation; it's your primary contribution to the community. It's where you explain not just *what* your model does, but *how* it was built, *why* you made certain decisions, and *what* you learned along the way. A good model card makes your work useful and accessible to others.


### Why it Matters

- **Reproducibility**: A detailed model card allows others to understand and reproduce your work.

- **Transparency**: It's an ethical practice that promotes transparency about a model's capabilities and limitations.

- **Usability**: It provides clear instructions on how to use your model, making it more valuable to the community.


### Model Card Structure

As we'll see in the workshop, a good model card includes:

1.  **Model Description**: What is it, and what is its architecture?
2.  **Training Details**: What data did you use? What were your hyperparameters?
3.  **Performance**: How well does it work? What are its limitations?
4.  **Usage**: How can others use it?

---

## 4. Testing and Validation (3 min)


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

## 5. Community Contribution (4 min)


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


### Beyond the Leaderboard: Sharing Your Story

The story behind a model is often as valuable as the model itself. The failures, the dead ends, and the unexpected discoveries are all part of the learning process. We encourage you to share not just your final results, but also your journey.

**Ways to contribute:**

- **Share your training logs**: A public Weights & Biases report can be incredibly insightful for others.

- **Write a blog post**: Share what you learned, what went wrong, and what you'd do differently next time.

- **Contribute to the discussion**: Help other students, answer questions, and share your experiences in the GitHub Discussions.

- **Improve the course**: If you find a bug, a typo, or have an idea for an improvement, open an issue or a pull request!

By sharing your story, you help us all learn and grow together.

---

# PART B: HANDS-ON WORKSHOP (45 minutes)

>

## Workshop Overview

In the next 45 minutes, you will:

- **Exercise 1**: Setup and push model to Hub with experiments (10 min)

- **Exercise 2**: Create model card and test variations (8 min)

- **Exercise 3**: Deploy and experiment with configurations (12 min)

- **Exercise 4**: Test, benchmark, and optimize deployment (10 min)

- **Bonus**: Advanced deployment experiments (5 min)

By the end, you'll have:

- A publicly accessible ASR model with documentation

- Production deployment with performance benchmarks

- Your results on the community leaderboard

- Experience with deployment optimization

**Note**: We'll experiment with different deployment configurations throughout!

---

## Workshop Exercise 1: Setup and Push to Hub (10 min)


### Goal

Set up HuggingFace account and push your model to the Hub.


### Your Task

Create account, get token, verify model files, and upload to Hub.


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

**Step 5: Choose model name**

Pick a good name for your model:

- Descriptive: "tiny-audio-my-name"

- No spaces: Use hyphens

- Lowercase recommended

- Your full model ID: "your-username/your-model-name"

**Step 6: Create push script**

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

**Step 7: Push to Hub**


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

**Step 8: Verify upload**

Visit: `https://huggingface.co/your-username/your-model-name`

You should see:

- Model files listed

- Auto-generated README

- Model card (needs editing!)


### Success Checkpoint

- [ ] HuggingFace account created

- [ ] Access token obtained and logged in

- [ ] Model uploaded to Hub successfully

- [ ] Can see model page on HuggingFace

- [ ] All files present


### Upload Experiments

**Experiment 1: Test different upload strategies**


```python
# Compare upload methods
import time

methods = [
    {"name": "Direct push", "command": "model.push_to_hub('repo-id')"},
    {"name": "Save then push", "command": "model.save_pretrained('local'); push_to_hub('local', 'repo-id')"},
    {"name": "Git LFS", "command": "git lfs track '*.safetensors'; git push"},
]

for method in methods:
    print(f"{method['name']}: {method['command']}")
    # Time the upload
    # Compare speeds


```

**Experiment 2: Model size optimization**


```python
# Check model sizes
import os

def get_size_mb(path):
    size = os.path.getsize(path)
    return size / (1024 * 1024)

files = ["model.safetensors", "config.json", "tokenizer.json"]
total_size = 0

for file in files:
    size = get_size_mb(f"outputs/stage1/{file}")
    print(f"{file}: {size:.2f} MB")
    total_size += size

print(f"Total: {total_size:.2f} MB")

# Compare with quantized version
# Test int8 quantization impact


```

---

## Workshop Exercise 2: Create Model Card (8 min)


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

- **Text Decoder**: Qwen-3 8B3-3B (3B params) with LoRA adapters (r=64, ~15M trainable)

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

- Qwen-3 8B3 (HuggingFace)

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

## Workshop Exercise 3: Deploy Demo to Hugging Face Spaces (12 min)

**Goal**: Create a public web demo for your model using Hugging Face Spaces.

**Instructions**

**Step 1: Create a new Space**

- Go to [huggingface.co/new-space](https://huggingface.co/new-space)

- **Owner**: Your username

- **Space name**: `tiny-audio-demo` (or your preferred name)

- **License**: MIT

- **Select SDK**: Gradio

- **Space hardware**: CPU basic (free) or CPU upgrade ($0.03/hour for faster)

- **Visibility**: Public

- Click "Create Space"

**Step 2: Prepare demo files**

Clone the demo files from the tiny-audio repository:


```bash
# Copy demo files to a temporary directory
mkdir ~/my-tiny-audio-demo
cp demo/app.py ~/my-tiny-audio-demo/
cp demo/requirements.txt ~/my-tiny-audio-demo/
cp demo/README.md ~/my-tiny-audio-demo/
cd ~/my-tiny-audio-demo


```

**Step 3: Customize for your model**

Edit `app.py` to use your model:


```python
# Find this line (around line 17):
def create_demo(model_path: str = "mazesmazes/tiny-audio"):

# Change to:
def create_demo(model_path: str = "your-username/your-model-name"):

# And update the main call (around line 34):
demo = create_demo("your-username/your-model-name")


```

Edit `README.md` to update:

- Model name and links

- Your username in tags and citations

- Description and features

**Step 4: Verify requirements.txt**

The file should only need:



```
gradio  # Gradio automatically handles model loading via HF Inference API


```

That's it! Gradio's `gr.load()` function automatically loads your model from the Hub.

**Step 5: Initialize git and push to Space**


```bash
# Initialize git repo
git init
git remote add origin https://huggingface.co/spaces/your-username/tiny-audio-demo

# Login if you haven't
git config user.email "you@example.com"
git config user.name "Your Name"

# Add files and commit
git add .
git commit -m "Initial demo deployment"

# Push to Space
git push -u origin main


```

**Step 6: Watch build process**

- Go to your Space: `https://huggingface.co/spaces/your-username/tiny-audio-demo`

- You'll see "Building..." status

- Wait 2-5 minutes for build to complete

- Once ready, status shows "Running"

**Step 7: Test your demo**

- Click on your Space URL

- Upload a test audio file or record using microphone

- Click "Submit" to transcribe

- Verify transcription appears correctly


### How It Works

Gradio's `gr.load()` automatically:

- Detects your model is an ASR model

- Creates the appropriate audio input UI

- Calls the HuggingFace Inference API with your model

- Returns transcription results

- All with just one line of code!


### Customization Ideas (Optional)

**Add examples:**


```python
examples = [
    ["demo/samples/sample1.wav"],
    ["demo/samples/sample2.wav"],
]

return gr.Interface(
    fn=lambda audio: transcribe_audio(audio, model_path),
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath"),
    outputs=gr.Textbox(label="Transcription"),
    examples=examples,  # Add this
    title="Tiny Audio - Speech Recognition",
    description="Upload audio or record to transcribe speech to text.",
)


```

**Add model info:**


```python
description = f"""
Upload an audio file or record directly to transcribe speech to text.

**Model**: [{model_path}](https://huggingface.co/{model_path})
**WER**: XX.XX% on LoquaciousSet test set
**Training**: Efficient fine-tuning with LoRA (3.2% trainable params)
"""


```

**Add microphone input:**


```python
inputs=gr.Audio(sources=["upload", "microphone"], type="filepath"),


```


### Success Checkpoint

- [ ] Space created on Hugging Face

- [ ] Demo is live and working

- [ ] Can transcribe audio through web interface

- [ ] Model automatically loaded via Gradio's HF Inference API


### Deployment Experiments

**Experiment 1: Benchmark inference speed**


```python
# benchmark_deployment.py
import time
import requests

def test_inference_speed(model_id, audio_file):
    times = []

    for i in range(5):
        start = time.time()
        # Call inference API
        result = pipe(audio_file)
        duration = time.time() - start
        times.append(duration)

    avg_time = sum(times) / len(times)
    audio_duration = get_audio_duration(audio_file)
    rtf = avg_time / audio_duration  # Real-time factor

    print(f"Average inference: {avg_time:.2f}s")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Real-time factor: {rtf:.2f}x")
    print(f"Can process: {1/rtf:.1f}x real-time")


```

**Experiment 2: Test different audio formats**


```python
# Test format support
formats = ["wav", "mp3", "flac", "ogg", "m4a"]
for fmt in formats:
    test_file = f"test.{fmt}"
    try:
        result = pipe(test_file)
        print(f"✓ {fmt}: Supported")
    except:
        print(f"✗ {fmt}: Not supported")


```

**Experiment 3: Compare deployment options**


```python
# Compare latencies
deployments = {
    "Local": test_local_inference,
    "Spaces": test_spaces_api,
    "Endpoints": test_inference_endpoint,
}

for name, test_func in deployments.items():
    latency = test_func(audio_file)
    print(f"{name}: {latency:.2f}s latency")


```

---

## Workshop Exercise 4: Test and Add to Leaderboard (10 min)


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

**Workshop (45 min):**

- Set up HuggingFace account and authentication

- Pushed model to Hub with all artifacts

- Created professional model card with documentation

- Deployed to either Inference Endpoints OR Spaces (student choice)

- Tested deployed model

- Added to community leaderboard

**Optional (after class):**

- Complete the deployment option you didn't choose

- Experiment with customizations

- Share your demo with the community

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

## Your Model Stats

**Fill in your achievements:**

- Model name: ______________________________

- HuggingFace URL: _________________________

- WER Score: ________%

- Training cost: $______

- Leaderboard rank: _______

## Next Steps


### Continue Improving

**Systematic Experimentation Plan:**

1. **Architecture Experiments:**
   - Test encoder swaps (Wav2Vec2 vs HuBERT vs Whisper encoder)
   - Try different decoder sizes (1B, 3B, 7B, 13B)
   - Experiment with projector architectures (MLP vs Transformer)

2. **Training Experiments:**
   - LoRA rank ablations (r=1 to r=256)
   - Learning rate schedules (cosine vs linear vs constant)
   - Batch size studies (effective batch 8 to 256)
   - Dataset mixing strategies

3. **Optimization Experiments:**
   - Quantization (int8, int4 with GPTQ/AWQ)
   - Distillation to smaller models
   - Pruning experiments
   - Flash Attention integration

4. **Domain Adaptation:**
   - Medical transcription
   - Legal proceedings
   - Podcast/meeting transcription
   - Multi-speaker scenarios

5. **Production Experiments:**
   - Streaming inference
   - Batch processing optimization
   - Multi-GPU serving
   - Edge deployment (mobile, browser)


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
