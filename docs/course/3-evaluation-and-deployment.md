# Class 3: Evaluation and Deployment

**Duration**: 1 hour (15 min lecture + 45 min hands-on)

**Goal**: Evaluate your model's performance, deploy a public demo, and add your results to the leaderboard.

## Learning Objectives

By the end of this class, you will:

- Understand Word Error Rate (WER) and how to interpret it
- Evaluate your model on multiple datasets
- Deploy a live demo to Hugging Face Spaces
- Set up Hugging Face Inference Endpoints (optional)
- Add your results to the community leaderboard

______________________________________________________________________

## PART A: LECTURE (15 min)

### 1. Understanding WER (5 min)

The industry-standard metric for ASR is **Word Error Rate (WER)**.

**Formula:**

```
WER = (Substitutions + Insertions + Deletions) / Total Reference Words
```

**Example:**

| | |
|---|---|
| Reference | "hello world" |
| Prediction | "hello there world" |
| Error | 1 insertion ("there") |
| WER | 1/2 = 50% |

**Interpretation:**

| WER | Quality |
|-----|---------|
| < 5% | Excellent (commercial systems) |
| 5-10% | Very good |
| 10-20% | Good (our target) |
| > 30% | Poor |

**Important**: WER is quantitative. Always do qualitative "vibe testing" too—run the demo and listen to outputs on different audio types.

### 2. Evaluation Datasets (5 min)

Different datasets test different capabilities. The training config (`combined.yaml`) uses 9 datasets with equal weights:

| Dataset | What it tests |
|---------|---------------|
| **LoquaciousSet** | General benchmark (synthetic diverse speech, leaderboard uses this) |
| **GigaSpeech** | Diverse sources (YouTube, podcasts, audiobooks) |
| **LibriSpeech Clean** | Read audiobooks (clear speech, standard benchmark) |
| **LibriSpeech Other** | Read audiobooks (more challenging acoustics) |
| **AMI** | Meetings (multi-speaker, conversational) |
| **Earnings22** | Financial domain (company names, financial terms, noisy audio) |
| **TEDLIUM** | TED talks (lectures, presentations) |
| **People's Speech** | Large-scale diverse speech (various domains) |
| **Common Voice** | Crowdsourced recordings (accents, varied quality) |

Testing on multiple datasets reveals where your model excels and struggles.

### 3. Deployment Options (5 min)

| Option | Cost | Use case |
|--------|------|----------|
| **HF Spaces** | Free | Interactive demos, portfolio |
| **HF Inference Endpoints** | Paid | Production APIs, autoscaling |
| **Self-hosted** | Varies | Full control, requires DevOps |

We'll set up a Space (free demo) and optionally an Inference Endpoint.

______________________________________________________________________

## PART B: HANDS-ON WORKSHOP (45 min)

## Exercise 1: Evaluate Your Model (15 min)

### Goal

Get quantitative performance metrics across multiple datasets.

### Instructions

**Step 1: Evaluate on LoquaciousSet (benchmark)**

This is the official benchmark for the leaderboard:

```bash
poetry run eval your-username/your-model --dataset loquacious --max-samples 500
```

**What this does:**

- Streams audio from Hugging Face (no huge downloads)
- Runs inference on each sample
- Normalizes text (Whisper normalizer: capitalization, numbers, contractions)
- Computes WER
- Saves results to `outputs/eval_loquacious_*/results.txt`

**Output:**

```
Sample 1: WER = 8.33%, Time = 1.23s
  Ref:  The quick brown fox jumps over the lazy dog
  Pred: The quick brown fox jumps over the lazy dog

Sample 2: WER = 15.00%, Time = 1.45s
  Ref:  She sells seashells by the seashore
  Pred: She sells sea shells by the sea shore
...
================================================================================
CHECKPOINT @ 100 samples:
  Corpus WER: 12.45%
  Avg Time/Sample: 1.35s
================================================================================
```

**Record your final Corpus WER!** You'll need it for the leaderboard.

**Step 2: Evaluate on Earnings22 (domain-specific)**

```bash
poetry run eval your-username/your-model --dataset earnings22 --max-samples 100
```

**Step 3: Evaluate on AMI (meetings)**

```bash
poetry run eval your-username/your-model --dataset ami --max-samples 100
```

**Step 4: Compare with AssemblyAI (optional)**

If you have an API key:

```bash
export ASSEMBLYAI_API_KEY='your_key'

poetry run eval --assemblyai --assemblyai-model slam_1 --dataset loquacious --max-samples 100
```

### Analyzing Results

Look at `outputs/eval_*/results.txt`. Ask:

- Which samples have high WER? Why?
- Are there patterns (accents, noise, specific words)?
- How does performance vary across datasets?

### Success Checkpoint

- [ ] Evaluated on LoquaciousSet (500 samples)
- [ ] Recorded your WER score
- [ ] Tried at least one other dataset
- [ ] Reviewed detailed results

______________________________________________________________________

## Exercise 2: Deploy to Hugging Face Spaces (15 min)

### Goal

Create a live demo anyone can use.

### Instructions

**Step 1: Create a Space**

1. Go to [huggingface.co](https://huggingface.co)
2. Click your profile → **New Space**
3. Configure:
   - **Name**: `tiny-audio-demo`
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU basic (free)
4. Click **Create Space**

**Step 2: Deploy with the script**

```bash
poetry run python scripts/deploy_to_hf_space.py \
    --space-url https://huggingface.co/spaces/your-username/tiny-audio-demo \
    --force
```

**What this does:**

- Copies `demo/gradio/app.py`, `requirements.txt`, `README.md`
- Sets up Git LFS for audio files
- Pushes to your Space

**Step 3: Configure your model**

Set your model ID using an environment variable:

1. Go to your Space
2. Click **Settings** → **Variables and secrets**
3. Click **New variable**
4. Add:
   - **Name**: `MODEL_ID`
   - **Value**: `your-username/your-model-name`
5. Click **Save**

Space rebuilds automatically (2-3 minutes).

**Step 4: Test**

- Upload audio files
- Record from microphone
- Share the link!

### Success Checkpoint

- [ ] Created Hugging Face Space
- [ ] Deployed with script
- [ ] Updated model ID
- [ ] Demo working

______________________________________________________________________

## Exercise 3: Set Up Inference Endpoints (10 min)

### Goal

Understand production deployment options.

### Option A: Free Serverless (Limited)

Your public model already has free inference:

```python
from huggingface_hub import InferenceClient

client = InferenceClient()
result = client.automatic_speech_recognition(
    "audio.wav",
    model="your-username/your-model"
)
print(result["text"])
```

**Limitations**: Rate limited, cold starts, not for production.

### Option B: Dedicated Endpoint (Paid)

For production:

1. Go to your model on Hugging Face
2. Click **Deploy** → **Inference Endpoints**
3. Configure region, GPU type, scaling
4. Click **Create Endpoint**

**Test with eval script:**

```bash
poetry run python scripts/eval.py \
    "https://your-endpoint.endpoints.huggingface.cloud" \
    --dataset loquacious \
    --max-samples 10
```

### Success Checkpoint

- [ ] Understand serverless vs dedicated
- [ ] (Optional) Created endpoint
- [ ] Know how to test endpoints

______________________________________________________________________

## Exercise 4: Add to Leaderboard (5 min)

### Goal

Add your results to the community leaderboard.

### Instructions

**Step 1: Get your official WER**

```bash
poetry run eval your-username/your-model --dataset loquacious --max-samples 500
```

Record the **Corpus WER**.

**Step 2: Edit the README**

In the repo's `README.md`, find the leaderboard table and add your entry:

```markdown
| Rank | Contributor | WER | Model | Date |
|------|------------|-----|-------|------|
| 🥇 | [@alexkroman](https://github.com/alexkroman) | **12.14** | [mazesmazes/tiny-audio](https://huggingface.co/mazesmazes/tiny-audio) | 2025-10-23 |
| 🥈 | [@your-username](https://github.com/your-username) | **XX.XX** | [your-username/your-model](https://huggingface.co/your-username/your-model) | 2025-XX-XX |
```

**Step 3: Submit a PR**

1. Fork the repository
2. Make your changes
3. Submit PR titled: "Add [username] to leaderboard (WER: XX.XX%)"

### Success Checkpoint

- [ ] Have official WER score
- [ ] Added to leaderboard
- [ ] Submitted PR

______________________________________________________________________

## Congratulations!

You've completed the course! You now have:

- ✅ A trained ASR model on Hugging Face
- ✅ Evaluation results across multiple datasets
- ✅ A live demo anyone can use
- ✅ Your name on the leaderboard

### What's Next?

**Improve your model:**

- Try different projector types (MoE, SwiGLU, Residual)
- Use a larger decoder
- Train on more diverse data

**Extend the project:**

- Add multilingual support
- Train other tasks (emotion, audio description)
- Build a real application

**Share your work:**

- Write a blog post
- Present at a meetup
- Contribute back to the repo

______________________________________________________________________

[← Class 2: Training](./2-training.md)
