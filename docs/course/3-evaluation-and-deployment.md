# Class 3: Evaluation and Deployment

*1 hour (15 min lecture + 45 min hands-on)*

**Goal**: Evaluate your model and deploy a public demo.

---

## Part A: Lecture (15 min)

### Word Error Rate (WER)

```
WER = (Substitutions + Insertions + Deletions) / Total Reference Words
```

| WER | Quality |
|-----|---------|
| < 5% | Excellent (commercial) |
| 5-10% | Very good |
| 10-20% | Good (our target) |
| > 30% | Poor |

### Evaluation Datasets

| Dataset | What it tests |
|---------|---------------|
| **LoquaciousSet** | General benchmark |
| **Earnings22** | Financial domain |
| **AMI** | Multi-speaker meetings |
| **LibriSpeech** | Clean read speech |

### Deployment Options

| Option | Cost | Use case |
|--------|------|----------|
| **HF Spaces** | Free | Interactive demos |
| **HF Inference Endpoints** | Paid | Production APIs |

---

## Part B: Hands-On (45 min)

### Exercise 1: Evaluate Your Model (15 min)

```bash
# Primary benchmark
poetry run ta eval -m your-username/your-model -n 500

# Domain-specific
poetry run ta eval -m your-username/your-model -d earnings22 -n 100

# Meetings
poetry run ta eval -m your-username/your-model -d ami -n 100
```

**Output:**

```
Sample 1: WER = 8.33%, Time = 1.23s
  Ref:  The quick brown fox jumps over the lazy dog
  Pred: The quick brown fox jumps over the lazy dog
...
CHECKPOINT @ 100 samples:
  Corpus WER: 12.45%
```

Results saved to `outputs/eval_*/results.txt`. Review high-WER samples to find patterns.

### Exercise 2: Deploy to Hugging Face Spaces (15 min)

**Create Space:**

1. Go to [huggingface.co](https://huggingface.co) → New Space
2. Name: `tiny-audio-demo`, SDK: Gradio, Hardware: CPU basic (free)

**Deploy:**

```bash
poetry run ta deploy hf --repo-id your-username/tiny-audio-demo --model your-username/your-model
```

Space builds in 2-3 minutes. Share the link!

### Exercise 3: Inference Endpoints (10 min)

**Free serverless** (rate limited):

```python
from huggingface_hub import InferenceClient

client = InferenceClient()
result = client.automatic_speech_recognition("audio.wav", model="your-username/your-model")
print(result["text"])
```

**Dedicated endpoint** (paid, production):

1. Go to your model → Deploy → Inference Endpoints
2. Configure GPU and scaling
3. Create endpoint

---

## Congratulations!

You now have:
- A trained ASR model on Hugging Face
- Evaluation results across datasets
- A live demo anyone can use

**Next steps:**
- Try different projector types
- Train on more data
- Build a real application

---

[← Class 2](./2-training.md) | [Quick Reference →](./4-quick-reference.md)
