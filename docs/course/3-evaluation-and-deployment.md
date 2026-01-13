# Class 3: Evaluation and Deployment

*1 hour (15 min lecture + 45 min hands-on)*

**Goal**: Evaluate your model, analyze errors, and deploy a public demo.

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
| **LoquaciousSet** | General benchmark (default) |
| **Earnings22** | Financial domain, earnings calls |
| **AMI** | Multi-speaker meetings |
| **LibriSpeech** | Clean read speech |

### Error Analysis

Understanding *why* your model fails is as important as measuring WER:

- **High-WER samples**: Which audio clips fail?
- **Entity errors**: Names, numbers, technical terms
- **Pattern detection**: Accents, noise, domain-specific issues

### Deployment Options

| Option | Cost | Use case |
|--------|------|----------|
| **HF Spaces** | Free | Interactive demos |
| **HF Inference Endpoints** | Paid | Production APIs |
| **Local server** | Self-hosted | Privacy, custom deployment |

---

## Part B: Hands-On (45 min)

### Exercise 1: Basic Evaluation (10 min)

```bash
# Evaluate on default dataset (LoquaciousSet)
poetry run ta eval -m your-username/your-model -n 500

# Evaluate on specific datasets
poetry run ta eval -m your-username/your-model -d earnings22 -n 100
poetry run ta eval -m your-username/your-model -d ami -n 100

# Evaluate multiple datasets at once
poetry run ta eval -m your-username/your-model -d loquacious -d earnings22 -n 100
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

Results saved to `outputs/eval_*/results.json`.

### Exercise 2: Error Analysis (15 min)

**Find high-error samples:**

```bash
# Find samples with WER > 30%
poetry run ta analysis high-wer your-username/your-model --threshold 30

# Find samples with WER > 50%
poetry run ta analysis high-wer your-username/your-model --threshold 50
```

This helps identify:
- Audio quality issues
- Accent/dialect challenges
- Domain-specific vocabulary gaps

**Compare models:**

```bash
# Compare your model against others
poetry run ta analysis compare your-model mazesmazes/tiny-audio

# Compare multiple models
poetry run ta analysis compare model1 model2 model3
```

**Find entity errors:**

```bash
# Extract named entities that were transcribed incorrectly
poetry run ta analysis entity-errors your-username/your-model
```

**Debug model health:**

```bash
# Check if MOSA model is healthy
poetry run ta debug check-mosa your-username/your-model

# Analyze LoRA adapter weights
poetry run ta debug analyze-lora your-username/your-model
```

### Exercise 3: Deploy to Hugging Face Spaces (10 min)

**Create Space:**

1. Go to [huggingface.co](https://huggingface.co) → New Space
2. Name: `tiny-audio-demo`, SDK: Gradio, Hardware: CPU basic (free)

**Deploy:**

```bash
poetry run ta deploy --repo-id your-username/tiny-audio-demo
```

The Space will use the default model. To use your own model, edit the deployment config.

Space builds in 2-3 minutes. Share the link!

**Run local demo:**

```bash
# Test locally before deploying
poetry run ta demo --model your-username/your-model --port 7860
```

### Exercise 4: Push Model to Hub (5 min)

```bash
# Push your trained model
poetry run ta push --repo-id your-username/your-model-name
```

This uploads:
- `config.json` - Model configuration
- `model.safetensors` - Projector weights
- `tokenizer.json` - Tokenizer files

### Exercise 5: Inference Endpoints (5 min)

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

## Advanced Evaluation

### Comparing with Commercial APIs

```bash
# Compare against AssemblyAI
export ASSEMBLYAI_API_KEY='your_key'
poetry run ta eval -m assemblyai --assemblyai-model universal -d loquacious -n 100 -w 4

# Compare against Deepgram
export DEEPGRAM_API_KEY='your_key'
poetry run ta eval -m deepgram -d loquacious -n 100
```

### Batch Evaluation

```bash
# Evaluate with multiple workers (faster for API-based models)
poetry run ta eval -m your-model -n 1000 -w 8
```

### Output Formats

```bash
# Save detailed results
poetry run ta eval -m your-model -n 100 -o ./my_results

# Results include:
# - results.json: Full results with predictions
# - summary.txt: WER statistics
# - errors.csv: High-error samples for analysis
```

---

## Debugging Poor Performance

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| High WER on all samples | Undertrained | Train longer, check loss curve |
| High WER on specific domain | Domain gap | Fine-tune on domain data |
| High WER on accented speech | Training data bias | Add diverse training data |
| Gibberish output | Model corrupted | Check checkpoint, retrain |
| Repeated words | Generation config | Check `no_repeat_ngram_size` |

---

## Congratulations!

You now have:
- A trained ASR model on Hugging Face
- Evaluation results across datasets
- Error analysis tools
- A live demo anyone can use

**Next steps:**
- Try different projector types (mosa, moe)
- Multi-stage training with LoRA
- Train on domain-specific data
- Build a real application

---

[← Class 2](./2-training.md) | [Quick Reference →](./4-quick-reference.md)
