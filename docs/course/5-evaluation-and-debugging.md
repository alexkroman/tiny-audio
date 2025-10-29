# Class 5: Evaluation and Debugging

**Duration**: 1 hour (20 min lecture + 40 min hands-on)
**Goal**: Evaluate your trained model and understand how to improve it

## Learning Objectives

By the end of this class, you will:

- Understand Word Error Rate (WER) and how it's calculated
- Evaluate your model on test datasets
- Debug common training issues
- Analyze model predictions and errors
- Know strategies to improve performance

---

# PART A: LECTURE (20 minutes)

> **Instructor**: Present these concepts. Students should just listen.

## 1. Understanding Word Error Rate (5 min)

### What is WER?

**WER** = **W**ord **E**rror **R**ate

The standard metric for evaluating ASR systems.

**Formula:**

```
WER = (Substitutions + Insertions + Deletions) / Total Words
```

**Example:**

```
Reference:  "the quick brown fox jumps"     (5 words)
Hypothesis: "the quick brown dog jumped"

Errors:
- Substitution: "fox" → "dog"  (1 error)
- Substitution: "jumps" → "jumped"  (1 error)

WER = 2 / 5 = 0.40 = 40%
```

### Types of Errors

**Substitution (S)**: Wrong word

```
Ref: "the cat sat"
Hyp: "the dog sat"
     └─ S ─┘
```

**Insertion (I)**: Extra word

```
Ref: "hello world"
Hyp: "hello big world"
          └ I ┘
```

**Deletion (D)**: Missing word

```
Ref: "hello world"
Hyp: "hello"
          └ D ┘
```

### What's a Good WER?

**ASR Performance Benchmarks:**

- **< 5%**: Excellent (commercial systems)
- **5-10%**: Very good (usable for most tasks)
- **10-20%**: Good (acceptable for many applications)
- **20-30%**: Fair (needs improvement)
- **> 30%**: Poor (significant issues)

**Tiny Audio target**: ~12% WER

**Context matters**:

- Clean speech: Lower WER expected
- Noisy environments: Higher WER acceptable
- Accented speech: Often higher WER
- Domain-specific: Medical/legal need very low WER

### Beyond WER

**Character Error Rate (CER)**: Similar but for characters

- Useful for languages without clear word boundaries
- More granular error analysis

**Normalized WER**: Apply text normalization first

- Remove punctuation
- Lowercase
- Expand contractions ("don't" → "do not")
- Makes comparison fairer

---

## 2. The Evaluation Pipeline (5 min)

### LoquaciousSet Benchmark

**Dataset**: LoquaciousSet test split

- Diverse speakers and accents
- Various acoustic conditions
- Multiple speech types
- Industry-standard benchmark

**Why this dataset?**

- Representative of real-world speech
- Balanced across demographics
- Fair comparison with other models
- Same data used for training (different split)

### Evaluation Process

```
1. Load trained model
2. Load test dataset samples
3. For each audio sample:
   a. Run inference (get transcription)
   b. Normalize reference and hypothesis
   c. Calculate WER
4. Aggregate results
5. Analyze errors
```

### Text Normalization

Both reference and hypothesis are normalized:

**Steps:**

1. Remove `<inaudible>` tags
2. Remove disfluencies ("uh", "um")
3. Apply Whisper normalization
   - Lowercase
   - Remove punctuation
   - Expand contractions
   - Standardize numbers
4. Apply truecasing

**Why normalize?**

- Punctuation differences shouldn't count as errors
- "Hello" vs "hello" - same word!
- Fair comparison across different text formats

---

## 3. Common Issues and Solutions (10 min)

### Training Issues

**Problem 1: Loss not decreasing**

Symptoms:

- Loss stays flat or increases
- No improvement over time

Possible causes:

- Learning rate too high
- Batch size too small
- Gradient clipping too aggressive
- Data quality issues

Solutions:

```yaml
# Try lower learning rate
training:
  learning_rate: 5e-5  # instead of 1e-4

# Increase effective batch size
training:
  gradient_accumulation_steps: 8  # instead of 4

# Adjust gradient clipping
training:
  max_grad_norm: 5.0  # instead of 1.0
```

**Problem 2: Loss decreasing but val loss increasing**

Symptoms:

- Training loss goes down
- Validation loss goes up
- Model is overfitting!

Solutions:

```yaml
# Add more data
data:
  max_train_samples: null  # Use all data

# Increase regularization (LoRA dropout)
peft:
  lora_dropout: 0.1

encoder_lora:
  lora_dropout: 0.1
```

**Problem 3: Model outputs gibberish**

Symptoms:

- Transcriptions are nonsense
- Repeating words
- Random characters

Possible causes:

- Projector not learning properly
- Decoder LoRA rank too low
- Learning rate too high

Solutions:

```yaml
# Increase decoder capacity
peft:
  r: 128  # instead of 64

# Lower learning rate
training:
  learning_rate: 5e-5
```

**Problem 4: Out of memory**

Symptoms:

- CUDA out of memory errors
- Training crashes

Solutions:

```yaml
# Reduce batch size
training:
  per_device_train_batch_size: 4  # instead of 8
  gradient_accumulation_steps: 8  # keep effective batch size

# Use gradient checkpointing
training:
  gradient_checkpointing: true

# Shorter audio
data:
  max_audio_seconds: 25  # instead of 30
```

### Inference Issues

**Problem: Slow inference**

Solutions:

- Use Flash Attention 2 (already enabled)
- Batch multiple samples
- Use mixed precision (bf16)
- Consider model quantization

**Problem: Poor quality on specific audio types**

Solutions:

- Collect more data of that type
- Analyze error patterns
- Adjust training data distribution
- Fine-tune on domain-specific data

---

# PART B: HANDS-ON WORKSHOP (40 minutes)

> **Students**: Follow these instructions step-by-step.
>
> **Instructor**: Circulate and help students.

---

## Workshop Exercise 1: Evaluate Your Model (15 min)

### Goal

Calculate WER for your trained model.

### Your Task

Run evaluation on test set and get your WER score.

### Instructions

**Step 1: Ensure training is complete**

Check that your model training finished:

```bash
ls outputs/stage1/  # or outputs/my_experiment/
```

You should see:

- `config.json`
- `model.safetensors`
- `trainer_state.json`

**Step 2: Run evaluation**

```bash
# Evaluate on 100 samples (quick test)
poetry run eval outputs/stage1 --max-samples 100

# Full evaluation (takes ~30 min)
poetry run eval outputs/stage1
```

**Alternative: Evaluate from HuggingFace Hub**

```bash
# If you already pushed your model
poetry run eval your-username/your-model-name --max-samples 100
```

**What happens:**

1. Loads your trained model
2. Loads LoquaciousSet test samples
3. Runs inference on each
4. Calculates WER
5. Saves detailed results

**Expected output:**

```
Loading model from outputs/stage1...
✓ Model loaded

Loading LoquaciousSet test set...
✓ Loaded 100 samples

Evaluating...
[████████████████████] 100/100

Results:
========================================
Word Error Rate (WER): 13.45%
Character Error Rate (CER): 6.23%

Total samples: 100
Total words: 5,432
Errors:
  Substitutions: 512
  Insertions: 87
  Deletions: 132

Results saved to: outputs/eval_<timestamp>/results.txt
```

**Step 3: Examine detailed results**

```bash
# Look at the detailed results file
cat outputs/eval_*/results.txt | head -50
```

You'll see line-by-line comparisons:

```
Sample 1:
Reference:  the quick brown fox jumps over the lazy dog
Hypothesis: the quick brown dog jumped over the lazy dog
WER: 40.0%

Sample 2:
Reference:  hello world this is a test
Hypothesis: hello world this is a test
WER: 0.0%

...
```

### Success Checkpoint

- [ ] Evaluation ran successfully
- [ ] Got a WER score
- [ ] Examined detailed results
- [ ] Saved results file for later

**Your WER**: ___________% (write it down!)

---

## Workshop Exercise 2: Analyze Error Patterns (15 min)

### Goal

Understand what types of errors your model makes.

### Your Task

Analyze common error patterns to identify improvement opportunities.

### Instructions

**Step 1: Create error analysis script**

Create `analyze_errors.py`:

```python
import re
from pathlib import Path
from collections import Counter

# Load results file
results_dir = Path("outputs").glob("eval_*")
latest_results = max(results_dir, key=lambda p: p.stat().st_mtime)
results_file = latest_results / "results.txt"

print(f"Analyzing: {results_file}\n")

# Parse results
errors = []
with open(results_file) as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    if lines[i].startswith("Reference:"):
        ref = lines[i].replace("Reference:", "").strip()
        hyp = lines[i+1].replace("Hypothesis:", "").strip()

        # Find word-level differences
        ref_words = ref.lower().split()
        hyp_words = hyp.lower().split()

        # Simple word-level comparison
        for r, h in zip(ref_words, hyp_words):
            if r != h:
                errors.append((r, h))

        i += 3
    else:
        i += 1

# Analyze most common errors
error_counts = Counter(errors)

print("="*60)
print("TOP 20 MOST COMMON ERRORS")
print("="*60)
print(f"{'Reference':<20} {'Hypothesis':<20} {'Count':<10}")
print("="*60)

for (ref, hyp), count in error_counts.most_common(20):
    print(f"{ref:<20} {hyp:<20} {count:<10}")

print(f"\n✓ Analyzed {len(errors)} total errors")
print(f"✓ Found {len(error_counts)} unique error types")
```

**Step 2: Run analysis**

```bash
poetry run python analyze_errors.py
```

**Expected output:**

```
============================================================
TOP 20 MOST COMMON ERRORS
============================================================
Reference            Hypothesis           Count
============================================================
the                  a                    47
to                   too                  32
they're              there                28
its                  it's                 23
...
```

**Step 3: Categorize errors**

Look for patterns:

- **Homophones**: "to/too", "they're/there/their"
- **Short words**: "a/the", "is/it"
- **Punctuation-related**: "its/it's", "cant/can't"
- **Domain-specific**: Technical terms, names

### Success Checkpoint

- [ ] Ran error analysis
- [ ] Identified top error patterns
- [ ] Understand common confusion pairs
- [ ] Ideas for improvement

**Common patterns I found:**

1. \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
2. \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
3. \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

---

## Workshop Exercise 3: Compare with Baseline (10 min)

### Goal

See how your model compares to the pre-trained baseline.

### Your Task

Evaluate the baseline model and compare results.

### Instructions

**Step 1: Evaluate baseline**

```bash
# Evaluate pre-trained model
poetry run eval mazesmazes/tiny-audio --max-samples 100
```

**Step 2: Create comparison script**

Create `compare_models.py`:

```python
# Your model WER
your_wer = 13.45  # Replace with your actual WER

# Baseline WER (from leaderboard)
baseline_wer = 12.14

# Calculate improvement
improvement = baseline_wer - your_wer
percent_change = (improvement / baseline_wer) * 100

print("="*60)
print("MODEL COMPARISON")
print("="*60)
print(f"Baseline WER:  {baseline_wer:.2f}%")
print(f"Your WER:      {your_wer:.2f}%")
print("="*60)

if improvement > 0:
    print(f"✓ Improvement: {improvement:.2f} percentage points ({percent_change:.1f}%)")
    print("  Congratulations! Your model is better!")
elif improvement == 0:
    print("= Tied with baseline")
else:
    print(f"⚠ Difference: {abs(improvement):.2f} percentage points worse")
    print("  Tips for improvement:")
    print("  • Train longer (more steps)")
    print("  • Use more data")
    print("  • Increase LoRA rank")
    print("  • Tune learning rate")
```

**Step 3: Run comparison**

```bash
poetry run python compare_models.py
```

### Success Checkpoint

- [ ] Evaluated baseline model
- [ ] Compared WER scores
- [ ] Understand relative performance
- [ ] Have improvement ideas if needed

---

# CLASS SUMMARY

## What We Covered Today

**Lecture (20 min):**

- Word Error Rate (WER) calculation
- Evaluation pipeline and normalization
- Common training and inference issues
- Debugging strategies

**Workshop (40 min):**

- Evaluated trained model and got WER score
- Analyzed error patterns
- Compared with baseline model

## Key Takeaways

✅ WER measures ASR accuracy: (S+I+D) / Total Words
✅ Target WER for Tiny Audio: ~12%
✅ Text normalization ensures fair comparison
✅ Error analysis reveals improvement opportunities
✅ Common issues have known solutions

## Homework

**Required** (Do before Class 6!):

1. **Note your final WER score**: ____________%
2. **Prepare model card** with:
   - Your name
   - Training details
   - WER score
   - Example transcriptions
3. **Create HuggingFace account** if you don't have one

**Optional**:

1. Try to improve your WER by tweaking hyperparameters
2. Run evaluation on different test sets
3. Analyze specific error categories

## Check Your Understanding

1. **How is WER calculated?**
   - (Substitutions + Insertions + Deletions) / Total Words
   - Measures word-level accuracy
   - Lower is better

2. **What's a good WER for speech recognition?**
   - < 10% is very good
   - 10-20% is acceptable
   - > 30% needs improvement

3. **Why normalize text before evaluation?**
   - Fair comparison (punctuation, case)
   - Focus on actual transcription errors
   - Standard practice in ASR evaluation

4. **What if my model has high WER?**
   - Check training loss curves
   - Analyze error patterns
   - Try adjusting hyperparameters
   - Train longer or with more data

---

## Further Reading (Optional)

### Papers

- [Evaluation Metrics for ASR](https://arxiv.org/abs/2104.02138)
- [Text Normalization for ASR](https://arxiv.org/abs/2109.12791)

### Tools

- [JiWER library](https://github.com/jitsi/jiwer) - WER calculation
- [Whisper normalization](https://github.com/openai/whisper/blob/main/whisper/normalizers/)

### Datasets

- [LoquaciousSet](https://huggingface.co/datasets/speechbrain/LoquaciousSet)
- [LibriSpeech](http://www.openslr.org/12/)
- [Common Voice](https://commonvoice.mozilla.org/)

---

## Next Class

In [Class 6: Publishing and Deployment](./6-publishing-and-deployment.md), we'll:

- Push your model to HuggingFace Hub
- Create a professional model card
- Test the deployed model
- **Add your results to the leaderboard!**

**Prerequisites**:

- Trained model with WER score
- HuggingFace account created
- HuggingFace token ready

[Previous: Class 4: Training](./4-training.md) | [Next: Class 6: Publishing and Deployment](./6-publishing-and-deployment.md)

**This is it - your model will be live and public!**
