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

## 1. A Philosophy of Evaluation (5 min)

Before we dive into the specifics of Word Error Rate (WER), let's talk about how to think about evaluation. Relying on a single metric can be misleading. A good evaluation strategy is layered and includes both quantitative and qualitative measures.

**The Layered Evaluation Suite:**

- **Primary Metric (WER)**: This is our main quantitative measure of performance. It's great for tracking progress and comparing models, but it doesn't tell the whole story.
- **Secondary Benchmarks**: In a real-world project, you would also track performance on a range of other benchmarks that test for different capabilities, such as robustness to noise, performance on different accents, or understanding of specific domains.
- **Qualitative Analysis ("Vibe Testing")**: This is where you interact with your model directly. You listen to its transcriptions, you try it on different kinds of audio, and you get a feel for its strengths and weaknesses. This is often where you'll find subtle bugs or quirks that the metrics don't capture.

Throughout this chapter, we'll focus on WER as our primary metric, but always keep in mind that it's just one piece of the puzzle.

---

## 2. Understanding Word Error Rate (5 min)

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

## 3. Proactive Stability Measures (5 min)

The best way to deal with training instabilities is to prevent them from happening in the first place. Here are a few proactive measures that are commonly used in large-scale training:

- **Data Filtering and Shuffling**: A surprising number of training instabilities can be traced back to bad data. Properly cleaning, filtering, and shuffling your training data is one of the most effective ways to ensure a stable training run.
- **Z-loss**: This is a regularization technique that penalizes the model for producing logits that are too large, which can be a source of instability. It's a simple and effective way to keep the model's outputs in a reasonable range.
- **QK-Norm**: This involves normalizing the query and key vectors in the attention mechanism. It's another technique that has been shown to improve training stability, especially in large models.

While we won't be implementing these techniques in this course, it's important to know that they exist and are part of the standard toolkit for large-scale training.

---

## 4. Debugging Common Issues (10 min)

Training instabilities, often seen as "spikes" in the loss curve, are a common part of the training marathon. Here's a more systematic way to think about them.

### Recoverable vs. Non-recoverable Spikes

- **Recoverable Spikes**: The loss jumps up but then returns to its previous trajectory. These are common and usually not a cause for alarm. They are often caused by a few "bad" batches of data.
- **Non-recoverable Spikes**: The loss jumps up and stays there, or even continues to increase. This is a sign of a more serious problem, and it usually requires intervention.

### Common Culprits

- **Learning Rate Too High**: This is the most common cause of instability, especially at the beginning of training.
- **Bad Data**: A batch of corrupted or out-of-distribution data can cause a sudden loss spike.
- **Data-Parameter Interactions**: Sometimes, a specific batch of data will interact with the model's weights in an unfortunate way, leading to a spike. This is often hard to predict.

### Damage Control

When you encounter a non-recoverable spike, here are a few things you can try:

1.  **Rewind and Skip**: The most common solution is to go back to a checkpoint from before the spike and restart the training, skipping the problematic batch of data.
2.  **Reduce the Learning Rate**: If the instability persists, you may need to reduce the learning rate. You can do this by modifying the `learning_rate` parameter in your Hydra config and restarting the training.
3.  **Tighten Gradient Clipping**: Gradient clipping prevents the gradients from becoming too large. If you're seeing a lot of instability, you can try reducing the `max_grad_norm` parameter in your Hydra config.

### Specific Scenarios

**Problem 1: Loss not decreasing**

- **Symptoms**: Loss stays flat or increases.
- **Solutions**: Try a lower learning rate, a larger effective batch size, or adjusting gradient clipping.

**Problem 2: Overfitting**

- **Symptoms**: Training loss goes down, but validation loss goes up.
- **Solutions**: Add more data, or increase regularization (e.g., by adding LoRA dropout).

**Problem 3: Model outputs gibberish**

- **Symptoms**: Transcriptions are nonsense or repetitive.
- **Solutions**: This could be a sign that the projector is not learning correctly. You could try increasing the LoRA rank of the decoder or lowering the learning rate.

**Problem 4: Out of memory**

- **Symptoms**: CUDA out of memory errors.
- **Solutions**: Reduce the `per_device_train_batch_size` and increase `gradient_accumulation_steps` to maintain the same effective batch size. You can also enable `gradient_checkpointing`.

---

# PART B: HANDS-ON WORKSHOP (40 minutes)

> **Students**: Follow these instructions step-by-step.
>
> **Instructor**: Circulate and help students.

## Workshop Overview

In the next 40 minutes, you will:

- **Exercise 1**: Run evaluation on your trained model
- **Exercise 2**: Analyze error patterns and predictions
- **Exercise 3**: Debug training issues (if any)
- **Exercise 4**: Calculate and interpret WER metrics

By the end, you'll know exactly how well your model performs!

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
