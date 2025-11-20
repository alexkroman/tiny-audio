# Class 4: Evaluation, Debugging, and Deployment

**Duration**: 1 hour (20 min lecture + 40 min hands-on)

**Goal**: Evaluate your trained model, learn basic debugging, and deploy it to the world.

## Learning Objectives

By the end of this class, you will:

- Understand Word Error Rate (WER) and how to calculate it.
- Run evaluation on your trained model.
- Know how to spot common training issues.
- Push your model to the Hugging Face Hub and write a model card.
- Deploy a live, interactive demo of your model using Hugging Face Spaces.

______________________________________________________________________

# PART A: LECTURE (20 minutes)

## 1. What is a "Good" Model? Understanding WER (5 min)

After training, the first question is: "How well did it do?" For ASR, the industry-standard metric is the **Word Error Rate (WER)**.

**Formula:**
`WER = (Substitutions + Insertions + Deletions) / Total Words in Reference`

**Example:**

- **Reference**: "hello world"
- **Hypothesis**: "hello there world" (1 Insertion)
- **WER**: `1 / 2 = 50%`

A lower WER is better. Here’s a general guide:

- **< 5%**: Excellent (Commercial systems)
- **5-10%**: Very Good
- **10-20%**: Good (Our target for this course)
- **> 30%**: Poor

WER is a great quantitative score, but always remember to perform qualitative "vibe testing" by listening to your model's outputs to truly understand its strengths and weaknesses.

______________________________________________________________________

## 2. Common Training Problems (5 min)

If your WER is high, you might have run into a common training issue. Here’s what to look for in your training logs:

1. **Loss is Not Decreasing**:

   - **Symptom**: The training loss stays flat or even goes up.
   - **Common Cause**: The **learning rate** is too high or too low. The model is either "overshooting" the solution or not moving at all.
   - **Quick Fix**: Try restarting the training with a learning rate 10x smaller or larger.

1. **Overfitting**:

   - **Symptom**: The training loss goes down steadily, but your evaluation WER is poor (or a validation loss, if used, starts increasing).
   - **Common Cause**: The model has "memorized" the training data but can't generalize to new, unseen data.
   - **Quick Fix**: Train on more data or for fewer steps.

______________________________________________________________________

## 3. Sharing Your Work: The Hugging Face Hub (5 min)

The Hugging Face Hub is the "GitHub for Machine Learning." It's the central place where the open-source community shares models, datasets, and demos.

**Why share your work?**

- **Build a Portfolio**: A public model is a fantastic demonstration of your practical ML engineering skills.
- **Contribute to the Community**: You're giving back to the ecosystem that provided the tools and pre-trained models you used.
- **Reproducibility & Feedback**: Others can easily use your model, verify your results, and provide valuable feedback.

______________________________________________________________________

## 4. From Model to Demo (5 min)

Once your model is on the Hub, there are two key things that make it useful to the world:

1. **The Model Card (`README.md`)**:

   - This is your model's official documentation. A good model card explains what the model is, how it was trained, its performance (WER), and its limitations. It is crucial for transparency and reproducibility.

1. **Hugging Face Spaces**:

   - Spaces are a free and easy way to build and host live, interactive demos of your models. With just a few lines of Python (using Gradio), you can create a web UI that anyone can use to try out your model, directly from their browser.

In the workshop, you'll put your trained model on the Hub and build a live demo in minutes.

______________________________________________________________________

# PART B: HANDS-ON WORKSHOP (40 minutes)

## Goal

Go from a locally trained model to a publicly deployed ASR demo in three steps: Evaluate, Push, and Deploy.

______________________________________________________________________

### Exercise 1: Evaluate Your Model (15 min)

First, let's get a quantitative score for the model you trained in the previous class.

**Instructions:**

1. Find the output directory of your last training run (it will be timestamped in the `outputs/` folder).

1. Run the evaluation script, pointing it to your local model directory. This will compute the WER on the test set.

   ```bash
   # Replace [your-training-output-dir] with the actual folder name
   poetry run python scripts/eval.py outputs/[your-training-output-dir] --max-samples 100
   ```

   *(Note: `--max-samples 100` gives a quick estimate. Run without it for the full ~30-minute evaluation.)*

1. Note your final WER score! This is the key performance metric for your model card.

______________________________________________________________________

### Exercise 2: Push Your Model to the Hub (15 min)

Now, let's share your model with the world.

**Instructions:**

1. If you haven't already, get a Hugging Face access token with "write" permissions from **Settings -> Access Tokens** on the Hub.

1. Log in via your terminal:

   ```bash
   poetry run huggingface-cli login
   ```

   (Paste your token when prompted.)

1. Use the provided `push_to_hub.py` script to upload your model. Choose a name for it like `tiny-audio-yourname`.

   ```bash
   # Usage: python push_to_hub.py <local_model_path> <hub_model_id>
   poetry run python scripts/push_to_hub.py outputs/[your-training-output-dir] your-username/tiny-audio-yourname
   ```

1. Visit your model's new page on the Hub (`https://huggingface.co/your-username/tiny-audio-yourname`).

1. Click **"Edit model card"** and fill it out. At a minimum, add your model's **WER score** to the performance section. A good model card is essential!

______________________________________________________________________

### Exercise 3: Create a Public Demo with Spaces (10 min)

Finally, let's create a live demo that anyone can use.

**Instructions:**

1. On the Hugging Face Hub, go to **New -> Space**.

   - Give it a name (e.g., `tiny-audio-demo`).
   - Select the **Gradio** SDK.
   - Choose the free **CPU basic** hardware.
   - Click **"Create Space"**.

1. You'll be taken to a page with instructions to clone the new repository. Clone it to your local machine.

   ```bash
   git clone https://huggingface.co/spaces/your-username/tiny-audio-demo
   cd tiny-audio-demo
   ```

1. Create a file named `app.py` with the following content. This code tells Gradio to load your model from the Hub and create a simple interface for it.

   ```python
   import gradio as gr

   # IMPORTANT: Replace this with YOUR model's ID on the Hub
   MODEL_ID = "your-username/tiny-audio-yourname"

   # Gradio will automatically create a UI for this ASR pipeline
   demo = gr.load(
       f"huggingface/{MODEL_ID}",
       title="Tiny Audio ASR Demo",
       description=f"A speech recognition demo for the model **{MODEL_ID}**. Upload audio or record from your microphone."
   )

   if __name__ == "__main__":
       demo.launch()
   ```

1. Push the `app.py` file to your Space.

   ```bash
   git add app.py
   git commit -m "Add Gradio app"
   git push
   ```

1. Go back to your Space page on the Hub. After a few moments of building, your interactive demo will be live! You can now share the link with anyone in the world.

______________________________________________________________________

[Previous: Class 3: Training](./3-training.md)
