______________________________________________________________________

license: mit
language:

- en
  datasets:
- speechbrain/LoquaciousSet
  base_model:
- facebook/hubert-xlarge-ls960-ft
- HuggingFaceTB/SmolLM3-3B
  pipeline_tag: automatic-speech-recognition
  tags:
- asr
- speech-recognition
- audio
- smollm
- hubert

______________________________________________________________________

# Tiny Audio Model Card

This model was born from a simple idea: what if anyone could train a powerful, modern speech recognition model for the price of a few coffees? This model is the result of the [Tiny Audio course](https://github.com/alexkroman/tiny-audio/blob/main/docs/course/0-course-overview.md), a free, hands-on guide to building your own ASR system from scratch.

## The Story of this Model

This model isn't the product of a massive research lab with an unlimited budget. It's the result of a 24-hour training run on a single GPU, made possible by an efficient projector-only training approach. By combining the strengths of a massive pretrained audio encoder (`facebook/hubert-xlarge-ls960-ft`) and a powerful language model (`HuggingFaceTB/SmolLM3-3B`), and only training a small projector between them, we can create a high-quality ASR model with minimal resources.

This model is a testament to the power of open-source and the incredible tools and models that are now available to everyone.

## Intended Use

This model is for you. It's for the curious, the builders, the learners. It's for anyone who wants to understand how modern AI works by getting their hands dirty. Use it to transcribe your podcasts, your meetings, your voice memos. But more importantly, use it as a starting point. Fork it, fine-tune it, break it, and make it your own.

## Performance

This model achieves a Word Error Rate (WER) of **12.14%** on the LoquaciousSet test set. It's not perfect, but it's a solid baseline that you can build on. See how it compares to other models on the [community leaderboard](https://github.com/alexkroman/tiny-audio#leaderboard).

## How to Use

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="mazesmazes/tiny-audio", trust_remote_code=True)

result = pipe("path/to/audio.wav")
print(result["text"])
```

## How to Get Involved

This project is more than just a model; it's a community. Here's how you can get involved:

- **Take the course**: The best way to start is to go through the [free 6-hour course](https://github.com/alexkroman/tiny-audio/blob/main/docs/course/0-course-overview.md) and train your own model.
- **Share your results**: Add your model to the [leaderboard](https://github.com/alexkroman/tiny-audio#leaderboard) and share what you've learned.
- **Join the conversation**: Ask questions, share your ideas, and connect with other builders in the [GitHub Discussions](https://github.com/alexkroman/tiny-audio/discussions).
