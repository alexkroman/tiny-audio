#!/usr/bin/env python3
"""Test batched decoding of larger audio files."""

import torch
from transformers import pipeline

# Create a test audio file (5 minutes of silence to test chunking)
sample_rate = 16000
duration = 5 * 60  # 5 minutes
audio_array = torch.zeros(sample_rate * duration)

print(f"Testing with {duration}s audio ({len(audio_array)} samples)")
print(f"Expected chunks: ~{duration / 30} (30s chunks with 5s overlap)")

# Load model on CPU to avoid MPS issues
print("\nLoading model...")
pipe = pipeline(
    "automatic-speech-recognition",
    model="mazesmazes/tiny-audio",
    trust_remote_code=True,
    device=-1,  # CPU
)

print(f"Model loaded on device: {pipe.model.device}")

# Test transcription
print("\nTranscribing...")
audio_dict = {"array": audio_array.numpy(), "sampling_rate": sample_rate}
result = pipe(audio_dict)

print(f"\nResult: {result}")
print(f"Text length: {len(result['text'])}")
