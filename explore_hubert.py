import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import librosa

print("Loading HuBERT model...")
encoder = AutoModel.from_pretrained("facebook/hubert-xlarge-ls960-ft")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/hubert-xlarge-ls960-ft"
)
print("✓ Model loaded!\n")

# Load audio
audio_path = "test.wav"  # Update to your audio file
waveform, sr = librosa.load(audio_path, sr=16000)

# Extract features
inputs = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt")

# Pass through encoder
print("Processing audio through HuBERT...")
with torch.no_grad():
    outputs = encoder(**inputs)

embeddings = outputs.last_hidden_state

# Print dimensions
print("\n" + "="*50)
print("INPUT (Raw Audio)")
print("="*50)
print(f"Shape: {inputs.input_values.shape}")
print(f"Interpretation: [batch_size=1, samples={inputs.input_values.shape[-1]}]")
print(f"Duration: {inputs.input_values.shape[-1] / sr:.2f} seconds")

print("\n" + "="*50)
print("OUTPUT (Audio Embeddings)")
print("="*50)
print(f"Shape: {embeddings.shape}")
print(f"Interpretation: [batch_size=1, time_steps={embeddings.shape[1]}, embedding_dim={embeddings.shape[2]}]")

# Calculate compression
time_reduction = inputs.input_values.shape[-1] / embeddings.shape[1]
time_per_frame = (inputs.input_values.shape[-1] / embeddings.shape[1]) / sr

print(f"\nTime dimension reduction: {time_reduction:.1f}x")
print(f"Each embedding represents: ~{time_per_frame * 1000:.1f}ms of audio")
print(f"Embedding dimensionality: {embeddings.shape[-1]}D")

# Statistics
print("\n" + "="*50)
print("EMBEDDING STATISTICS")
print("="*50)
print(f"Mean: {embeddings.mean():.4f}")
print(f"Std: {embeddings.std():.4f}")
print(f"Min: {embeddings.min():.4f}")
print(f"Max: {embeddings.max():.4f}")

print("\n✓ Processing complete!")
