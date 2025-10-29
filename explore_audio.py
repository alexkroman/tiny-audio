import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from transformers import Wav2Vec2FeatureExtractor

# Load an audio file
audio_path = "test.wav"  # Update to your audio file
waveform, sr = librosa.load(audio_path, sr=16000)

print(f"Sample rate: {sr} Hz")
print(f"Duration: {len(waveform) / sr:.2f} seconds")
print(f"Shape: {waveform.shape}")
print(f"Value range: [{waveform.min():.4f}, {waveform.max():.4f}]")

# Apply feature extraction
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/hubert-xlarge-ls960-ft"
)
inputs = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt")

print(f"\nAfter feature extraction:")
print(f"Shape: {inputs.input_values.shape}")
print(f"Mean: {inputs.input_values.mean():.4f}")
print(f"Std: {inputs.input_values.std():.4f}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

# Original waveform
ax1.plot(np.arange(len(waveform)) / sr, waveform)
ax1.set_title("Raw Waveform")
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Amplitude")
ax1.grid(True, alpha=0.3)

# Normalized waveform
normalized = inputs.input_values.squeeze().numpy()
ax2.plot(np.arange(len(normalized)) / sr, normalized)
ax2.set_title("After Z-Normalization")
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("Normalized Amplitude")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("audio_processing.png", dpi=150)
print("\n✓ Saved visualization to audio_processing.png")
