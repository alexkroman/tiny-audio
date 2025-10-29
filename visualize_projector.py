import torch
import matplotlib.pyplot as plt
from src.asr_modeling import ASRModel
from src.asr_config import ASRConfig
from transformers import Wav2Vec2FeatureExtractor
import librosa

# Load model
config = ASRConfig.from_pretrained("mazesmazes/tiny-audio", trust_remote_code=True)
model = ASRModel.from_pretrained("mazesmazes/tiny-audio", config=config)

# Load audio
audio_path = "test.wav"
waveform, sr = librosa.load(audio_path, sr=16000)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/hubert-xlarge-ls960-ft"
)
inputs = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt")

# Get embeddings at each stage
with torch.no_grad():
    # Encoder output
    audio_emb = model.encoder(**inputs).last_hidden_state

    # Projector output
    text_emb = model.projector(audio_emb)

    # LLM embedding for comparison (sample text)
    sample_text = model.tokenizer("Hello world", return_tensors="pt")
    llm_emb = model.decoder.get_input_embeddings()(sample_text.input_ids)

# Flatten for histograms
audio_flat = audio_emb.flatten().numpy()
text_flat = text_emb.flatten().numpy()
llm_flat = llm_emb.flatten().numpy()

# Plot distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(audio_flat, bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0].set_title('Audio Embeddings (HuBERT Output)')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')
axes[0].axvline(audio_flat.mean(), color='red', linestyle='--', label=f'Mean: {audio_flat.mean():.3f}')
axes[0].legend()

axes[1].hist(text_flat, bins=50, alpha=0.7, color='green', edgecolor='black')
axes[1].set_title('After Projector (Ready for LLM)')
axes[1].set_xlabel('Value')
axes[1].axvline(text_flat.mean(), color='red', linestyle='--', label=f'Mean: {text_flat.mean():.3f}')
axes[1].legend()

axes[2].hist(llm_flat, bins=50, alpha=0.7, color='orange', edgecolor='black')
axes[2].set_title('Text Embeddings (LLM Native)')
axes[2].set_xlabel('Value')
axes[2].axvline(llm_flat.mean(), color='red', linestyle='--', label=f'Mean: {llm_flat.mean():.3f}')
axes[2].legend()

plt.tight_layout()
plt.savefig('embedding_distributions.png', dpi=150)
print("✓ Saved visualization to embedding_distributions.png")

# Print statistics
print("\n" + "="*60)
print("EMBEDDING STATISTICS")
print("="*60)
print(f"{'Stage':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
print("="*60)
print(f"{'Audio (HuBERT)':<25} {audio_flat.mean():<10.4f} {audio_flat.std():<10.4f} {audio_flat.min():<10.4f} {audio_flat.max():<10.4f}")
print(f"{'After Projector':<25} {text_flat.mean():<10.4f} {text_flat.std():<10.4f} {text_flat.min():<10.4f} {text_flat.max():<10.4f}")
print(f"{'Text (LLM Native)':<25} {llm_flat.mean():<10.4f} {llm_flat.std():<10.4f} {llm_flat.min():<10.4f} {llm_flat.max():<10.4f}")
