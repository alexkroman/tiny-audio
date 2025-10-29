import torch
from src.asr_modeling import ASRModel
from src.asr_config import ASRConfig
from transformers import Wav2Vec2FeatureExtractor
import librosa

# Load model
print("Loading Tiny Audio model...")
config = ASRConfig.from_pretrained("mazesmazes/tiny-audio", trust_remote_code=True)
model = ASRModel.from_pretrained("mazesmazes/tiny-audio", config=config)
print("✓ Model loaded!\n")

# Load and process audio
audio_path = "test.wav"
waveform, sr = librosa.load(audio_path, sr=16000)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/hubert-xlarge-ls960-ft"
)
inputs = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt")

# Get encoder output
print("="*60)
print("STEP 1: AUDIO ENCODER (HuBERT)")
print("="*60)
with torch.no_grad():
    encoder_output = model.encoder(**inputs).last_hidden_state

print(f"Encoder output shape: {encoder_output.shape}")
print(f"  [batch={encoder_output.shape[0]}, time={encoder_output.shape[1]}, dim={encoder_output.shape[2]}]")
print(f"  Each frame = ~20ms of audio")
print(f"  Total coverage = ~{encoder_output.shape[1] * 20}ms")

# Manually trace through projector
projector = model.projector
k = projector.k  # Downsampling rate

print("\n" + "="*60)
print("STEP 2: FRAME STACKING (5x downsampling)")
print("="*60)

# Stack frames
batch_size, seq_len, dim = encoder_output.shape
remainder = seq_len % k
if remainder:
    pad_len = k - remainder
    encoder_output = torch.nn.functional.pad(encoder_output, (0, 0, 0, pad_len))
    print(f"Padded sequence: {seq_len} → {encoder_output.shape[1]} frames")

stacked = encoder_output.contiguous().view(batch_size, -1, dim * k)
print(f"After stacking: {stacked.shape}")
print(f"  [batch={stacked.shape[0]}, time={stacked.shape[1]}, dim={stacked.shape[2]}]")
print(f"  Time reduction: {seq_len} → {stacked.shape[1]} ({seq_len/stacked.shape[1]:.1f}x)")
print(f"  Each frame now = ~{20 * k}ms of audio")

print("\n" + "="*60)
print("STEP 3: PRE-NORMALIZATION (RMSNorm)")
print("="*60)
prenorm = projector.ln_pre(stacked)
print(f"Shape: {prenorm.shape} (unchanged)")
print(f"Before norm - Mean: {stacked.mean():.4f}, Std: {stacked.std():.4f}")
print(f"After norm  - Mean: {prenorm.mean():.4f}, Std: {prenorm.std():.4f}")

print("\n" + "="*60)
print("STEP 4: SwiGLU TRANSFORMATION")
print("="*60)
gate = projector.gate_proj(prenorm)
up = projector.up_proj(prenorm)
print(f"Gate projection: {gate.shape}")
print(f"Up projection:   {up.shape}")

activated = torch.nn.functional.silu(gate) * up
print(f"After SwiGLU:    {activated.shape}")
print(f"  SiLU(gate) ⊗ up = gated features")

print("\n" + "="*60)
print("STEP 5: DOWN PROJECTION")
print("="*60)
down = projector.down_proj(activated)
print(f"Down projection: {down.shape}")
print(f"  Dimension: 8192 → 2048 (LLM input size)")

print("\n" + "="*60)
print("STEP 6: POST-NORMALIZATION")
print("="*60)
output = projector.ln_post(down)
print(f"Final output: {output.shape}")
print(f"Before norm - Mean: {down.mean():.4f}, Std: {down.std():.4f}")
print(f"After norm  - Mean: {output.mean():.4f}, Std: {output.std():.4f}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Input:  {encoder_output.shape[1]} frames × {encoder_output.shape[2]}D (HuBERT)")
print(f"Output: {output.shape[1]} frames × {output.shape[2]}D (SmolLM3-ready)")
print(f"Time reduction: {encoder_output.shape[1] / output.shape[1]:.1f}x")
print(f"Dimension change: {encoder_output.shape[2]}D → {output.shape[2]}D")
print(f"\n✓ Audio embeddings are now ready for the language model!")
