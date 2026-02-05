#!/usr/bin/env python3
"""Test Mimi encoder-decoder round trip with real audio."""

import scipy.io.wavfile
import torch
import torchaudio
from datasets import load_dataset
from transformers import MimiModel

# Use HuggingFace Mimi but with proper encode/decode methods
print("Loading Mimi...")

mimi = MimiModel.from_pretrained("kyutai/mimi")
mimi.eval()

# Load real speech from LibriSpeech
print("Loading real speech sample...")

ds = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True
)
sample = ds[0]["audio"]
audio = torch.tensor(sample["array"]).float()
orig_sr = sample["sampling_rate"]

# Resample to 24kHz if needed
sr = 24000
if orig_sr != sr:
    audio = torchaudio.functional.resample(audio, orig_sr, sr)

# Take first 3 seconds
audio = audio[: sr * 3]
audio = audio.unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
print(f"Input audio: shape={audio.shape}, range=[{audio.min():.3f}, {audio.max():.3f}]")

# Save input
scipy.io.wavfile.write("test_input.wav", sr, (audio.squeeze().numpy() * 32767).astype("int16"))
print("Saved test_input.wav")

# Test 1: Standard quantized encode/decode (should work)
print("\n=== Test 1: Quantized encode/decode ===")
with torch.no_grad():
    encoder_out = mimi.encode(audio, num_quantizers=mimi.config.num_quantizers)
    codes = encoder_out.audio_codes
    print(f"Codes: shape={codes.shape}")

    decoded_quant = mimi.decode(codes).audio_values
    print(
        f"Decoded (quantized): shape={decoded_quant.shape}, range=[{decoded_quant.min():.3f}, {decoded_quant.max():.3f}]"
    )

# Save quantized output
decoded_quant_np = decoded_quant.squeeze().numpy()[: audio.shape[-1]]
scipy.io.wavfile.write("test_output_quantized.wav", sr, (decoded_quant_np * 32767).astype("int16"))
print("Saved test_output_quantized.wav")

# Test 2: Continuous latent encode/decode (our approach with projection scaling)
print("\n=== Test 2: Continuous latent encode/decode (with projection scaling) ===")
with torch.no_grad():
    # Encode to continuous latents
    emb = mimi.encoder(audio)
    emb = emb.transpose(1, 2)
    enc_out = mimi.encoder_transformer(emb)
    emb = enc_out.last_hidden_state if hasattr(enc_out, "last_hidden_state") else enc_out[0]
    emb = emb.transpose(1, 2)
    latents = mimi.downsample(emb)
    print(
        f"Continuous latents: shape={latents.shape}, range=[{latents.min():.3f}, {latents.max():.3f}], std={latents.std():.3f}"
    )

    # Apply quantizer projection scaling (key fix!)
    # The RVQ has input_proj (512→256) and output_proj (256→512) that scale latents.
    sem_q = mimi.quantizer.semantic_residual_vector_quantizer
    acou_q = mimi.quantizer.acoustic_residual_vector_quantizer

    sem_256 = sem_q.input_proj(latents) if sem_q.input_proj else latents
    sem_512 = sem_q.output_proj(sem_256) if sem_q.output_proj else sem_256

    acou_256 = acou_q.input_proj(latents) if acou_q.input_proj else latents
    acou_512 = acou_q.output_proj(acou_256) if acou_q.output_proj else acou_256

    scaled_latents = sem_512 + acou_512
    print(
        f"Scaled latents: shape={scaled_latents.shape}, range=[{scaled_latents.min():.3f}, {scaled_latents.max():.3f}], std={scaled_latents.std():.3f}"
    )

    # Decode from scaled latents
    emb = mimi.upsample(scaled_latents)
    emb = emb.transpose(1, 2)
    dec_out = mimi.decoder_transformer(emb)
    emb = dec_out.last_hidden_state if hasattr(dec_out, "last_hidden_state") else dec_out[0]
    emb = emb.transpose(1, 2)
    decoded = mimi.decoder(emb)
    print(
        f"Decoded (continuous): shape={decoded.shape}, range=[{decoded.min():.3f}, {decoded.max():.3f}]"
    )

# Save output
decoded_np = decoded.squeeze().numpy()
# Trim to match input length
decoded_np = decoded_np[: audio.shape[-1]]
scipy.io.wavfile.write("test_output.wav", sr, (decoded_np * 32767).astype("int16"))
print("Saved test_output.wav")

print("\nCompare test_input.wav and test_output.wav!")
