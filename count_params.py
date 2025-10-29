from src.asr_modeling import ASRModel
from src.asr_config import ASRConfig

print("Loading Tiny Audio model...")
config = ASRConfig.from_pretrained("mazesmazes/tiny-audio", trust_remote_code=True)
model = ASRModel.from_pretrained("mazesmazes/tiny-audio", config=config)
print("✓ Model loaded!\n")

def count_params(module, name):
    """Count total and trainable parameters in a module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    frozen = total - trainable
    percent = 100 * trainable / total if total > 0 else 0

    print(f"{name}")
    print(f"{'='*50}")
    print(f"  Total params:      {total:>15,}")
    print(f"  Trainable params:  {trainable:>15,}")
    print(f"  Frozen params:     {frozen:>15,}")
    print(f"  Trainable:         {percent:>14.2f}%")
    print()

# Count by component
count_params(model.encoder, "ENCODER (HuBERT + LoRA)")
count_params(model.projector, "PROJECTOR (SwiGLU MLP)")
count_params(model.decoder, "DECODER (SmolLM3 + LoRA)")

# Overall
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("="*50)
print("OVERALL MODEL")
print("="*50)
print(f"  Total params:      {total:>15,}")
print(f"  Trainable params:  {trainable:>15,}")
print(f"  Frozen params:     {total - trainable:>15,}")
print(f"  Efficiency:        {100 * trainable / total:>14.2f}%")
print("\n✓ We train only 3.2% of the total parameters!")
