from transformers import AutoModel

model = AutoModel.from_pretrained(
    "mazesmazes/tiny-audio",
    trust_remote_code=True
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

# Explore structure
print("\nModel structure:")
print(model)
