from transformers import AutoConfig

config = AutoConfig.from_pretrained(
    "mazesmazes/tiny-audio",
    trust_remote_code=True
)

print(f"Audio encoder: {config.audio_model_id}")
print(f"Language model: {config.text_model_id}")
print(f"Encoder dimension: {config.encoder_dim}")
print(f"LLM dimension: {config.llm_dim}")
print(f"Downsampling rate: {config.audio_downsample_rate}x")
