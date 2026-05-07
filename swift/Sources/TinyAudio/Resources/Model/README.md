# TinyAudio model bundle

This directory holds the runtime bundle the Swift SDK ships:

- `encoder.safetensors` — quantized GLM-ASR audio encoder
- `projector.safetensors` — trained MLP projector
- `decoder.safetensors` — MLX-quantized LM decoder
- `decoder_config.json`, `tokenizer.json`, `tokenizer_config.json` — decoder metadata
- `config.json` — bundle layout (encoder dims, projector pool stride, audio token id, …)
- `manifest.json` — sha256 + size for every other file in this dir

These artifacts are **not** tracked in git. Generate them locally before
building the Swift package:

```bash
poetry install
poetry run ta mlx build-bundle --projector mazesmazes/tiny-audio-embedded-2
```

`build-bundle` pulls the projector + (when needed) auto-runs
`convert-decoder` to populate this directory. SwiftPM's
`.copy("Resources/Model")` rule in `Package.swift` then ships the bundle
into the SDK at build time.
