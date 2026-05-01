# Changelog

All notable changes to the TinyAudio Swift SDK are documented here.

## [1.0.0] — 2026-04-30

### Added

- DocC catalog at `Sources/TinyAudio/Documentation.docc/TinyAudio.md` with overview + quickstart.
- DocC `///` documentation on all 7 public types.
- WER parity report (`swift/docs/wer-parity.md`): Swift SDK 9.93% vs Python MLX 13.88% on a 200-sample loquacious slice.
- Example SwiftUI app at `swift/Examples/TinyAudioDemo/` showing live-mic transcription.

### Performance

- Pre-built audio-token logit mask: removes per-decode-step `[Float]` allocation.
- Incremental tokenizer detokenization in `transcribeStream`: removes O(N²) full-prefix decode.
- Multi-shape encoder warmup: pre-JITs kernels for 1 s / 5 s / 15 s audio durations.

## [0.2.0] — 2026-04-30

### Added

- `MicrophoneTranscriber` actor for live-microphone transcription.
- `VADConfig` public struct for tuning endpointing thresholds.
- Silero VAD bundled as a Core ML resource (~600 KB).
- `MicrophoneTranscriber.Event` (`.partial` / `.final` / `.error`) emitted via non-throwing `AsyncStream`.

## [0.1.0] — 2026-04-30

### Added

- `Transcriber` actor for file / buffer / sample transcription.
- `Transcriber.transcribeStream(_:options:)` for incremental text deltas.
- `AudioInput`, `TranscriptionOptions`, `WeightSource`, `TinyAudioError` public types.
- Hub-based weight loading via `swift-transformers` `Hub.snapshot`.
- Token-ID parity test (gated `TINY_AUDIO_E2E=1`) confirming bit-exact match against the Python MLX reference for the first 10 tokens of the librispeech sample.
- 4-bit quantized GLM-ASR encoder (mlx-audio-swift's `WhisperEncoder`).
- Vendored Qwen3 from `mlx-swift-lm` patched to expose `inputEmbeddings`.
- fp16 MLP projector with trained weights from the tiny-audio checkpoint.
