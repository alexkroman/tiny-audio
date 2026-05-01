# TinyAudio (Swift)

Fully on-device speech-to-text for iOS / macOS / visionOS, using the tiny-audio MLX model.

## Requirements

- Xcode 15.3+ (for Swift Testing) and Swift 6.0+
- iOS 17 / macOS 14 / visionOS 1+
- Apple Silicon recommended for full performance

## Install

In your `Package.swift`:

```swift
.package(url: "https://github.com/<owner>/tiny-audio.git", from: "0.1.0"),

// In your target dependencies:
.product(name: "TinyAudio", package: "tiny-audio"),
```

> Note: this package lives under `swift/` in the `tiny-audio` repo. SwiftPM
> resolves the manifest at that path automatically.

## Quickstart — file transcription

```swift
import TinyAudio

let transcriber = try await Transcriber.load { progress in
    print("Downloading model: \(Int(progress.fractionCompleted * 100))%")
}

let url = URL(fileURLWithPath: "/path/to/audio.wav")
let text = try await transcriber.transcribe(.file(url))
print(text)
```

## Streaming text deltas

```swift
for try await delta in transcriber.transcribeStream(.file(url)) {
    print(delta, terminator: "")
}
```

## In-memory PCM buffers

```swift
import AVFoundation

let buffer: AVAudioPCMBuffer = ...   // any sample rate / channel layout
let text = try await transcriber.transcribe(.pcm(buffer: buffer))
```

## Raw float samples

```swift
let samples: [Float] = ...           // mono Float32 at any sample rate
let text = try await transcriber.transcribe(.samples(samples, sampleRate: 16_000))
```

## What ships in v0.1

- `Transcriber` — async actor for file/buffer/sample transcription.
- `AudioInput` — file URL, `AVAudioPCMBuffer`, or raw `[Float]` samples.
- `TranscriptionOptions` — `maxNewTokens`, optional system prompt.
- `WeightSource` — default Hub bundle, custom Hub repo, or local directory.
- `TinyAudioError` — typed errors for download / verification / runtime failures.

Live-microphone transcription with VAD endpointing ships in **v0.2**.

## Model details

- ~460 MB total (4-bit quantized GLM-ASR encoder + Qwen3-0.6B decoder, fp16 projector).
- Downloaded on first `Transcriber.load()` call from the Hugging Face Hub repo
  `mazesmazes/tiny-audio-mlx` into the standard HF cache.
- Subsequent loads are offline (manifest SHA256-verified once, then trusted).
- Greedy decoding only (no sampling, no beam search). Optimised for ASR transcription.

## Architecture

```
AudioInput → AudioDecoder → 16 kHz mono Float32
                               ↓
                       LogMelSpectrogram (vDSP-backed via MLXAudioCore)
                               ↓
                       GLMASREncoder (4-bit, mlx-audio-swift)
                               ↓
                       MLPProjector (fp16, our trained weights)
                               ↓
                       Audio embedding splice into Qwen3 prompt embeddings
                               ↓
                       Vendored Qwen3 decoder (4-bit, from mlx-swift-lm)
                               ↓
                       Greedy decode → text
```

The encoder and decoder are reused from upstream Swift packages
(`mlx-audio-swift` and a vendored copy of `mlx-swift-lm`'s Qwen3 patched to
accept input embeddings). The trained projector that bridges audio→text is
the only model-specific component.

## Running tests

```bash
cd swift
swift test
```

The end-to-end token-ID parity test against the Python reference is gated
behind an environment variable (downloads ~460 MB on first run):

```bash
TINY_AUDIO_E2E=1 swift test --filter E2ETokenID
```

## License

[Match the project's license — likely MIT / Apache 2.0.]
