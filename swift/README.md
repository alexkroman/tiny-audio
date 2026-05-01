# TinyAudio (Swift)

Fully on-device speech-to-text for iOS / macOS / visionOS, using the tiny-audio MLX model.

## What ships in v1.0

- File / buffer / sample transcription via `Transcriber`
- Streaming text deltas via `transcribeStream`
- Live mic + Silero VAD endpointing via `MicrophoneTranscriber`
- Public API surface: 7 types, fully documented via DocC
- Quality gates: token-ID parity test, WER parity vs Python (see `swift/docs/wer-parity.md`)

## Requirements

- Xcode 15.3+ (for Swift Testing) and Swift 6.0+
- iOS 17 / macOS 14 / visionOS 1+
- Apple Silicon recommended for full performance

## Install

In your `Package.swift`:

```swift
.package(url: "https://github.com/<owner>/tiny-audio.git", from: "1.0.0"),

// In your target dependencies:
.product(name: "TinyAudio", package: "tiny-audio"),
```

> Note: this package lives under `swift/` in the `tiny-audio` repo. SwiftPM
> resolves the manifest at that path automatically.

> **Disk size**: the SDK is ~675 MB because model weights (~663 MB of safetensors) are
> bundled via Git LFS. Make sure `git-lfs` is installed before cloning so the blobs
> are resolved (`brew install git-lfs && git lfs install`).

## Quickstart — file transcription

```swift
import TinyAudio

// No download required — model weights are bundled with the SDK.
let transcriber = try await Transcriber.load()

let url = URL(fileURLWithPath: "/path/to/audio.wav")
let text = try await transcriber.transcribe(.file(url))
print(text)
```

To use a different model revision from the HuggingFace Hub:

```swift
let transcriber = try await Transcriber.load(from: .hub(repoID: "mazesmazes/tiny-audio-mlx", revision: nil)) { progress in
    print("Downloading model: \(Int(progress.fractionCompleted * 100))%")
}
```

## Streaming text deltas

```swift
for try await delta in transcriber.transcribeStream(.file(url)) {
    print(delta, terminator: "")
}
```

## Live microphone transcription

```swift
import TinyAudio

let transcriber = try await Transcriber.load()
let mic = try MicrophoneTranscriber(transcriber: transcriber)

try await mic.start()
for await event in mic.events {
    switch event {
    case let .partial(_, delta):
        print(delta, terminator: "")
    case let .final(id, text):
        print("\n[utterance \(id): \(text)]")
    case let .error(error):
        print("\n[error: \(error)]")
    }
}
```

The `MicrophoneTranscriber` actor uses `AVAudioEngine` to capture the system mic, runs Silero VAD (on-device, ~600 KB Core ML model bundled with the SDK) to chop the stream into utterances, and routes each utterance through the `Transcriber` model. The `events` stream is non-throwing — per-utterance failures emit `.error(_)` events but the session continues until you call `stop()`.

### Permission

The host app must declare `NSMicrophoneUsageDescription` in its `Info.plist`. `MicrophoneTranscriber.start()` triggers the standard permission prompt; if the user denies, it throws `TinyAudioError.micPermissionDenied`.

### Tuning the VAD

`MicrophoneTranscriber(transcriber:vad:)` accepts an optional `VADConfig`:

```swift
var config = VADConfig.default
config.minSilenceDurationMs = 800   // wait longer before declaring offset
config.preSpeechPaddingMs = 300     // capture more lead-in audio
let mic = try MicrophoneTranscriber(transcriber: transcriber, vad: config)
```

Defaults: 0.5 speech threshold, 500 ms minimum silence to end an utterance, 200 ms minimum speech to start one, 200 ms of pre-speech audio captured before the detected onset.

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

## Public API surface

| Type | Description |
|------|-------------|
| `Transcriber` | Async actor for file / buffer / sample transcription |
| `MicrophoneTranscriber` | Async actor for live-mic transcription with Silero VAD |
| `AudioInput` | File URL, `AVAudioPCMBuffer`, or raw `[Float]` samples |
| `TranscriptionOptions` | `maxNewTokens`, optional system prompt |
| `WeightSource` | Default Hub bundle, custom Hub repo, or local directory |
| `VADConfig` | Tuning knobs for Silero VAD endpointing thresholds |
| `TinyAudioError` | Typed errors for download / verification / runtime failures |

## Model details

- ~675 MB total on disk (4-bit quantized GLM-ASR encoder + Qwen3-0.6B decoder, fp16 projector),
  bundled as Git LFS resources inside the Swift package.
- `Transcriber.load(from: .defaultHub)` reads directly from `Bundle.module` — no network
  request on first launch. Use `.hub(repoID:revision:)` to pull a different revision.
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

## Example app

A minimal SwiftUI demo lives at `swift/Examples/TinyAudioDemo/`. Run it with:

```bash
swift run --package-path swift/Examples/TinyAudioDemo TinyAudioDemo
```

It demonstrates `Transcriber.load()` with progress callback and `MicrophoneTranscriber` live-mic streaming through `@Published` SwiftUI state.

## Documentation

Full API documentation is generated from the source as DocC. To build locally:

```bash
swift package --package-path swift generate-documentation --target TinyAudio
```

(Requires `swift-docc-plugin`; install via Package.swift if not already present.)

## WER parity

The Swift SDK was validated against the Python MLX reference on a 200-sample slice of loquacious:

| Backend | WER % |
|---------|-------|
| Python MLX | 13.88 |
| Swift SDK | 9.93 |

The Swift path achieves lower WER (delta = −3.95 pp), driven by the Python path hallucinating on a small number of short clips. Full methodology, per-sample breakdown, and reproduction steps are in [`swift/docs/wer-parity.md`](docs/wer-parity.md).

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
