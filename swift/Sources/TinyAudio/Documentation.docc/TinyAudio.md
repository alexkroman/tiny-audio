# ``TinyAudio``

Fully on-device speech-to-text for iOS, macOS, and visionOS, using a 4-bit quantized
GLM-ASR encoder + Qwen3-0.6B decoder via Apple's MLX framework.

## Overview

TinyAudio loads a ~460 MB MLX model (downloaded once on first call), then runs
greedy-decoded transcription on audio inputs — files, PCM buffers, raw float
samples, or live microphone capture.

The headline entry points:

- ``Transcriber/load(from:progress:)`` — async actor construction. Downloads,
  verifies, and loads the model.
- ``Transcriber/transcribe(_:options:)`` — single-shot file/buffer transcription
  returning a `String`.
- ``Transcriber/transcribeStream(_:options:)`` — `AsyncThrowingStream<String,
  Error>` of incremental text deltas.
- ``MicrophoneTranscriber/start()`` — live-mic transcription with Silero VAD
  endpointing. Per-utterance results stream as ``MicrophoneTranscriber/Event``.

## Quickstart

```swift
import TinyAudio

let transcriber = try await Transcriber.load { progress in
    print("Loading: \(Int(progress.fractionCompleted * 100))%")
}

let url = URL(fileURLWithPath: "audio.wav")
let text = try await transcriber.transcribe(.file(url))
print(text)
```

For live microphone:

```swift
let mic = try MicrophoneTranscriber(transcriber: transcriber)
try await mic.start()
for await event in mic.events {
    if case let .partial(_, delta) = event {
        print(delta, terminator: "")
    }
}
```

## Topics

### Transcribing files and buffers

- ``Transcriber``
- ``AudioInput``
- ``TranscriptionOptions``

### Live microphone

- ``MicrophoneTranscriber``
- ``VADConfig``

### Loading

- ``WeightSource``

### Errors

- ``TinyAudioError``
- ``AnyError``
