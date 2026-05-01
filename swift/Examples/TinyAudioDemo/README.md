# TinyAudioDemo

Minimal SwiftUI demo of the TinyAudio Swift SDK with live-microphone transcription.

## Run

```bash
cd swift/Examples/TinyAudioDemo
swift run TinyAudioDemo
```

The first launch downloads the model (~460 MB) into the standard HuggingFace
cache. Subsequent launches are offline.

## What it shows

- `Transcriber.load()` with progress callback wired into `ProgressView`.
- `MicrophoneTranscriber` for live mic capture with Silero VAD endpointing.
- `MicrophoneTranscriber.events` consumed via SwiftUI `@Published` state.

## Microphone permission

For an Xcode-built app, declare `NSMicrophoneUsageDescription` in `Info.plist`.
The `swift run` path on macOS doesn't enforce sandbox; the system permission
prompt fires on first `mic.start()`.
