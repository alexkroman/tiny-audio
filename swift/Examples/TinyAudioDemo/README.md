# TinyAudioDemo

Minimal SwiftUI demo of the TinyAudio Swift SDK, runnable on macOS and iOS.

## macOS

```bash
swift run --package-path swift/Examples/TinyAudioDemo TinyAudioDemo
```

## iOS Simulator

Open `swift/` in Xcode (it picks up `swift/TinyAudio.xcworkspace`), pick the
`TinyAudioDemo_iOS` scheme and an iPhone simulator, and Run (⌘R). The workspace
also exposes `TinyAudioDemo_macOS` and the SDK schemes
(`TinyAudio`, `tiny-audio-swift-eval`, `tiny-audio-vad-bench`).

You can also open `swift/Examples/TinyAudioDemo/TinyAudioDemo.xcodeproj`
directly if you only want the demo.

If the project file is missing or stale, regenerate with:

```bash
brew install xcodegen
cd swift/Examples/TinyAudioDemo
xcodegen
```

## What it shows

- `Transcriber.load()` with progress callback wired into `ProgressView`.
- `MicrophoneTranscriber` for live mic capture with Silero VAD endpointing.
- `MicrophoneTranscriber.events` consumed via SwiftUI `@Published` state.

## Microphone permission

The app declares `NSMicrophoneUsageDescription` in `Info.plist`. The system
permission prompt fires on first `mic.start()`. On the iOS Simulator, grant
access in Settings if prompted.

## Caveat

The TinyAudio SDK bundles ~675 MB of model weights as Git LFS resources. This
makes the demo's binary size much larger than a typical app. For App Store
distribution, switch the demo's `Transcriber.load(from:)` argument from
`.defaultHub` (bundled) to `.hub(repoID:revision:)` (download on first launch).
