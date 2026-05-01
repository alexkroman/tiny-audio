// swift/Tests/TinyAudioTests/MicrophoneTranscriberTests.swift
import Foundation
import Testing

@testable import TinyAudio

@Suite("MicrophoneTranscriber")
struct MicrophoneTranscriberTests {
  /// Exercises construction end-to-end (Transcriber + MicrophoneTranscriber) when
  /// TINY_AUDIO_E2E=1 (bundle already cached from prior E2E runs). We intentionally
  /// never call `start()`, so `AVAudioApplication.requestRecordPermission` is never
  /// triggered. This catches resource-bundling regressions for the Silero VAD model.
  ///
  /// Without TINY_AUDIO_E2E=1, falls back to a compile-time API-shape check.
  @Test func initSucceedsWithoutMicPermissionPrompt() async throws {
    guard ProcessInfo.processInfo.environment["TINY_AUDIO_E2E"] == "1" else {
      // API-shape sanity check: confirm public Event cases are reachable.
      _ = MicrophoneTranscriber.Event.final(utteranceID: UUID(), text: "x")
      print(
        "Skipping full init: set TINY_AUDIO_E2E=1 to exercise MicrophoneTranscriber construction.")
      return
    }

    let transcriber = try await Transcriber.load()
    let mic = try MicrophoneTranscriber(transcriber: transcriber)
    // Confirm the events stream is constructed and consumable. No start() call.
    let stream = mic.events
    _ = stream  // silence unused-variable warning
  }

  /// Live mic flow. Gated behind TINY_AUDIO_MIC=1.
  /// Run with:
  ///   TINY_AUDIO_MIC=1 swift test --package-path swift --filter MicrophoneTranscriber
  /// Then speak. The test will collect events for ~10 seconds and print them.
  @Test func liveMicTranscribesUtterance() async throws {
    guard ProcessInfo.processInfo.environment["TINY_AUDIO_MIC"] == "1" else {
      print("Skipping liveMicTranscribesUtterance: set TINY_AUDIO_MIC=1 to run.")
      return
    }

    let transcriber = try await Transcriber.load()
    let mic = try MicrophoneTranscriber(transcriber: transcriber)

    print("\n=== Speak now (10 s window) ===\n")
    try await mic.start()

    // Drain events for 10 s, printing as they arrive.
    let consumer = Task {
      for await ev in mic.events {
        switch ev {
        case .final(_, let text):
          print("\n[FINAL] \(text)")
        case .error(let e):
          print("\n[ERROR] \(e)")
        }
      }
      print("\n[stream finished]")
    }

    try await Task.sleep(nanoseconds: 10 * 1_000_000_000)
    await mic.stop()
    await consumer.value

    print("\n=== Done ===\n")
  }
}
