// Byte-for-byte parity between the compiled per-token decode step and the
// uncompiled fallback (`TINY_AUDIO_NO_COMPILE=1`).
//
// Gated behind `TINY_AUDIO_E2E=1` because (a) it loads the full SDK model
// bundle, which the test target's `Bundle.module` cannot reach (the
// `Resources/Model` directory is owned by the SDK target), and (b) it
// runs `Transcriber.load()` twice plus two transcribes — slow for every
// CI build.
//
// Run with:
//   TINY_AUDIO_E2E=1 swift test --package-path swift --filter CompiledStep

import Foundation
import Testing

@testable import TinyAudio

@Suite("CompiledStep")
struct CompiledStepTests {
  /// Decode the same audio with the compiled per-token step vs. the
  /// uncompiled fallback (`TINY_AUDIO_NO_COMPILE=1`). Both paths must
  /// produce identical transcripts.
  @Test func compiledStepMatchesUncompiled() async throws {
    guard ProcessInfo.processInfo.environment["TINY_AUDIO_E2E"] == "1" else {
      print("Skipping: set TINY_AUDIO_E2E=1 to exercise compiled vs uncompiled parity.")
      return
    }
    let url = try #require(
      Bundle.module.url(
        forResource: "librispeech_sample", withExtension: "wav", subdirectory: "Fixtures"))

    setenv("TINY_AUDIO_NO_COMPILE", "1", 1)
    let uncompiled = try await Transcriber.load()
    let uncompiledText = try await uncompiled.transcribe(.file(url))

    unsetenv("TINY_AUDIO_NO_COMPILE")
    let compiled = try await Transcriber.load()
    let compiledText = try await compiled.transcribe(.file(url))

    #expect(
      compiledText == uncompiledText,
      "compiled per-token step diverged: '\(compiledText)' vs '\(uncompiledText)'")
  }
}
