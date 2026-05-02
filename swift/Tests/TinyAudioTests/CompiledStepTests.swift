import Foundation
import Testing

@testable import TinyAudio

@Suite("CompiledStep")
struct CompiledStepTests {
  /// Decode the same audio with the compiled per-token step vs. the
  /// uncompiled fallback (TINY_AUDIO_NO_COMPILE=1). Both paths must
  /// produce identical transcripts.
  ///
  /// Skipped unless the bundled model is present.
  @Test func compiledStepMatchesUncompiled() async throws {
    guard Bundle.module.url(forResource: "Model", withExtension: nil) != nil else {
      print("Skipping: bundled Model not present")
      return
    }
    let url = try #require(
      Bundle.module.url(
        forResource: "librispeech_sample", withExtension: "wav", subdirectory: "Fixtures"))

    setenv("TINY_AUDIO_NO_COMPILE", "1", 1)
    let uncompiled = try await Transcriber.load()
    let uncompiledText = try await uncompiled.transcribe(.file(url))

    setenv("TINY_AUDIO_NO_COMPILE", "0", 1)
    let compiled = try await Transcriber.load()
    let compiledText = try await compiled.transcribe(.file(url))

    setenv("TINY_AUDIO_NO_COMPILE", "0", 1)  // reset
    #expect(
      compiledText == uncompiledText,
      "compiled per-token step diverged: '\(compiledText)' vs '\(uncompiledText)'")
  }
}
