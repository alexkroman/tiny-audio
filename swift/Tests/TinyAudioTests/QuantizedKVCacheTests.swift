import Foundation
import MLX
import Testing

@testable import TinyAudio

@Suite("QuantizedKVCache")
struct QuantizedKVCacheTests {
  /// Decode the same audio with and without 8-bit KV cache quantization;
  /// the produced transcripts must match.
  ///
  /// Skipped unless the bundled model is present (same gating used by
  /// EncoderParityTests).
  @Test func quantizedCacheMatchesSimpleCache() async throws {
    guard Bundle.module.url(forResource: "Model", withExtension: nil) != nil else {
      print("Skipping: bundled Model not present")
      return
    }
    let url = try #require(
      Bundle.module.url(
        forResource: "librispeech_sample", withExtension: "wav", subdirectory: "Fixtures"))

    setenv("TINY_AUDIO_KV_BITS", "", 1)  // ensure default OFF
    let baseline = try await Transcriber.load()
    let baselineText = try await baseline.transcribe(.file(url))

    setenv("TINY_AUDIO_KV_BITS", "8", 1)
    let quantised = try await Transcriber.load()
    let quantisedText = try await quantised.transcribe(.file(url))

    setenv("TINY_AUDIO_KV_BITS", "", 1)  // reset

    #expect(
      baselineText == quantisedText,
      "8-bit KV cache changed transcription: '\(baselineText)' vs '\(quantisedText)'")
  }
}
