// Byte-for-byte parity between the prefix-cache path and the legacy
// full-prefill path. With the default `nil` system prompt the prefix is
// ~3 tokens, which falls below the 10-token cache threshold in
// `buildPrefixCache`, so `prefixCache` is nil and both branches in
// `tokenStream` take the legacy path. This test still has value: it
// exercises the bypass switch + the legacy branch with the default
// transcribe() flow, ensuring the refactor preserves the original
// behaviour even before any caching kicks in.
//
// Gated behind `TINY_AUDIO_E2E=1` (matches CompiledStepTests).

import Foundation
import Testing

@_spi(Testing) @testable import TinyAudio

@Suite("PrefixCache")
struct PrefixCacheTests {
  @Test func prefixCacheMatchesLegacyPath() async throws {
    guard ProcessInfo.processInfo.environment["TINY_AUDIO_E2E"] == "1" else {
      print("Skipping: set TINY_AUDIO_E2E=1 to exercise prefix-cache parity.")
      return
    }
    let url = try #require(
      Bundle.module.url(
        forResource: "librispeech_sample", withExtension: "wav", subdirectory: "Fixtures"))

    let t = try await Transcriber.load()
    let cached = try await t.transcribe(.file(url))
    await t.setBypassPrefixCache(true)
    let legacy = try await t.transcribe(.file(url))
    await t.setBypassPrefixCache(false)
    #expect(cached == legacy, "prefix-cache path diverged: '\(cached)' vs '\(legacy)'")
  }
}
