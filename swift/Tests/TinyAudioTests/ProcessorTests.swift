// swift/Tests/TinyAudioTests/ProcessorTests.swift
import Foundation
import Hub
import MLX
import Testing
import Tokenizers

@testable import TinyAudio

@Suite("Processor")
struct ProcessorTests {
  @Test func promptIdsMatchPythonReference() async throws {
    guard let bundle = BundleResolver.locate() else {
      print("Skipping ProcessorTests: tiny-audio-mlx bundle not cached.")
      return
    }

    // Read reference: audio token id + expected input_ids.
    let refURL = Bundle.module.url(
      forResource: "reference_prompt_token_ids",
      withExtension: "json",
      subdirectory: "Fixtures"
    )!
    let refData = try Data(contentsOf: refURL)
    let ref = try JSONSerialization.jsonObject(with: refData) as! [String: Any]
    let refInputIds = ref["input_ids"] as! [Int]
    let audioTokenId = ref["audio_token_id"] as! Int

    // The bundle tokenizer is the base Qwen3 tokenizer. The Python training
    // pipeline adds <audio> as a special token (via ASRModel._init_tokenizer),
    // which gets assigned the next available ID (151669 for Qwen3-0.6B).
    // We replicate that here by patching tokenizer.json with the <audio> entry
    // before constructing PreTrainedTokenizer, matching Python's add_tokens(['<audio>']).
    let tokenizer = try makeTokenizerWithAudioToken(
      bundle: bundle,
      audioToken: "<audio>",
      audioTokenId: audioTokenId
    )

    // Count <audio> tokens in the reference. Whatever number the Python
    // path used, we use the same to drive Swift's prompt build.
    let numAudio = refInputIds.filter { $0 == audioTokenId }.count

    let ids = try Processor.buildPromptInputIds(
      tokenizer: tokenizer,
      numAudioTokens: numAudio,
      systemPrompt: nil
    )
    let idsFlat = ids.asArray(Int32.self).map { Int($0) }

    #expect(idsFlat == refInputIds, "Swift prompt token IDs differ from Python reference")
  }

  /// Verify the algebraic identity that Task 5's prefix KV cache reuse will
  /// rely on: `buildPromptParts(...).prefixIds + [audioId]×N + .suffixIds`
  /// must equal `buildPromptInputIds(numAudioTokens: N, ...)` for any N.
  ///
  /// Gated on `TINY_AUDIO_E2E=1` because it needs the full bundled tokenizer;
  /// the existing `promptIdsMatchPythonReference` test covers the byte-parity
  /// invariant against the Python reference, so this one only adds value
  /// when actually run.
  @Test func promptPartsConcatToFullPrompt() async throws {
    guard ProcessInfo.processInfo.environment["TINY_AUDIO_E2E"] == "1" else {
      print("Skipping: set TINY_AUDIO_E2E=1 to exercise prefix/suffix concat parity.")
      return
    }
    guard let bundle = BundleResolver.locate() else {
      print("Skipping ProcessorTests: tiny-audio-mlx bundle not cached.")
      return
    }

    // Use a fixed test-side audio token ID; we control the patched
    // tokenizer.json so the value just has to match what we then look up.
    let audioTokenId = 151669
    let tokenizer = try makeTokenizerWithAudioToken(
      bundle: bundle,
      audioToken: Processor.audioToken,
      audioTokenId: audioTokenId
    )

    let parts = Processor.buildPromptParts(tokenizer: tokenizer, systemPrompt: nil)
    let n = 7
    let combined = try Processor.buildPromptInputIds(
      tokenizer: tokenizer, numAudioTokens: n, systemPrompt: nil)
    let combinedIds = combined.asArray(Int32.self)

    let audioId = Int32(tokenizer.convertTokenToId(Processor.audioToken)!)
    let expected = parts.prefixIds + [Int32](repeating: audioId, count: n) + parts.suffixIds
    #expect(combinedIds == expected)
  }

  /// Build a `PreTrainedTokenizer` from the bundle's tokenizer files, patching
  /// `added_tokens` in `tokenizer.json` to include `<audio>` so it is treated as
  /// a single token (matching Python's `tokenizer.add_tokens(['<audio>'])`).
  private func makeTokenizerWithAudioToken(
    bundle: URL,
    audioToken: String,
    audioTokenId: Int
  ) throws -> any Tokenizer {
    // Load tokenizer_config.json as raw dict for Config construction.
    let configURL = bundle.appendingPathComponent("tokenizer_config.json")
    let configData = try Data(contentsOf: configURL)
    let configDict = try JSONSerialization.jsonObject(with: configData) as! [String: Any]
    let tokenizerConfig = Config(configDict as [NSString: Any])

    // Load tokenizer.json and patch added_tokens.
    let tokenizerURL = bundle.appendingPathComponent("tokenizer.json")
    let tokenizerRaw = try Data(contentsOf: tokenizerURL)
    var tokenizerDict = try JSONSerialization.jsonObject(with: tokenizerRaw) as! [String: Any]

    var addedTokens = tokenizerDict["added_tokens"] as! [[String: Any]]
    let alreadyAdded = addedTokens.contains { ($0["content"] as? String) == audioToken }
    if !alreadyAdded {
      addedTokens.append(
        [
          "id": audioTokenId,
          "content": audioToken,
          "single_word": false,
          "lstrip": false,
          "rstrip": false,
          "normalized": false,
          "special": true,
        ] as [String: Any])
      tokenizerDict["added_tokens"] = addedTokens
    }

    let tokenizerData = Config(tokenizerDict as [NSString: Any])
    return try PreTrainedTokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
  }
}
