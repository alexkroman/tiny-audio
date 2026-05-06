import Foundation
import Hub
import MLX
import MLXNN
import Tokenizers

/// Text-only chat session that runs against an already-loaded Qwen3 decoder.
///
/// `ChatSession` is the text-generation counterpart to ``Transcriber``: it
/// reuses the same `Qwen3Model` weights + tokenizer that power audio
/// transcription, but skips the encoder, projector, and audio-embedding
/// splice. It exists so callers (e.g. the cooking demo) can run a structured
/// intent classifier or other text-only chat without paying the cost of
/// loading a second backbone.
///
/// The actor isolates the underlying MLX model state. The decode loop runs on
/// a detached task so long generations don't pin the actor's executor.
public actor ChatSession {
  private let decoder: Qwen3Model
  private let tokenizer: any Tokenizer
  private let numDecoderLayers: Int
  private let eosTokenIds: Set<Int32>

  internal init(
    decoder: Qwen3Model,
    tokenizer: any Tokenizer,
    numDecoderLayers: Int,
    eosTokenIds: Set<Int32>
  ) {
    self.decoder = decoder
    self.tokenizer = tokenizer
    self.numDecoderLayers = numDecoderLayers
    self.eosTokenIds = eosTokenIds
  }

  /// Generate a completion for `prompt` using the bundled Qwen3 chat template.
  ///
  /// The prompt is wrapped as a single user-role message and rendered through
  /// the tokenizer's chat template (the one baked into `tokenizer.json`),
  /// producing token IDs that match Qwen3's training-time chat format.
  ///
  /// Decoding is greedy (argmax). The loop terminates when any EOS token
  /// fires or after `maxNewTokens` steps, whichever comes first. EOS tokens
  /// are NOT included in the returned string.
  ///
  /// - Parameters:
  ///   - prompt: The user message. Whitespace-only prompts are rejected with
  ///     ``TinyAudioError/promptEmpty``.
  ///   - maxNewTokens: Maximum number of tokens to generate. Must be > 0;
  ///     otherwise ``TinyAudioError/invalidArgument(reason:)`` is thrown.
  /// - Returns: The detokenized completion (EOS stripped).
  /// - Throws: ``TinyAudioError/promptEmpty``,
  ///   ``TinyAudioError/invalidArgument(reason:)``, or
  ///   ``TinyAudioError/mlxModuleLoadFailed(name:underlying:)`` if the chat
  ///   template fails to render.
  public func chat(prompt: String, maxNewTokens: Int = 256) async throws -> String {
    let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { throw TinyAudioError.promptEmpty }
    guard maxNewTokens > 0 else {
      throw TinyAudioError.invalidArgument(reason: "maxNewTokens must be > 0")
    }

    let messages: [[String: String]] = [["role": "user", "content": trimmed]]
    let inputIds: [Int]
    do {
      inputIds = try tokenizer.applyChatTemplate(messages: messages)
    } catch {
      throw TinyAudioError.mlxModuleLoadFailed(name: "tokenizer", underlying: AnyError(error))
    }

    // The decoder is not Sendable; ferry it across the detached boundary in
    // an unchecked-Sendable box. Safe because the actor's properties are
    // immutable after init.
    let decoderBox = SendableBox(value: self.decoder)
    let numLayers = self.numDecoderLayers
    let eosTokenIds = self.eosTokenIds
    let inputIds32 = inputIds.map { Int32($0) }

    let generatedIds: [Int] = try await withCheckedThrowingContinuation { continuation in
      Task.detached {
        // Mirror Python's `finally: mx.clear_cache()` — release the Metal
        // pool regardless of how the loop exits.
        defer { Memory.clearCache() }

        let decoder = decoderBox.value
        let cache: [KVCache] = (0..<numLayers).map { _ in KVCacheSimple() }
        let inputs = MLXArray(inputIds32).expandedDimensions(axis: 0)
        let prefillLogits = decoder(inputs, cache: cache)
        var y = MLX.argMax(
          prefillLogits[0..., (-1)..., 0...],
          axis: -1
        ).expandedDimensions(axis: 0)
        MLX.asyncEval(y)

        // Pipelined greedy decode: dispatch step N+1 before syncing step N.
        var emitted: [Int] = []
        for n in 0..<maxNewTokens {
          var nextY: MLXArray? = nil
          if n < maxNewTokens - 1 {
            let nextLogits = decoder(y, cache: cache)
            nextY = MLX.argMax(
              nextLogits[0..., (-1)..., 0...],
              axis: -1
            ).expandedDimensions(axis: 0)
            MLX.asyncEval(nextY!)
          }
          let tid = y.item(Int32.self)
          if eosTokenIds.contains(tid) {
            continuation.resume(returning: emitted)
            return
          }
          emitted.append(Int(tid))
          if let next = nextY { y = next }
        }
        continuation.resume(returning: emitted)
      }
    }

    return tokenizer.decode(tokens: generatedIds)
  }
}

#if DEBUG
  extension ChatSession {
    /// Test-only constructor that loads a fresh Qwen3 decoder + tokenizer from
    /// the SDK's bundled model directory. The encoder/projector are skipped —
    /// text-only chat doesn't need them.
    ///
    /// Internal; not part of the public SDK surface.
    internal static func makeForTests(modelDirectory: URL) async throws -> ChatSession {
      let decoder: Qwen3Model
      let numDecoderLayers: Int
      do {
        let decoderConfigURL = modelDirectory.appendingPathComponent("decoder_config.json")
        let decoderConfigData = try Data(contentsOf: decoderConfigURL)
        let qwenConfig = try JSONDecoder().decode(Qwen3Configuration.self, from: decoderConfigData)
        let quantSpec = try? JSONDecoder().decode(
          ChatSessionDecoderQuantizationSpec.self, from: decoderConfigData)
        numDecoderLayers = qwenConfig.hiddenLayers
        decoder = Qwen3Model(qwenConfig)
        if let q = quantSpec?.quantization {
          quantize(model: decoder, groupSize: q.groupSize, bits: q.bits)
        }
        let rawWeights = Transcriber.castWeightsForCompute(
          try MLX.loadArrays(url: modelDirectory.appendingPathComponent("decoder.safetensors"))
        )
        let weights = decoder.sanitize(weights: rawWeights)
        try decoder.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
      } catch {
        throw TinyAudioError.mlxModuleLoadFailed(name: "decoder", underlying: AnyError(error))
      }

      // Text-only chat skips the <audio>-token patching that Transcriber.load
      // applies; this loader never emits the audio placeholder.
      let tokenizer: any Tokenizer
      do {
        let configURL = modelDirectory.appendingPathComponent("tokenizer_config.json")
        let configData = try Data(contentsOf: configURL)
        guard let configDict = try JSONSerialization.jsonObject(with: configData) as? [String: Any]
        else {
          throw TinyAudioError.mlxModuleLoadFailed(
            name: "tokenizer",
            underlying: AnyError(
              NSError(
                domain: "TinyAudio.ChatSession", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "tokenizer_config.json is not a dict"])))
        }
        let tokenizerConfig = Config(configDict as [NSString: Any])

        let tokenizerURL = modelDirectory.appendingPathComponent("tokenizer.json")
        let tokenizerRaw = try Data(contentsOf: tokenizerURL)
        guard
          let tokenizerDict = try JSONSerialization.jsonObject(with: tokenizerRaw)
            as? [String: Any]
        else {
          throw TinyAudioError.mlxModuleLoadFailed(
            name: "tokenizer",
            underlying: AnyError(
              NSError(
                domain: "TinyAudio.ChatSession", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "tokenizer.json is not a dict"])))
        }
        let tokenizerData = Config(tokenizerDict as [NSString: Any])
        tokenizer = try PreTrainedTokenizer(
          tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
      } catch let err as TinyAudioError {
        // Already wrapped at the throw site — propagate unchanged.
        throw err
      } catch {
        throw TinyAudioError.mlxModuleLoadFailed(name: "tokenizer", underlying: AnyError(error))
      }

      var eosTokenIds: Set<Int32> = []
      for eosStr in Transcriber.qwen3EosTokenStrings {
        if let id = tokenizer.convertTokenToId(eosStr) {
          eosTokenIds.insert(Int32(id))
        }
      }
      if let eosId = tokenizer.eosTokenId {
        eosTokenIds.insert(Int32(eosId))
      }

      return ChatSession(
        decoder: decoder,
        tokenizer: tokenizer,
        numDecoderLayers: numDecoderLayers,
        eosTokenIds: eosTokenIds
      )
    }
  }

  /// Local copy of `DecoderQuantizationSpec` from `Transcriber.swift`; the
  /// original is `fileprivate` so we duplicate the shape here rather than
  /// widen access for a DEBUG-only loader.
  private struct ChatSessionDecoderQuantizationSpec: Codable {
    struct Block: Codable {
      let groupSize: Int
      let bits: Int
      enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
      }
    }
    let quantization: Block
  }
#endif
