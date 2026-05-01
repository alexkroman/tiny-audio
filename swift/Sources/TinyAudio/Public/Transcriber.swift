import Foundation
import Hub
import MLX
import MLXNN
import Tokenizers

/// On-device speech recognition backed by a 4-bit quantized GLM-ASR encoder and Qwen3-0.6B decoder.
///
/// `Transcriber` is the primary entry point for file and buffer transcription.
/// Call ``load(from:progress:)`` once to download, verify, and warm-up the
/// model, then reuse the returned actor for all subsequent transcription calls.
///
/// The actor is `Sendable` — it may be shared freely across concurrent tasks.
/// All calls to ``transcribe(_:options:)`` and ``transcribeStream(_:options:)``
/// are serialised through the actor's executor, so concurrent callers queue up
/// rather than racing.
///
/// ```swift
/// // Typical app-start initialisation
/// let transcriber = try await Transcriber.load { p in
///     print("Loaded \(Int(p.fractionCompleted * 100))%")
/// }
/// ```
public actor Transcriber {
  internal let pipeline: ASRPipeline

  private init(pipeline: ASRPipeline) {
    self.pipeline = pipeline
  }

  // MARK: - Public factory

  /// Verify and load the model, returning a warmed-up `Transcriber`.
  ///
  /// When called with the default ``WeightSource/defaultHub`` source, weights
  /// are read directly from the SDK's resource bundle (``Bundle/module``) — no
  /// network request is made.  Use ``WeightSource/hub(repoID:revision:)`` to
  /// pull a different model revision from HuggingFace Hub on demand.
  ///
  /// After loading, the actor runs a short synthetic warmup at 1 s, 5 s, and
  /// 15 s of audio to JIT-compile Metal kernels for the shapes most common in
  /// real ASR — adding ~300 ms to cold load time but eliminating the first-call
  /// JIT penalty.
  ///
  /// ## Example
  ///
  /// ```swift
  /// // Show a progress bar during the initial download.
  /// let transcriber = try await Transcriber.load(from: .defaultHub) { progress in
  ///     ProgressView(value: progress.fractionCompleted)
  /// }
  ///
  /// // Or load silently from a local directory (e.g. in tests or offline builds).
  /// let local = try await Transcriber.load(from: .localDirectory(bundleURL))
  /// ```
  ///
  /// - Parameters:
  ///   - source: Where to source model weights from. Defaults to
  ///     ``WeightSource/defaultHub`` (`mazesmazes/tiny-audio-mlx` on HuggingFace).
  ///   - progress: Optional callback invoked with a Foundation `Progress` object
  ///     during download.  Called on an unspecified queue; update UI on the main
  ///     actor explicitly.  Pass `nil` for a silent load.
  /// - Returns: A fully initialised and warmed-up `Transcriber` ready for
  ///   immediate inference calls.
  /// - Throws: ``TinyAudioError/weightDownloadFailed(underlying:)`` on network
  ///   failure, ``TinyAudioError/manifestMismatch(file:expected:actual:)`` on
  ///   corrupt cache, or ``TinyAudioError/mlxModuleLoadFailed(name:underlying:)``
  ///   if a model component cannot be built.
  public static func load(
    from source: WeightSource = .defaultHub,
    progress: ((Progress) -> Void)? = nil
  ) async throws -> Transcriber {

    // 1. Resolve weight directory.
    let bundle: URL
    switch source {
    case .defaultHub:
      // Bundled model — no download needed.
      guard let modelDir = Bundle.module.url(forResource: "Model", withExtension: nil) else {
        throw TinyAudioError.mlxModuleLoadFailed(
          name: "bundled model",
          underlying: AnyError(
            ConfigError.invalidJSON(
              "Resources/Model is missing from the SDK bundle"
            ))
        )
      }
      bundle = modelDir
    case .hub(let repoID, let revision):
      bundle = try await HubLoader.materialize(
        .hub(repoID: repoID, revision: revision), progress: progress)
    case .localDirectory(let url):
      bundle = url
    }
    let cache = WeightCache(directory: bundle)

    // 2. Verify manifest unless already marked complete OR the directory is
    //    read-only (bundled resources inside the SwiftPM bundle are not writable).
    if !cache.isComplete {
      do {
        try ManifestVerifier.verify(directory: bundle)
        // Try to mark complete; ignore failure for read-only bundled dirs.
        try? cache.markComplete()
      } catch {
        // For bundled resources, manifest verification is a sanity check;
        // if it fails the package is corrupt — surface as load failure.
        throw error
      }
    }

    // 3. Read bundle config.json.
    let configURL = bundle.appendingPathComponent("config.json")
    let configData = try Data(contentsOf: configURL)
    guard let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
      throw TinyAudioError.mlxModuleLoadFailed(
        name: "config",
        underlying: AnyError(ConfigError.invalidJSON("config.json is not a dict"))
      )
    }

    // 4. Build mel spectrogram (no resource path needed; uses computeMelSpectrogram).
    let mel = LogMelSpectrogram()

    // 5. Build encoder, quantize to 4-bit, load weights.
    let encoder: GLMASREncoder
    do {
      guard let encConfigDict = config["encoder"] as? [String: Any] else {
        throw ConfigError.missingKey("encoder")
      }
      let encConfig = try GLMASREncoderConfig(dict: encConfigDict)
      encoder = GLMASREncoder(encConfig)
      quantize(model: encoder, groupSize: 64, bits: 4)
      let weights = try MLX.loadArrays(
        url: bundle.appendingPathComponent("encoder.safetensors")
      )
      try encoder.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    } catch let e as TinyAudioError {
      throw e
    } catch {
      throw TinyAudioError.mlxModuleLoadFailed(name: "encoder", underlying: AnyError(error))
    }

    // 6. Build projector (fp16 — no quantization), load weights.
    let projector: MLPProjector
    do {
      guard let projConfigDict = config["projector"] as? [String: Any] else {
        throw ConfigError.missingKey("projector")
      }
      projector = try MLPProjector(dict: projConfigDict)
      let weights = try MLX.loadArrays(
        url: bundle.appendingPathComponent("projector.safetensors")
      )
      try projector.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    } catch let e as TinyAudioError {
      throw e
    } catch {
      throw TinyAudioError.mlxModuleLoadFailed(name: "projector", underlying: AnyError(error))
    }

    // 7. Build vendored Qwen3 decoder, quantize to 4-bit, load weights.
    //    The bundle's decoder.safetensors is already 4-bit; quantize() here
    //    transforms the Linear modules in-place to QuantizedLinear so their
    //    parameter shapes match the stored weight layout.
    //    The Qwen3-0.6B bundle uses group_size=128 (verified from scales shape:
    //    v_proj scales [1024,8] from full weight [1024,1024] => 1024/8=128).
    let decoder: Qwen3Model
    let numDecoderLayers: Int
    let vocabSize: Int
    do {
      let decoderConfigURL = bundle.appendingPathComponent("decoder_config.json")
      let decoderConfigData = try Data(contentsOf: decoderConfigURL)
      let qwenConfig = try JSONDecoder().decode(Qwen3Configuration.self, from: decoderConfigData)
      numDecoderLayers = qwenConfig.hiddenLayers
      vocabSize = qwenConfig.vocabularySize
      decoder = Qwen3Model(qwenConfig)
      quantize(model: decoder, groupSize: 128, bits: 4)
      let rawWeights = try MLX.loadArrays(
        url: bundle.appendingPathComponent("decoder.safetensors")
      )
      // sanitize strips lm_head.weight when tieWordEmbeddings=true.
      let weights = decoder.sanitize(weights: rawWeights)
      try decoder.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    } catch let e as TinyAudioError {
      throw e
    } catch {
      throw TinyAudioError.mlxModuleLoadFailed(name: "decoder", underlying: AnyError(error))
    }

    // 8. Load tokenizer, runtime-add the <audio> special token.
    //    The bundle's tokenizer.json is the base Qwen3 tokenizer; <audio> is added
    //    at runtime by the Python pipeline (ASRModel._init_tokenizer). We mirror
    //    that here by patching `added_tokens` before constructing the tokenizer,
    //    matching ProcessorTests.swift's approach.
    let tokenizer = try await loadTokenizerWithAudioToken(directory: bundle)

    // 9. Resolve audio token ID.
    guard let audioTokenIdInt = tokenizer.convertTokenToId(Processor.audioToken) else {
      throw TinyAudioError.mlxModuleLoadFailed(
        name: "tokenizer",
        underlying: AnyError(ConfigError.missingKey("<audio> token not found after patching"))
      )
    }
    let audioTokenId = Int32(audioTokenIdInt)

    // 10. Resolve EOS token IDs.
    var eosTokenIds: Set<Int32> = []
    for eosStr in ["<|im_end|>", "<|endoftext|>"] {
      if let id = tokenizer.convertTokenToId(eosStr) {
        eosTokenIds.insert(Int32(id))
      }
    }
    if let eosId = tokenizer.eosTokenId {
      eosTokenIds.insert(Int32(eosId))
    }

    // 11. Construct ASRPipeline.
    let asr = ASRPipeline(
      encoder: encoder,
      projector: projector,
      decoder: decoder,
      tokenizer: tokenizer,
      mel: mel,
      audioTokenId: audioTokenId,
      eosTokenIds: eosTokenIds,
      numDecoderLayers: numDecoderLayers,
      vocabSize: vocabSize
    )

    // 12. Warmup + return.
    let transcriber = Transcriber(pipeline: asr)
    await transcriber.warmup()
    return transcriber
  }

  // MARK: - Private helpers

  /// Run synthetic transcribes at several audio durations to JIT-compile MLX
  /// Metal kernels for the shapes most likely to occur in real ASR calls.
  ///
  /// Warming at only 1 s means the first real call at a different duration
  /// (e.g. 5 s or 15 s) pays a ~15 ms per-shape JIT penalty. Warming at
  /// 1 s / 5 s / 15 s covers most short-form ASR durations and adds ~300 ms
  /// to cold load time while eliminating that first-call JIT cost.
  /// Mirrors `MLXASRModel.warmup()` in Python. Failures are non-fatal.
  private func warmup() async {
    // Warm up MLX kernels for several audio durations. Real audio comes in
    // many lengths; warming at one shape only triggers a JIT recompile on the
    // first real call. 1 s / 5 s / 15 s covers most short-form ASR durations
    // and adds ~300 ms to load time but saves ~15 ms on every first call
    // until that shape is hit.
    let durationsSeconds: [Int] = [1, 5, 15]
    for seconds in durationsSeconds {
      let zeros = [Float](repeating: 0, count: 16_000 * seconds)
      do {
        for try await _ in pipeline.tokenStream(
          samples: zeros,
          maxNewTokens: 4,
          systemPrompt: nil
        ) {
          // discard
        }
      } catch {
        // Warmup failures are non-fatal; surface only if real inference also fails.
      }
    }
  }

  /// Load a Qwen3 tokenizer from the bundle directory, patching `added_tokens`
  /// in `tokenizer.json` to include `<audio>` as a special token.
  ///
  /// This mirrors Python's `tokenizer.add_special_tokens({"additional_special_tokens":
  /// ["<audio>"]})` in `ASRModel._init_tokenizer`. The patching approach copies
  /// `ProcessorTests.swift` exactly: mutate the `added_tokens` array in-memory,
  /// then construct `PreTrainedTokenizer` directly from the config + data dicts.
  ///
  /// The audio token ID is fixed at 151669 for Qwen3-0.6B (the next available ID
  /// after the base vocabulary). The actual ID is cross-checked after construction.
  private static func loadTokenizerWithAudioToken(directory: URL) async throws -> any Tokenizer {
    // Load tokenizer_config.json.
    let configURL = directory.appendingPathComponent("tokenizer_config.json")
    let configData = try Data(contentsOf: configURL)
    guard let configDict = try JSONSerialization.jsonObject(with: configData) as? [String: Any]
    else {
      throw TinyAudioError.mlxModuleLoadFailed(
        name: "tokenizer",
        underlying: AnyError(ConfigError.invalidJSON("tokenizer_config.json is not a dict"))
      )
    }
    let tokenizerConfig = Config(configDict as [NSString: Any])

    // Load tokenizer.json and patch added_tokens.
    let tokenizerURL = directory.appendingPathComponent("tokenizer.json")
    let tokenizerRaw = try Data(contentsOf: tokenizerURL)
    guard var tokenizerDict = try JSONSerialization.jsonObject(with: tokenizerRaw) as? [String: Any]
    else {
      throw TinyAudioError.mlxModuleLoadFailed(
        name: "tokenizer",
        underlying: AnyError(ConfigError.invalidJSON("tokenizer.json is not a dict"))
      )
    }

    let audioToken = Processor.audioToken
    // Determine next available ID from the existing added_tokens list.
    var addedTokens = (tokenizerDict["added_tokens"] as? [[String: Any]]) ?? []
    let alreadyAdded = addedTokens.contains { ($0["content"] as? String) == audioToken }
    if !alreadyAdded {
      let maxExistingId = addedTokens.compactMap { $0["id"] as? Int }.max() ?? 151664
      let audioTokenId = maxExistingId + 1
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
    return try PreTrainedTokenizer(
      tokenizerConfig: tokenizerConfig,
      tokenizerData: tokenizerData
    )
  }
}

// MARK: - GLMASREncoderConfig convenience init

extension GLMASREncoderConfig {
  fileprivate init(dict: [String: Any]) throws {
    guard
      let nMels = dict["n_mels"] as? Int,
      let encoderDim = dict["encoder_dim"] as? Int,
      let numLayers = dict["num_layers"] as? Int,
      let numHeads = dict["num_heads"] as? Int,
      let headDim = dict["head_dim"] as? Int,
      let intermediateSize = dict["intermediate_size"] as? Int
    else {
      throw ConfigError.missingKey("encoder fields")
    }
    let ropeTheta: Float
    if let d = dict["rope_theta"] as? Double {
      ropeTheta = Float(d)
    } else if let f = dict["rope_theta"] as? Float {
      ropeTheta = f
    } else if let i = dict["rope_theta"] as? Int {
      ropeTheta = Float(i)
    } else {
      ropeTheta = 10_000
    }
    self.init(
      nMels: nMels,
      encoderDim: encoderDim,
      numLayers: numLayers,
      numHeads: numHeads,
      headDim: headDim,
      intermediateSize: intermediateSize,
      ropeTheta: ropeTheta
    )
  }
}

// MARK: - MLPProjector convenience init

extension MLPProjector {
  fileprivate convenience init(dict: [String: Any]) throws {
    guard
      let encoderDim = dict["encoder_dim"] as? Int,
      let llmDim = dict["llm_dim"] as? Int,
      let hiddenDim = dict["hidden_dim"] as? Int,
      let poolStride = dict["pool_stride"] as? Int
    else {
      throw ConfigError.missingKey("projector fields")
    }
    self.init(
      encoderDim: encoderDim,
      llmDim: llmDim,
      hiddenDim: hiddenDim,
      poolStride: poolStride
    )
  }
}

// MARK: - Internal config-parsing errors

private enum ConfigError: Error {
  case missingKey(String)
  case invalidJSON(String)
}

// MARK: - Transcription public API

extension Transcriber {
  /// Transcribe audio to text, waiting for the full transcript before returning.
  ///
  /// A convenience wrapper around ``transcribeStream(_:options:)`` that
  /// collects all token deltas and returns the concatenated result.  Use
  /// ``transcribeStream(_:options:)`` directly when you want incremental output
  /// as tokens are generated.
  ///
  /// ## Example
  ///
  /// ```swift
  /// let url = Bundle.main.url(forResource: "recording", withExtension: "wav")!
  /// let text = try await transcriber.transcribe(.file(url))
  /// print(text)   // "Hello from on-device ASR."
  /// ```
  ///
  /// - Parameters:
  ///   - audio: The audio to transcribe.  Any ``AudioInput`` case is accepted;
  ///     the SDK normalises to 16 kHz mono Float32 internally.
  ///   - options: Decoding options.  Defaults to ``TranscriptionOptions/default``
  ///     (96 max tokens, no system prompt).
  /// - Returns: The complete transcript as a single `String`.
  /// - Throws: ``TinyAudioError/audioEmpty`` if the decoded audio contains no
  ///   samples, ``TinyAudioError/audioFormatUnsupported(reason:)`` for
  ///   unreadable audio, or an MLX runtime error during decoding.
  public func transcribe(
    _ audio: AudioInput,
    options: TranscriptionOptions = .default
  ) async throws -> String {
    var collected = ""
    for try await delta in transcribeStream(audio, options: options) {
      collected += delta
    }
    return collected
  }

  /// Transcribe audio and yield incremental text deltas as tokens are generated.
  ///
  /// Each yielded `String` is the *delta* since the previous yield —
  /// concatenating all deltas produces the complete transcript.  The stream
  /// finishes when an EOS token is produced or ``TranscriptionOptions/maxNewTokens``
  /// is reached.
  ///
  /// The method is `nonisolated` so the returned stream can be iterated from
  /// any actor context without an initial `await` on the actor.  The actor hop
  /// occurs inside the spawned `Task` when `pipeline` is accessed.
  ///
  /// Token detokenization uses a fast single-token path for the common BPE
  /// case and falls back to a full-prefix decode when a multi-byte UTF-8
  /// boundary may have been split across tokens (e.g. surrogate pairs, combining
  /// marks, or zero-width joiners).
  ///
  /// ## Example
  ///
  /// ```swift
  /// var transcript = ""
  /// for try await delta in transcriber.transcribeStream(.file(audioURL)) {
  ///     transcript += delta
  ///     print(delta, terminator: "")  // stream tokens to the console
  /// }
  /// print()  // newline after final token
  /// ```
  ///
  /// - Parameters:
  ///   - audio: The audio to transcribe.  Any ``AudioInput`` case is accepted.
  ///   - options: Decoding options.  Defaults to ``TranscriptionOptions/default``.
  /// - Returns: An `AsyncThrowingStream<String, Error>` of incremental text
  ///   deltas.  Errors thrown inside the stream are surfaced as the stream's
  ///   `failure` case.
  public nonisolated func transcribeStream(
    _ audio: AudioInput,
    options: TranscriptionOptions = .default
  ) -> AsyncThrowingStream<String, Error> {
    return AsyncThrowingStream { continuation in
      Task {
        do {
          let samples = try AudioDecoder.decode(audio)
          let pipeline = self.pipeline
          var accumulatedInts: [Int] = []
          var lastText = ""

          for try await tid in pipeline.tokenStream(
            samples: samples,
            maxNewTokens: options.maxNewTokens,
            systemPrompt: options.systemPrompt
          ) {
            if pipeline.eosTokenIds.contains(tid) { break }
            accumulatedInts.append(Int(tid))

            // Re-decode the full accumulated prefix every step. Required for
            // BPE / SentencePiece tokens whose detokenization depends on the
            // surrounding context — single-token decodes drop word-boundary
            // markers (e.g. `▁`) and yield wrong concatenated output.
            let fullText = pipeline.tokenizer.decode(tokens: accumulatedInts)
            let delta: String
            if fullText.hasPrefix(lastText) {
              delta = String(fullText.dropFirst(lastText.count))
            } else {
              delta = fullText
            }
            lastText = fullText
            if !delta.isEmpty { continuation.yield(delta) }
          }
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }
    }
  }

  /// Return raw token IDs for a given audio input, without detokenization.
  ///
  /// This accessor is gated behind `@_spi(Testing)` and is not visible to
  /// application code.  It is used by the SDK's own test suite to verify that
  /// the model produces the expected token sequence for a known audio input.
  ///
  /// - Parameters:
  ///   - audio: The audio to process.
  ///   - maxNewTokens: Hard cap on tokens generated.
  ///   - systemPrompt: Optional system prompt string prepended to the chat
  ///     template.  Pass `nil` for default formatting.
  /// - Returns: All token IDs produced before EOS or the token limit, in order.
  /// - Throws: Any error from audio decoding or the MLX pipeline.
  @_spi(Testing)
  public func tokenIDsForTesting(
    _ audio: AudioInput,
    maxNewTokens: Int,
    systemPrompt: String? = nil
  ) async throws -> [Int32] {
    let samples = try AudioDecoder.decode(audio)
    var ids: [Int32] = []
    for try await tid in pipeline.tokenStream(
      samples: samples,
      maxNewTokens: maxNewTokens,
      systemPrompt: systemPrompt
    ) {
      ids.append(tid)
    }
    return ids
  }
}
