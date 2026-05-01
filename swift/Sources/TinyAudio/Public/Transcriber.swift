import Foundation
import Hub
import MLX
import MLXNN
import Tokenizers

/// Main entry point for on-device ASR transcription.
///
/// `Transcriber` downloads (or reads from cache) a tiny-audio-mlx weight bundle,
/// verifies its integrity, builds and quantizes the encoder + decoder, loads the
/// fp16 projector, wires the full `ASRPipeline`, and runs a synthetic warmup to
/// JIT-compile MLX Metal kernels — all in a single `load` call.
///
/// Subsequent `transcribe` / `transcribeStream` calls (Task 27) are dispatched
/// through `pipeline`.
public actor Transcriber {
    internal let pipeline: ASRPipeline

    private init(pipeline: ASRPipeline) {
        self.pipeline = pipeline
    }

    // MARK: - Public factory

    /// Download (if needed) and load the model into memory. Subsequent
    /// `transcribe` calls run against the loaded model.
    ///
    /// - Parameters:
    ///   - source: Where to source weights from. Defaults to the public Hub bundle
    ///             (`mazesmazes/tiny-audio-mlx`).
    ///   - progress: Optional callback receiving Foundation `Progress` updates
    ///     during download. Pass `nil` for silent loads.
    /// - Returns: A fully initialised, warmed-up `Transcriber`.
    public static func load(
        from source: WeightSource = .defaultHub,
        progress: ((Progress) -> Void)? = nil
    ) async throws -> Transcriber {

        // 1. Materialise weights to a local directory.
        let bundle = try await HubLoader.materialize(source, progress: progress)
        let cache = WeightCache(directory: bundle)

        // 2. Verify manifest unless already marked complete.
        if !cache.isComplete {
            try ManifestVerifier.verify(directory: bundle)
            try cache.markComplete()
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
        let decoder: Qwen3Model
        let numDecoderLayers: Int
        do {
            let decoderConfigURL = bundle.appendingPathComponent("decoder_config.json")
            let decoderConfigData = try Data(contentsOf: decoderConfigURL)
            let qwenConfig = try JSONDecoder().decode(Qwen3Configuration.self, from: decoderConfigData)
            numDecoderLayers = qwenConfig.hiddenLayers
            decoder = Qwen3Model(qwenConfig)
            quantize(model: decoder, groupSize: 64, bits: 4)
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
            numDecoderLayers: numDecoderLayers
        )

        // 12. Warmup + return.
        let transcriber = Transcriber(pipeline: asr)
        await transcriber.warmup()
        return transcriber
    }

    // MARK: - Private helpers

    /// Run a synthetic 1-second transcribe to JIT-compile MLX Metal kernels.
    /// Mirrors `MLXASRModel.warmup()` in Python. Failures are non-fatal.
    private func warmup() async {
        let zeros = [Float](repeating: 0, count: 16_000)
        do {
            for try await _ in pipeline.tokenStream(
                samples: zeros, maxNewTokens: 4, systemPrompt: nil
            ) {
                // discard — warmup output is meaningless
            }
        } catch {
            // Non-fatal: surface only if real inference also fails.
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
        guard let configDict = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw TinyAudioError.mlxModuleLoadFailed(
                name: "tokenizer",
                underlying: AnyError(ConfigError.invalidJSON("tokenizer_config.json is not a dict"))
            )
        }
        let tokenizerConfig = Config(configDict as [NSString: Any])

        // Load tokenizer.json and patch added_tokens.
        let tokenizerURL = directory.appendingPathComponent("tokenizer.json")
        let tokenizerRaw = try Data(contentsOf: tokenizerURL)
        guard var tokenizerDict = try JSONSerialization.jsonObject(with: tokenizerRaw) as? [String: Any] else {
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
            addedTokens.append([
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

private extension GLMASREncoderConfig {
    init(dict: [String: Any]) throws {
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

private extension MLPProjector {
    convenience init(dict: [String: Any]) throws {
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
