import Foundation
import MLX
import MLXNN
import Tokenizers

// MARK: - Profiling helper

/// Per-call phase-level timing helper.
///
/// All measurement is gated on `TINY_AUDIO_PROFILE=1`. When the env var is
/// unset every method is a no-op, so production callers pay zero overhead.
///
/// **Important**: the forced `MLX.eval` calls inserted at phase boundaries add
/// latency that the fused production graph does not pay — MLX is lazy and would
/// otherwise defer evaluation to the first `.item()` sync inside the decode loop.
/// The profile therefore attributes cost to its *actual* phase rather than the
/// first sync point, which is the right breakdown for diagnosis. Do not treat
/// the per-phase numbers as production steady-state costs; treat them as "where
/// does the work happen if you had to pay it all up front?"
private struct PipelineProfile {
    private var phases: [(name: String, ms: Int)] = []
    private var lastTime = ContinuousClock.now
    let isEnabled: Bool

    init() {
        self.isEnabled = ProcessInfo.processInfo.environment["TINY_AUDIO_PROFILE"] == "1"
    }

    mutating func reset() {
        guard isEnabled else { return }
        phases = []
        lastTime = ContinuousClock.now
    }

    mutating func mark(_ name: String) {
        guard isEnabled else { return }
        let now = ContinuousClock.now
        let elapsed = lastTime.duration(to: now)
        let ns = elapsed.components.seconds * 1_000_000_000 + elapsed.components.attoseconds / 1_000_000_000
        let ms = Int(ns / 1_000_000)
        phases.append((name, ms))
        lastTime = now
    }

    func dump() {
        guard isEnabled else { return }
        let total = phases.reduce(0) { $0 + $1.ms }
        for p in phases {
            let pct = total > 0 ? Int(Double(p.ms) / Double(total) * 100) : 0
            print(String(format: "  %-28s %5d ms  (%2d%%)", (p.name as NSString).utf8String!, p.ms, pct))
        }
        print("  -----------------------------------")
        print(String(format: "  %-28s %5d ms", "TOTAL pipeline", total))
    }
}

// MARK: - Concurrency helpers

/// Box for passing a non-Sendable value across a Task boundary when external
/// safety is guaranteed (the enclosed value is immutable after construction).
private struct SendableBox<T>: @unchecked Sendable {
    let value: T
}

/// End-to-end MLX inference orchestrator. Mirrors
/// `tiny_audio/mlx/model.py::MLXASRModel::_iter_token_ids` (lines 322-389).
///
/// Wires:
///   audio → mel → `GLMASREncoder` → `MLPProjector` → splice →
///   vendored `Qwen3Model` (prefill, then greedy decode) → token IDs
///
/// The decoder is the vendored mlx-swift-lm `Qwen3Model` (Apple's Swift
/// counterpart to Python's `mlx-lm`), patched to accept `inputEmbeddings`.
/// This aligns with the Python reference which calls `mlx_lm.load("Qwen/Qwen3-0.6B-MLX-4bit")`.
///
/// mlx-audio-swift's `Qwen3ASRTextModel` (Prince Canuma's port) has been
/// replaced to guarantee architectural parity with the Python reference.
final class ASRPipeline: @unchecked Sendable {
    let encoder: GLMASREncoder
    let projector: MLPProjector
    /// Vendored mlx-swift-lm Qwen3 backbone.
    /// `callAsFunction(_:cache:inputEmbeddings:)` returns logits `[B, T, vocabSize]`
    /// (lm_head or tied-embedding projection is applied inside `Qwen3Model`).
    let decoder: Qwen3Model
    let tokenizer: any Tokenizer
    let mel: LogMelSpectrogram
    let audioTokenId: Int32
    let eosTokenIds: Set<Int32>
    /// Number of transformer layers — used to allocate per-layer KV-caches.
    let numDecoderLayers: Int
    /// Pre-built logit bias: −∞ at the audio-token position, 0 elsewhere.
    /// Built once at init so the decode loop never allocates a fresh [Float]/MLXArray.
    let audioLogitBias: MLXArray

    init(
        encoder: GLMASREncoder,
        projector: MLPProjector,
        decoder: Qwen3Model,
        tokenizer: any Tokenizer,
        mel: LogMelSpectrogram,
        audioTokenId: Int32,
        eosTokenIds: Set<Int32>,
        numDecoderLayers: Int,
        vocabSize: Int
    ) {
        self.encoder = encoder
        self.projector = projector
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.mel = mel
        self.audioTokenId = audioTokenId
        self.eosTokenIds = eosTokenIds
        self.numDecoderLayers = numDecoderLayers
        var maskData = [Float](repeating: 0, count: vocabSize)
        maskData[Int(audioTokenId)] = -.infinity
        self.audioLogitBias = MLXArray(maskData)
    }

    // MARK: - Public API

    /// Greedy-decode token IDs from raw audio samples until EOS or `maxNewTokens`.
    ///
    /// Mirrors `_iter_token_ids` in Python exactly:
    /// - Mel → encoder → projector → splice → prefill → pipelined argmax loop.
    /// - Releases the MLX Metal pool on exit (normal, EOS, or error) via
    ///   `Memory.clearCache()`, mirroring Python's `finally: mx.clear_cache()`.
    ///
    /// - Parameters:
    ///   - samples:       Raw 16 kHz mono float PCM.
    ///   - maxNewTokens:  Maximum tokens to generate.
    ///   - systemPrompt:  Optional system message for the Qwen3 chat template.
    /// - Returns: An `AsyncThrowingStream` of Int32 token IDs (EOS included).
    func tokenStream(
        samples: [Float],
        maxNewTokens: Int,
        systemPrompt: String?
    ) -> AsyncThrowingStream<Int32, Error> {
        // Box self in an @unchecked Sendable to cross the Task boundary safely.
        // Safety: ASRPipeline properties are only mutated at initialisation time;
        // the stream body is the only concurrent user at runtime.
        let sendableSelf = SendableBox(value: self)

        return AsyncThrowingStream { continuation in
            Task.detached {
                let pipeline = sendableSelf.value
                defer {
                    // Mirror Python's `finally: mx.clear_cache()` — release the
                    // Metal pool regardless of how the generator exits.
                    Memory.clearCache()
                }
                do {
                    var profile = PipelineProfile()
                    profile.reset()

                    // 1. Mel spectrogram → encoder → projector.
                    //
                    // NOTE (profiling): MLX is lazy — calling encoder(mel) returns an
                    // unevaluated graph immediately. Without the forced MLX.eval below
                    // the entire encoder + projector cost would appear in "prefill" (the
                    // first real sync point). The gated MLX.eval calls break the graph at
                    // each phase boundary so the profile shows where work actually lives.
                    // Production calls skip the forced evals entirely (no overhead).
                    let melArr = pipeline.mel.compute(samples)   // [1, 128, T_mel]
                    if profile.isEnabled { MLX.eval(melArr) }
                    profile.mark("mel")

                    let encOut = pipeline.encoder(melArr)        // [1, T_enc, encoderDim]
                    if profile.isEnabled { MLX.eval(encOut) }
                    profile.mark("encoder")

                    let projOut = pipeline.projector(encOut)     // [1, T_proj, llmDim]
                    if profile.isEnabled { MLX.eval(projOut) }
                    profile.mark("projector")

                    let numAudio = projOut.dim(1)

                    // 2. Build chat-template prompt with N <audio> placeholders.
                    let inputIds = try Processor.buildPromptInputIds(
                        tokenizer: pipeline.tokenizer,
                        numAudioTokens: numAudio,
                        systemPrompt: systemPrompt
                    )  // [1, T_prompt]

                    // 3. Validate <audio> placeholder count matches projector output.
                    let idsFlat = inputIds.asArray(Int32.self)
                    var audioPositions: [Int32] = []
                    for (i, id) in idsFlat.enumerated() where id == pipeline.audioTokenId {
                        audioPositions.append(Int32(i))
                    }
                    guard audioPositions.count == numAudio else {
                        throw TinyAudioError.promptAudioTokenMismatch(
                            prompt: audioPositions.count,
                            projector: numAudio
                        )
                    }

                    // 4. Replace <audio> token ids with 0 before the embed lookup.
                    //    Audio positions are overwritten by the splice immediately
                    //    after, so the 0 placeholder never contributes to the output.
                    let audioMask = inputIds .== MLXArray(pipeline.audioTokenId)
                    let safeIds = MLX.where(audioMask, MLXArray.zeros(like: inputIds), inputIds)

                    // 5. Embed text tokens via the decoder's inner embed_tokens table,
                    //    then splice projected audio at <audio> positions.
                    let textEmbeds = pipeline.decoder.model.embedTokens(safeIds)  // [1, T, D]
                    let audioFrames = projOut[0, 0 ..< numAudio, 0...]            // [N, D]
                    let prefill = AudioEmbeddingSplice.splice(
                        textEmbeds: textEmbeds,
                        audioEmbeds: audioFrames,
                        audioPositions: audioPositions
                    )  // [1, T, D]
                    profile.mark("prompt + splice")

                    // 6. Prefill: one forward pass over the full prompt via inputEmbeddings.
                    //    Decoder returns logits [1, T, vocabSize] directly — lm_head is
                    //    applied inside Qwen3Model.callAsFunction.
                    let cache = pipeline.makeCache()
                    let prefillLogits = pipeline.decoder(
                        nil,
                        cache: cache,
                        inputEmbeddings: prefill
                    )  // [1, T, vocabSize]

                    // 7. First greedy step: argmax over the last token position.
                    //    Slice keeps the sequence dim ([1, 1, V]) so the shape
                    //    mirrors single-token decode steps below.
                    var y = MLX.argMax(
                        pipeline.maskAudioLogit(prefillLogits[0..., (-1)..., 0...]),
                        axis: -1
                    ).expandedDimensions(axis: 0)  // [1, 1]

                    // Force eval before marking prefill boundary so prefill cost is
                    // attributed here rather than to the first decode-step sync.
                    if profile.isEnabled { MLX.eval(y) }
                    MLX.asyncEval(y)
                    profile.mark("prefill")

                    // 8. Pipelined greedy decode loop.
                    //    Dispatch step N+1 before syncing step N so Metal compute
                    //    overlaps with the host .item() sync — same pattern as
                    //    mlx_lm.generate and the Python reference (_iter_token_ids).
                    var emittedCount = 0
                    for n in 0 ..< maxNewTokens {
                        var nextY: MLXArray? = nil
                        if n < maxNewTokens - 1 {
                            let nextLogits = pipeline.decoder(y, cache: cache)
                            nextY = MLX.argMax(
                                pipeline.maskAudioLogit(nextLogits[0..., (-1)..., 0...]),
                                axis: -1
                            ).expandedDimensions(axis: 0)
                            MLX.asyncEval(nextY!)
                        }

                        // Sync step N (overlaps with step N+1 compute above).
                        let tid = y.item(Int32.self)
                        emittedCount += 1
                        continuation.yield(tid)
                        if pipeline.eosTokenIds.contains(tid) {
                            profile.mark("decode (\(emittedCount) tokens)")
                            profile.dump()
                            continuation.finish()
                            return
                        }
                        if let next = nextY { y = next }
                    }
                    profile.mark("decode (\(emittedCount) tokens)")
                    profile.dump()
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Private helpers

    /// Build one `KVCacheSimple` per decoder transformer layer.
    private func makeCache() -> [KVCacheSimple] {
        (0 ..< numDecoderLayers).map { _ in KVCacheSimple() }
    }

    /// Add a −∞ bias at the audio-token position so it can never be sampled.
    ///
    /// `logits` shape: `[1, 1, V]` (single decode step) or `[1, T, V]`
    /// (last-token slice of the prefill). Broadcast-adds cleanly in both cases.
    /// Uses the pre-built `audioLogitBias` — no per-step allocation.
    private func maskAudioLogit(_ logits: MLXArray) -> MLXArray {
        return logits + audioLogitBias
    }
}
