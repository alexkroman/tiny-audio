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
    let ns =
      elapsed.components.seconds * 1_000_000_000 + elapsed.components.attoseconds / 1_000_000_000
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
  /// Pre-computed token IDs for the constant chat-template prefix and suffix.
  /// Built once at init from the system prompt the pipeline was constructed
  /// with; reused on every call.
  let promptParts: Processor.PromptParts
  /// Per-layer KV-cache snapshot of the prefix prefill. `nil` when the
  /// prefix is too short to bother caching (default-system-prompt case);
  /// in that case `tokenStream` falls back to the legacy full-prefill path.
  let prefixCache: [KVCache]?
  /// Test-only: when true, `tokenStream` always takes the legacy full
  /// prefill path even if `prefixCache` is non-nil. Used by
  /// `PrefixCacheTests` to compare both code paths against the same audio.
  @_spi(Testing) public var bypassPrefixCacheForTesting: Bool = false

  init(
    encoder: GLMASREncoder,
    projector: MLPProjector,
    decoder: Qwen3Model,
    tokenizer: any Tokenizer,
    mel: LogMelSpectrogram,
    audioTokenId: Int32,
    eosTokenIds: Set<Int32>,
    numDecoderLayers: Int,
    vocabSize: Int,
    cachedSystemPrompt: String? = nil
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

    self.promptParts = Processor.buildPromptParts(
      tokenizer: tokenizer, systemPrompt: cachedSystemPrompt)
    self.prefixCache = Self.buildPrefixCache(
      decoder: decoder,
      numDecoderLayers: numDecoderLayers,
      prefixIds: promptParts.prefixIds
    )
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
          let melArr = pipeline.mel.compute(samples)  // [1, 128, T_mel]
          if profile.isEnabled { MLX.eval(melArr) }
          profile.mark("mel")

          let encOut = pipeline.encoder(melArr)  // [1, T_enc, encoderDim]
          if profile.isEnabled { MLX.eval(encOut) }
          profile.mark("encoder")

          let projOut = pipeline.projector(encOut)  // [1, T_proj, llmDim]
          if profile.isEnabled { MLX.eval(projOut) }
          profile.mark("projector")

          // 2. Build the audio + suffix portion. The prefix tokens are
          //    already prefilled into `pipeline.prefixCache` (when
          //    available — see fallback below).
          let parts = pipeline.promptParts
          let numAudio = projOut.dim(1)

          // Build audio+suffix token IDs (audio placeholders that we
          // overwrite via embedding splice immediately afterward).
          let audioRun = [Int32](repeating: pipeline.audioTokenId, count: numAudio)
          let postfixIds = audioRun + parts.suffixIds
          let postfixIdsArr = MLXArray(postfixIds).expandedDimensions(axis: 0)

          // Replace audio token IDs with 0 before the embed lookup; the
          // splice below overwrites those rows so the 0 placeholder never
          // contributes to the output.
          let audioMask = postfixIdsArr .== MLXArray(pipeline.audioTokenId)
          let safeIds = MLX.where(
            audioMask, MLXArray.zeros(like: postfixIdsArr), postfixIdsArr)
          let postfixEmbeds = pipeline.decoder.model.embedTokens(safeIds)

          // Audio positions inside the postfix segment are 0..<numAudio
          // (the audio run is the leading slice of postfix).
          let audioPositions = (0..<Int32(numAudio)).map { $0 }
          let audioFrames = projOut[0, 0..<numAudio, 0...]
          let postfix = AudioEmbeddingSplice.splice(
            textEmbeds: postfixEmbeds,
            audioEmbeds: audioFrames,
            audioPositions: audioPositions
          )
          profile.mark("prompt + splice")

          // 3. Either copy the prebuilt prefix cache and forward only the
          //    postfix, or fall back to a fresh full prefill when:
          //    - the caller passed a different systemPrompt than the
          //      pipeline was built with, or
          //    - the prefix was too short to cache (returns nil), or
          //    - tests have set bypassPrefixCacheForTesting.
          var cache: [KVCache]
          let prefillLogits: MLXArray
          let promptLen: Int
          let useCachedPrefix =
            (systemPrompt == nil)
            && !pipeline.bypassPrefixCacheForTesting
          if useCachedPrefix, let prefix = pipeline.prefixCache {
            cache = prefix.map { $0.copy() }
            prefillLogits = pipeline.decoder(
              nil, cache: cache, inputEmbeddings: postfix)
            promptLen = parts.prefixIds.count + postfix.dim(1)
          } else {
            // Legacy path: prefill prefix + audio + suffix in one shot.
            // Re-tokenize the prefix here because the caller may have
            // supplied a different systemPrompt than `pipeline.promptParts`
            // was built with.
            let actualParts = Processor.buildPromptParts(
              tokenizer: pipeline.tokenizer, systemPrompt: systemPrompt)
            let prefixEmbeds = pipeline.decoder.model.embedTokens(
              MLXArray(actualParts.prefixIds).expandedDimensions(axis: 0))
            let fullPrefill = MLX.concatenated([prefixEmbeds, postfix], axis: 1)
            cache = pipeline.makeCache()
            prefillLogits = pipeline.decoder(
              nil, cache: cache, inputEmbeddings: fullPrefill)
            promptLen = fullPrefill.dim(1)
          }

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

          // Pre-grow the cache to its final size so the compiled step's
          // input identity stays stable for the whole loop.
          Self.growCacheToFit(
            cache: cache,
            targetTokens: promptLen + maxNewTokens + 8,  // +8 slack
            decoder: pipeline.decoder
          )

          // Build the compiled per-token decode closure once. Captures
          // `cache` and `pipeline.decoder` by reference; the
          // `inputs:`/`outputs:` declarations tell MLX to track cache
          // mutation across calls.
          //
          // Honor TINY_AUDIO_NO_COMPILE=1 as a kill-switch.
          let useCompile =
            ProcessInfo.processInfo.environment[
              "TINY_AUDIO_NO_COMPILE"] != "1"
          let updatableCache: [any Updatable] = cache
          let compiledStep: (MLXArray) -> MLXArray
          if useCompile {
            let fn = MLX.compile(
              inputs: updatableCache,
              outputs: updatableCache
            ) { (ys: [MLXArray]) -> [MLXArray] in
              [pipeline.decoder(ys[0], cache: cache)]
            }
            compiledStep = { y in fn([y])[0] }
          } else {
            compiledStep = { y in pipeline.decoder(y, cache: cache) }
          }

          // 8. Pipelined greedy decode loop (unchanged structure, only
          //    the inner forward call swapped to `compiledStep`).
          //    Dispatch step N+1 before syncing step N so Metal compute
          //    overlaps with the host .item() sync — same pattern as
          //    mlx_lm.generate and the Python reference (_iter_token_ids).
          var emittedCount = 0
          for n in 0..<maxNewTokens {
            var nextY: MLXArray? = nil
            if n < maxNewTokens - 1 {
              let nextLogits = compiledStep(y)
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
  private func makeCache() -> [KVCache] {
    (0..<numDecoderLayers).map { _ in KVCacheSimple() }
  }

  /// Run a one-shot prefill over `prefixIds` to populate a `KVCacheSimple`
  /// per layer; returns the populated caches. We do **not** apply any
  /// post-prefill quantize/transform here — the per-call cache list is
  /// what `tokenStream` operates on; this snapshot is only `copy()`'d.
  ///
  /// Returns `nil` when the prefix is too short to be worth caching —
  /// the savings don't justify the load-time cost. ~10 tokens is roughly
  /// the break-even.
  private static func buildPrefixCache(
    decoder: Qwen3Model,
    numDecoderLayers: Int,
    prefixIds: [Int32]
  ) -> [KVCache]? {
    guard prefixIds.count > 10 else { return nil }

    let cache: [KVCache] = (0..<numDecoderLayers).map { _ in KVCacheSimple() }
    let inputs = MLXArray(prefixIds).expandedDimensions(axis: 0)
    _ = decoder(inputs, cache: cache)
    // Force materialisation so the snapshot is ready in memory.
    for c in cache {
      MLX.eval(c.innerState())
    }
    return cache
  }

  /// Force `KVCacheSimple` to allocate its key/value buffers up-front to a
  /// size sufficient for the whole decode loop, so the buffer pointer stays
  /// stable. Required when the loop body is wrapped in `MLX.compile` — a
  /// mid-loop buffer realloc invalidates the compiled graph's input identity
  /// and produces wrong outputs.
  ///
  /// `KVCacheSimple` allocates in `step`-sized chunks. We trigger a one-shot
  /// `update()` with a synthetic zero tensor of the right shape, then walk
  /// `offset` back so the cache appears at its original logical position.
  private static func growCacheToFit(
    cache: [KVCache],
    targetTokens: Int,
    decoder: Qwen3Model
  ) {
    let cfg = decoder.configuration
    let kvHeads = cfg.kvHeads
    let headDim = cfg.headDim
    for c in cache {
      guard let simple = c as? KVCacheSimple else { continue }
      if let keys = simple.keys, keys.dim(2) >= targetTokens { continue }
      let needed = max(1, targetTokens - simple.offset)
      // Match the existing buffer's dtype if one exists; the prefill above
      // has already written real K/V at the model's native dtype, so any
      // mismatch here would force a concat across dtypes inside `update()`.
      // Decode-loop-only invariant: caller emits one token per `update()`;
      // `previous + 1 > currentKeys.dim(2)` won't trip after this grow.
      let dtype = simple.keys?.dtype ?? .float16
      let dummyK = MLXArray.zeros([1, kvHeads, needed, headDim], dtype: dtype)
      let dummyV = MLXArray.zeros([1, kvHeads, needed, headDim], dtype: dtype)
      _ = simple.update(keys: dummyK, values: dummyV)
      simple.offset -= needed
    }
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
