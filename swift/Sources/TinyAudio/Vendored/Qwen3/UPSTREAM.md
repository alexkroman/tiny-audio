# Vendored Qwen3 from mlx-swift-lm

## Source

- Repo: https://github.com/ml-explore/mlx-swift-lm
- Commit: `7e2b7107be52ffbfe488f3c7987d3f52c1858b4b`
- Files vendored from `Libraries/`:
  - `MLXLLM/Models/Qwen3.swift` → `Qwen3Model.swift`
  - Minimal subset of `MLXLMCommon/` → `MLXLMCommonTypes.swift`
    - `JSONDecodingTypes.swift`: `StringOrNumber`
    - `KVCache.swift`: `KVCache`, `BaseKVCache`, `KVCacheSimple`,
      `createCausalMask`, `createAttentionMask`
    - `AttentionUtils.swift`: `attentionWithCacheUpdate`
    - `RoPEUtils.swift`: `RoPELayer` typealias
    - `RoPEApplication.swift`: `BatchPositionedKVCache`, `applyRotaryPosition`
    - `Adapters/LoRA/LoRAModel.swift`: `LoRAModel` protocol

## Why we vendor

`MLXLLM.Qwen3Model` (Apple's mlx-swift-lm) does not expose `inputEmbeddings`
on its forward path. TinyAudio's ASRPipeline needs that hook to splice audio
embeddings into the prompt embedding stream before the LLM forward pass —
the same pattern already used in `MLXVLM.Qwen3VL.swift`.

The Python reference (`tiny_audio/mlx/model.py`) calls
`mlx_lm.load("Qwen/Qwen3-0.6B-MLX-4bit")` — Apple's Python `mlx-lm` package.
The Swift counterpart is `mlx-swift-lm` (also by Apple). Using this instead of
mlx-audio-swift's `Qwen3ASRTextModel` (Prince Canuma's port for Alibaba's
Qwen3-ASR) guarantees architectural parity with the Python reference rather than
relying on "architecturally similar" alignment.

mlx-audio-swift remains a dependency (WhisperEncoder + mel processing).

## License

Vendored under the upstream MIT license. See `LICENSE` next to this file.

## Patches applied

### 1. Removed `import MLXLMCommon`

The vendored code compiles as part of the `TinyAudio` module. Required
MLXLMCommon types are inlined into `MLXLMCommonTypes.swift`.

### 2. Dropped `public` visibility

All types are `internal` (module-scoped). `ASRPipeline` is the only consumer
and it lives in the same Swift module.

### 3. `inputEmbeddings` patch — the core change

**`Qwen3ModelInner.callAsFunction` (was line 161 upstream):**

```swift
// Upstream (before patch):
public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
    var h = embedTokens(inputs)
    let mask = createAttentionMask(h: h, cache: cache?.first)
    // ...
}

// After patch:
func callAsFunction(
    _ inputs: MLXArray? = nil,
    cache: [KVCache]? = nil,
    inputEmbeddings: MLXArray? = nil
) -> MLXArray {
    var h: MLXArray
    if let inputEmbeddings {
        h = inputEmbeddings
    } else if let inputs {
        h = embedTokens(inputs)
    } else {
        fatalError("Qwen3ModelInner: either inputs or inputEmbeddings must be provided")
    }
    let mask = createAttentionMask(h: h, cache: cache?.first)
    // ... rest of body unchanged
}
```

**`Qwen3Model.callAsFunction` (was line 194 upstream):**

```swift
// Upstream (before patch):
public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
    var out = model(inputs, cache: cache)
    // ...
}

// After patch:
func callAsFunction(
    _ inputs: MLXArray? = nil,
    cache: [KVCache]? = nil,
    inputEmbeddings: MLXArray? = nil
) -> MLXArray {
    var out = model(inputs, cache: cache, inputEmbeddings: inputEmbeddings)
    // ... rest unchanged
}
```

### 4. Dropped `LLMModel` / `KVCacheDimensionProvider` / `LoRAModel`

The `Qwen3Model` class no longer declares conformance to `LLMModel`,
`KVCacheDimensionProvider`, or `LoRAModel`. These are only used by the
mlx-swift-lm inference / fine-tuning pipeline, not by TinyAudio's ASRPipeline.

### 5. Quantized KV cache types added back

`MLXLMCommonTypes.swift` now also vendors the quantized KV cache types
that an earlier patch had dropped. `attentionWithCacheUpdate` routes
`QuantizedKVCacheProtocol` instances through the quantized SDPA path.

Vendored from upstream `Libraries/MLXLMCommon/KVCache.swift`:

- `QuantizedKVCacheProtocol` (lines 94-118)
- `QuantizedKVCache` class (lines 728-988)
- `KVCacheSimple.toQuantized(...)` (lines 407-427)
- `quantizedScaledDotProductAttention` (lines 1532-1620)
- `maybeQuantizeKVCache` (lines 1636-1659)

As with the rest of `MLXLMCommonTypes.swift`, visibility was lowered from
`public` to internal (module-scoped). Helper methods on `QuantizedKVCache`
(`treeMap`, `treeMapPair`, `initQuant`, `expandQuant`) remain `private`.

These additions are purely additive: no existing call sites yet use the
quantized cache. They are wired in by follow-up commits (KV-cache
quantization and prefix-cache reuse).

#### Cosmetic deviations from upstream (swift-format-driven)

- `let B = keys.dim(0)` in `QuantizedKVCache.updateQuantized` was renamed to
  `let batchSize = keys.dim(0)` to satisfy `swift-format`'s
  `AlwaysUseLowerCamelCase` rule. (Upstream tuple destructurings like
  `let (B, ...) = (...)` are not flagged, only standalone `let B = ...`.)
- `for i in 0 ..< cache.count` in `maybeQuantizeKVCache` was tightened to
  `for i in 0..<cache.count` to match the spacing convention used elsewhere
  in `MLXLMCommonTypes.swift`.
- The new `attentionWithCacheUpdate` quantized branch uses shorter local
  names (`qCache`, `qK`, `qV`) than upstream `AttentionUtils.swift:54-66`,
  which uses `quantizedKVCache`, `quantizedKeys`, `quantizedValues`. The
  logic is identical; the rename keeps the routing block compact.

When rebasing this file against upstream, re-apply the rename + spacing
deltas (or just run `swift-format -i`).

## How to rebase

When rebasing to a newer mlx-swift-lm commit:

1. Download the new `Qwen3.swift` from `Libraries/MLXLLM/Models/`.
1. Strip `import MLXLMCommon`, lower visibility (`public` → nothing).
1. Re-apply the `inputEmbeddings` patch to `Qwen3ModelInner` and `Qwen3Model`.
1. Drop `LLMModel` / `KVCacheDimensionProvider` / `LoRAModel` conformances.
1. Update the commit SHA in this file and in `Qwen3Model.swift`'s header comment.
1. `swift build` + `swift test` to verify.
