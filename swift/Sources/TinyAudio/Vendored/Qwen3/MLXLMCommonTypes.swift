// Vendored from ml-explore/mlx-swift-lm (MIT License).
// See LICENSE and UPSTREAM.md next to this file.
//
// Minimal subset of MLXLMCommon types required to compile Qwen3Model.swift
// without importing the full MLXLMCommon module.
//
// Sources:
//   Libraries/MLXLMCommon/JSONDecodingTypes.swift  -> StringOrNumber
//   Libraries/MLXLMCommon/KVCache.swift            -> KVCache, BaseKVCache, KVCacheSimple,
//                                                     createCausalMask, createAttentionMask
//   Libraries/MLXLMCommon/AttentionUtils.swift     -> attentionWithCacheUpdate
//   Libraries/MLXLMCommon/RoPEUtils.swift          -> RoPELayer typealias
//   Libraries/MLXLMCommon/RoPEApplication.swift    -> BatchPositionedKVCache, applyRotaryPosition
//   Libraries/MLXLMCommon/Adapters/LoRA/LoRAModel.swift -> LoRAModel (protocol only)

import Foundation
import MLX
import MLXNN

// MARK: - StringOrNumber (from JSONDecodingTypes.swift)

/// Representation of a heterogenous type in a JSON configuration file.
enum StringOrNumber: Codable, Equatable, Sendable {
  case string(String)
  case int(Int)
  case float(Float)
  case ints([Int])
  case floats([Float])
  case bool(Bool)

  init(from decoder: Decoder) throws {
    let values = try decoder.singleValueContainer()
    if let v = try? values.decode(Int.self) {
      self = .int(v)
    } else if let v = try? values.decode(Float.self) {
      self = .float(v)
    } else if let v = try? values.decode([Int].self) {
      self = .ints(v)
    } else if let v = try? values.decode([Float].self) {
      self = .floats(v)
    } else if let v = try? values.decode(Bool.self) {
      self = .bool(v)
    } else {
      let v = try values.decode(String.self)
      self = .string(v)
    }
  }

  func encode(to encoder: Encoder) throws {
    var container = encoder.singleValueContainer()
    switch self {
    case .string(let v): try container.encode(v)
    case .int(let v): try container.encode(v)
    case .float(let v): try container.encode(v)
    case .ints(let v): try container.encode(v)
    case .floats(let v): try container.encode(v)
    case .bool(let v): try container.encode(v)
    }
  }

  func asFloat() -> Float? {
    switch self {
    case .string: nil
    case .int(let v): Float(v)
    case .float(let v): v
    case .ints(let a): a.count == 1 ? Float(a[0]) : nil
    case .floats(let a): a.count == 1 ? a[0] : nil
    case .bool(let b): b ? 1.0 : 0.0
    }
  }
}

// MARK: - KVCache protocol (from KVCache.swift)

/// Interface for Key/Value cache for LLMs.
protocol KVCache: Evaluatable {
  var offset: Int { get }
  var maxSize: Int? { get }
  func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
  var state: [MLXArray] { get set }
  var metaState: [String] { get set }
  var isTrimmable: Bool { get }
  @discardableResult func trim(_ n: Int) -> Int
  func makeMask(
    n: Int, windowSize: Int?, returnArray: Bool
  ) -> MLXFast.ScaledDotProductAttentionMaskMode
  func copy() -> any KVCache
}

// MARK: - BaseKVCache (from KVCache.swift)
//
// Note: `open` is removed here (would require `public` protocol, which we don't want);
// `class` with `func` (= `internal`) is sufficient for our module-internal use.

class BaseKVCache: KVCache {
  var offset: Int = 0
  var maxSize: Int? { nil }

  func innerState() -> [MLXArray] { [] }

  func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
    fatalError("update(keys:values:) must be implemented by subclass")
  }

  var state: [MLXArray] {
    get { [] }
    set {
      if !newValue.isEmpty {
        fatalError("This cache has no state but a state was set.")
      }
    }
  }

  var metaState: [String] {
    get { [""] }
    set {
      guard newValue.count == 1 && newValue[0].isEmpty else {
        fatalError("This cache has no meta_state but a meta_state was set.")
      }
    }
  }

  var isTrimmable: Bool { false }

  @discardableResult
  func trim(_ n: Int) -> Int { 0 }

  func copy() -> any KVCache {
    fatalError("copy() must be implemented by subclass")
  }

  func makeMask(
    n: Int, windowSize: Int?, returnArray: Bool
  ) -> MLXFast.ScaledDotProductAttentionMaskMode {
    if n == 1 { return .none }
    if returnArray || (windowSize != nil && n > windowSize!) {
      return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
    }
    return .causal
  }
}

// MARK: - createCausalMask (from KVCache.swift)

func createCausalMask(
  n: Int,
  offset: Int,
  windowSize: Int? = nil
) -> MLXArray {
  var rinds = MLXArray(Int32(0)..<Int32(offset + n))
  var linds = offset != 0 ? MLXArray(Int32(offset)..<Int32(offset + n)) : rinds
  linds = linds[0..., .newAxis]
  rinds = rinds[.newAxis]
  var mask = linds .>= rinds
  if let windowSize {
    mask = mask & (linds .< rinds + windowSize)
  }
  return mask
}

// MARK: - createAttentionMask (from KVCache.swift)

/// Create an attention mask (single-cache variant).
func createAttentionMask(
  h: MLXArray,
  cache: KVCache?,
  windowSize: Int? = nil,
  returnArray: Bool = false
) -> MLXFast.ScaledDotProductAttentionMaskMode {
  let n = h.dim(1)
  if let cache = cache {
    return cache.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
  }
  if n == 1 { return .none }
  if returnArray || (windowSize != nil && n > windowSize!) {
    return .array(createCausalMask(n: n, offset: 0, windowSize: windowSize))
  }
  return .causal
}

// MARK: - KVCacheSimple (from KVCache.swift)

/// Standard KV cache implementation.
final class KVCacheSimple: BaseKVCache {
  var keys: MLXArray?
  var values: MLXArray?
  var step = 256

  override init() {
    super.init()
  }

  override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
    let previous = self.offset

    let reset: Bool
    if let currentKeys = self.keys, (previous + keys.dim(2)) > currentKeys.dim(2) {
      reset = true
    } else {
      reset = self.keys == nil
    }

    if reset {
      let batchSize = keys.dim(0)
      let kvHeads = keys.dim(1)
      let kHeadDim = keys.dim(3)
      let vHeadDim = values.dim(3)

      let nSteps = (step + keys.dim(2) - 1) / step
      let kShape = [batchSize, kvHeads, nSteps * step, kHeadDim]
      let vShape = [batchSize, kvHeads, nSteps * step, vHeadDim]
      let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
      let newV = MLXArray.zeros(vShape, dtype: values.dtype)

      if var currentKeys = self.keys, var currentValues = self.values {
        if previous % step != 0 {
          currentKeys = currentKeys[.ellipsis, ..<previous, 0...]
          currentValues = currentValues[.ellipsis, ..<previous, 0...]
        }
        self.keys = concatenated([currentKeys, newK], axis: 2)
        self.values = concatenated([currentValues, newV], axis: 2)
      } else {
        self.keys = newK
        self.values = newV
      }
    }

    self.offset += keys.dim(2)

    self.keys?[.ellipsis, previous..<self.offset, 0...] = keys
    self.values?[.ellipsis, previous..<self.offset, 0...] = values

    let returnedKeys = self.keys![.ellipsis, ..<self.offset, 0...]
    let returnedValues = self.values![.ellipsis, ..<self.offset, 0...]

    return (returnedKeys, returnedValues)
  }

  override var state: [MLXArray] {
    get {
      guard let keys = self.keys, let values = self.values else { return [] }
      if offset == keys.dim(2) {
        return [keys, values]
      } else {
        return [
          keys[.ellipsis, ..<offset, 0...],
          values[.ellipsis, ..<offset, 0...],
        ]
      }
    }
    set {
      guard newValue.count == 2 else {
        fatalError("KVCacheSimple state must have exactly 2 arrays (keys, values)")
      }
      self.keys = newValue[0]
      self.values = newValue[1]
      self.offset = self.keys!.dim(2)
    }
  }

  override var isTrimmable: Bool { true }

  @discardableResult
  override func trim(_ n: Int) -> Int {
    let trimmed = min(offset, n)
    offset -= trimmed
    return trimmed
  }

  override func copy() -> any KVCache {
    let new = KVCacheSimple()
    new.step = self.step
    let s = self.state
    if !s.isEmpty {
      new.state = s.map { $0[.ellipsis] }
    }
    return new
  }
}

// MARK: - attentionWithCacheUpdate (from AttentionUtils.swift)
//
// Note: QuantizedKVCacheProtocol path is omitted — ASRPipeline uses KVCacheSimple only.
// Model weights may be quantized (Linear / QuantizedLinear), but the KV cache itself
// is never quantized in TinyAudio's inference path.

func attentionWithCacheUpdate(
  queries: MLXArray,
  keys: MLXArray,
  values: MLXArray,
  cache: KVCache?,
  scale: Float,
  mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> MLXArray {
  guard let cache else {
    return MLXFast.scaledDotProductAttention(
      queries: queries, keys: keys, values: values, scale: scale, mask: mask)
  }
  let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
  return MLXFast.scaledDotProductAttention(
    queries: queries, keys: cachedKeys, values: cachedValues, scale: scale, mask: mask)
}

// MARK: - RoPELayer typealias (from RoPEUtils.swift)

typealias RoPELayer = OffsetLayer & ArrayOffsetLayer

// MARK: - BatchPositionedKVCache + applyRotaryPosition (from RoPEApplication.swift)

/// Protocol for KV caches that expose per-sequence RoPE offsets.
protocol BatchPositionedKVCache: KVCache {
  var batchOffset: MLXArray { get }
}

func applyRotaryPosition<R: RoPELayer>(_ rope: R, to x: MLXArray, cache: KVCache?) -> MLXArray {
  if let batchCache = cache as? BatchPositionedKVCache {
    return rope(x, offset: batchCache.batchOffset)
  } else {
    return rope(x, offset: cache?.offset ?? 0)
  }
}

// MARK: - LoRAModel (from Adapters/LoRA/LoRAModel.swift)

/// Marker protocol for models that support LoRA fine-tuning.
protocol LoRAModel {
  var loraLayers: [Module] { get }
  var loraDefaultKeys: [String] { get }
}

extension LoRAModel {
  var loraDefaultKeys: [String] {
    let namedModules = loraLayers.flatMap { $0.namedModules() }
    let linearKeys = namedModules.compactMap { key, module -> String? in
      module is Linear ? key : nil
    }
    return Array(Set(linearKeys))
  }
}
