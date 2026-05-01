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

// MARK: - QuantizedKVCacheProtocol (from KVCache.swift)

protocol QuantizedKVCacheProtocol: KVCache {
  var groupSize: Int { get }
  var bits: Int { get }
  var mode: QuantizationMode { get }

  func updateQuantized(keys: MLXArray, values: MLXArray) -> (
    (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
  )

  func getQuantizedState() -> ((MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?))?
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

  func toQuantized(groupSize: Int = 64, bits: Int = 4) -> QuantizedKVCache {
    let q = QuantizedKVCache(groupSize: groupSize, bits: bits)
    q.offset = self.offset
    if let keys = self.keys, let values = self.values {
      let curK = keys[.ellipsis, ..<offset, 0...]
      let curV = values[.ellipsis, ..<offset, 0...]
      let qK = quantized(curK, groupSize: groupSize, bits: bits)
      let qV = quantized(curV, groupSize: groupSize, bits: bits)
      q.state = [qK.wq, qK.scales, qK.biases, qV.wq, qV.scales, qV.biases].compactMap { $0 }
    }
    return q
  }
}

// MARK: - QuantizedKVCache (from KVCache.swift)

final class QuantizedKVCache: BaseKVCache, QuantizedKVCacheProtocol {
  private var keys: (MLXArray, MLXArray, MLXArray?)?
  private var values: (MLXArray, MLXArray, MLXArray?)?
  private let step: Int
  let groupSize: Int
  let bits: Int
  let mode: QuantizationMode

  init(groupSize: Int = 64, bits: Int = 8, mode: QuantizationMode = .affine) {
    self.groupSize = groupSize
    self.bits = bits
    self.step = 256
    self.mode = mode
    super.init()
  }

  override func innerState() -> [MLXArray] {
    var arrays: [MLXArray] = []
    if let keys = keys {
      arrays.append(contentsOf: [keys.0, keys.1, keys.2].compactMap { $0 })
    }
    if let values = values {
      arrays.append(contentsOf: [values.0, values.1, values.2].compactMap { $0 })
    }
    return arrays
  }

  /// Tree map equivalent for applying function to tuple elements
  private func treeMap<T>(_ transform: (MLXArray) -> T, _ tuple: (MLXArray, MLXArray, MLXArray?))
    -> (T, T, T?)
  {
    if let biases = tuple.2 {
      return (transform(tuple.0), transform(tuple.1), transform(biases))
    } else {
      return (transform(tuple.0), transform(tuple.1), nil)
    }
  }

  /// Tree map for two tuples (like Python's tree_map over (keys, values))
  private func treeMapPair<T>(
    _ transform: (MLXArray) -> T, _ tuple1: (MLXArray, MLXArray, MLXArray?),
    _ tuple2: (MLXArray, MLXArray, MLXArray?)
  ) -> ((T, T, T?), (T, T, T?)) {
    return (treeMap(transform, tuple1), treeMap(transform, tuple2))
  }

  /// Create initial quantized tuples (like Python's init_quant)
  private func initQuant(dim: Int, shape: [Int], dtype: DType) -> (MLXArray, MLXArray, MLXArray?) {
    let tempArray = MLXArray.zeros(shape + [dim], dtype: dtype)
    let q = quantized(tempArray, groupSize: groupSize, bits: bits)
    return (q.wq, q.scales, q.biases)
  }

  /// Expand quantized tuple
  private func expandQuant(_ quantTuple: (MLXArray, MLXArray, MLXArray?), newShape: [Int]) -> (
    MLXArray, MLXArray, MLXArray?
  ) {
    return treeMap(
      { array in
        let newArray = MLXArray.zeros(newShape + [array.dim(-1)], dtype: array.dtype)
        return concatenated([array, newArray], axis: -2)
      }, quantTuple)
  }

  func getQuantizedState() -> (
    (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
  )? {
    guard let keys = keys, let values = values else { return nil }
    let trimmedKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, keys)
    let trimmedValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, values)
    return (trimmedKeys, trimmedValues)
  }

  func updateQuantized(keys: MLXArray, values: MLXArray) -> (
    (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
  ) {
    let batchSize = keys.dim(0)
    let nKVHeads = keys.dim(1)
    let numSteps = keys.dim(2)
    let kHeadDim = keys.dim(3)
    let vHeadDim = values.dim(3)
    let prev = offset

    if self.keys == nil || (prev + numSteps) > self.keys!.0.dim(-2) {
      let newSteps = ((step + numSteps - 1) / step) * step
      let shape = [batchSize, nKVHeads, newSteps]

      if let existingKeys = self.keys, let existingValues = self.values {
        if prev % step != 0 {
          let (trimmedKeys, trimmedValues) = treeMapPair(
            { array in
              array[.ellipsis, ..<prev, 0...]
            }, existingKeys, existingValues)

          self.keys = trimmedKeys
          self.values = trimmedValues
        }

        self.keys = expandQuant(self.keys!, newShape: shape)
        self.values = expandQuant(self.values!, newShape: shape)
      } else {
        self.keys = initQuant(dim: kHeadDim, shape: shape, dtype: keys.dtype)
        self.values = initQuant(dim: vHeadDim, shape: shape, dtype: keys.dtype)
      }
    }

    offset += numSteps

    let quantizedKeys = quantized(keys, groupSize: groupSize, bits: bits)
    let quantizedValues = quantized(values, groupSize: groupSize, bits: bits)

    let qKeys = (quantizedKeys.wq, quantizedKeys.scales, quantizedKeys.biases)
    let qValues = (quantizedValues.wq, quantizedValues.scales, quantizedValues.biases)

    guard let currentKeys = self.keys, let currentValues = self.values else {
      fatalError("Quantized cache not properly initialized")
    }

    currentKeys.0[.ellipsis, prev..<offset, 0...] = qKeys.0
    currentKeys.1[.ellipsis, prev..<offset, 0...] = qKeys.1
    if let qKeysBiases = qKeys.2 {
      currentKeys.2![.ellipsis, prev..<offset, 0...] = qKeysBiases
    }

    currentValues.0[.ellipsis, prev..<offset, 0...] = qValues.0
    currentValues.1[.ellipsis, prev..<offset, 0...] = qValues.1
    if let qValuesBiases = qValues.2 {
      currentValues.2![.ellipsis, prev..<offset, 0...] = qValuesBiases
    }

    self.keys = currentKeys
    self.values = currentValues

    let trimmedKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, currentKeys)
    let trimmedValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, currentValues)

    return (trimmedKeys, trimmedValues)
  }

  override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
    fatalError(
      "`update` was called on `QuantizedKVCache`. Use `updateQuantized` instead."
    )
  }

  override var state: [MLXArray] {
    get {
      guard let keys = keys, let values = values else { return [] }

      if offset < keys.0.dim(2) {
        let trimmedKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, keys)
        let trimmedValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, values)
        return [
          trimmedKeys.0, trimmedKeys.1, trimmedKeys.2, trimmedValues.0, trimmedValues.1,
          trimmedValues.2,
        ].compactMap { $0 }
      } else {
        return [keys.0, keys.1, keys.2, values.0, values.1, values.2].compactMap { $0 }
      }
    }
    set {
      switch newValue.count {
      case 4:
        keys = (newValue[0], newValue[1], nil)
        values = (newValue[2], newValue[3], nil)
      case 6:
        keys = (newValue[0], newValue[1], newValue[2])
        values = (newValue[3], newValue[4], newValue[5])
      default:
        fatalError(
          "QuantizedKVCache state must have exactly 6 or 4 arrays (3/2 for keys, 3/2 for values)"
        )
      }
    }
  }

  override var metaState: [String] {
    get { [String(step), String(offset), String(groupSize), String(bits)] }
    set {
      guard newValue.count == 4 else {
        fatalError("QuantizedKVCache metaState must have exactly 4 values")
      }
      self.offset = Int(newValue[1]) ?? 0
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
    let new = QuantizedKVCache(groupSize: groupSize, bits: bits, mode: mode)
    let s = self.state
    if !s.isEmpty {
      new.state = s.map { $0[.ellipsis] }
    }
    new.metaState = self.metaState
    return new
  }
}

// MARK: - attentionWithCacheUpdate (from AttentionUtils.swift)

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
  if let qCache = cache as? QuantizedKVCacheProtocol {
    let (qK, qV) = qCache.updateQuantized(keys: keys, values: values)
    return quantizedScaledDotProductAttention(
      queries: queries,
      quantizedKeys: qK,
      quantizedValues: qV,
      scale: scale,
      mask: mask,
      groupSize: qCache.groupSize,
      bits: qCache.bits,
      mode: qCache.mode
    )
  }
  let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
  return MLXFast.scaledDotProductAttention(
    queries: queries, keys: cachedKeys, values: cachedValues, scale: scale, mask: mask)
}

// MARK: - quantizedScaledDotProductAttention (from KVCache.swift)

func quantizedScaledDotProductAttention(
  queries: MLXArray,
  quantizedKeys: (MLXArray, MLXArray, MLXArray?),
  quantizedValues: (MLXArray, MLXArray, MLXArray?),
  scale: Float,
  mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
  groupSize: Int = 64,
  bits: Int = 8,
  mode: QuantizationMode = .affine
) -> MLXArray {

  let (B, nQHeads, L, D) = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3))
  let nKVHeads = quantizedKeys.0.dim(-3)
  let nRepeats = nQHeads / nKVHeads

  // Scale queries
  var scaledQueries = queries * scale

  // Handle GQA (Grouped Query Attention)
  var qKeys = quantizedKeys
  var qValues = quantizedValues
  if nRepeats > 1 {
    scaledQueries = scaledQueries.reshaped([B, nKVHeads, nRepeats, L, D])
    qKeys = (
      expandedDimensions(qKeys.0, axis: -3),
      expandedDimensions(qKeys.1, axis: -3),
      qKeys.2 == nil ? nil : expandedDimensions(qKeys.2!, axis: -3)
    )
    qValues = (
      expandedDimensions(qValues.0, axis: -3),
      expandedDimensions(qValues.1, axis: -3),
      qValues.2 == nil ? nil : expandedDimensions(qValues.2!, axis: -3)
    )
  }

  // Compute attention scores using quantized matmul
  var scores = quantizedMM(
    scaledQueries, qKeys.0, scales: qKeys.1, biases: qKeys.2,
    transpose: true, groupSize: groupSize, bits: bits,
    mode: mode
  )

  // Apply mask
  switch mask {
  case .causal:
    let (qL, kL) = (scores.dim(-2), scores.dim(-1))
    let qIndices = MLXArray(0..<qL) + MLXArray(kL - qL)
    let kIndices = MLXArray(0..<kL)
    let causalMask = greaterEqual(
      expandedDimensions(qIndices, axis: -1), expandedDimensions(kIndices, axis: -2))
    scores = MLX.where(causalMask, scores, MLXArray(Float.leastNormalMagnitude))

  case .array(let maskArray):
    if maskArray.dtype == .bool {
      scores = MLX.where(maskArray, scores, MLXArray(Float.leastNormalMagnitude))
    } else {
      scores = scores + maskArray
    }

  case .arrays(let maskArrays):
    if let maskArray = maskArrays.first {
      if maskArray.dtype == .bool {
        scores = MLX.where(maskArray, scores, MLXArray(Float.leastNormalMagnitude))
      } else {
        scores = scores + maskArray
      }
    }

  case .none:
    break
  }

  let attentionWeights = softmax(scores, axis: -1)

  // Compute output using quantized matmul
  var output = quantizedMM(
    attentionWeights, qValues.0, scales: qValues.1, biases: qValues.2,
    transpose: false, groupSize: groupSize, bits: bits,
    mode: mode
  )

  if nRepeats > 1 {
    output = output.reshaped([B, nQHeads, L, D])
  }

  return output
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

// MARK: - maybeQuantizeKVCache (from KVCache.swift)

/// Dynamically quantize KV caches during generation if conditions are met.
///
/// Converts regular caches to quantized caches when:
/// - kvBits is specified
/// - The cache is not already quantized
/// - The cache offset is greater than quantizedKVStart
func maybeQuantizeKVCache(
  cache: inout [KVCache],
  kvBits: Int?,
  kvGroupSize: Int = 64,
  quantizedKVStart: Int = 0
) {
  guard let kvBits = kvBits,
    !cache.isEmpty,
    !(cache[0] is QuantizedKVCache),
    cache[0].offset > quantizedKVStart
  else {
    return
  }

  for i in 0..<cache.count {
    if let simpleCache = cache[i] as? KVCacheSimple {
      cache[i] = simpleCache.toQuantized(groupSize: kvGroupSize, bits: kvBits)
    }
  }
}
