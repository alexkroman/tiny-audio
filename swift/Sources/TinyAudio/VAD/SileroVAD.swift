// swift/Sources/TinyAudio/VAD/SileroVAD.swift
import CoreML
import Foundation

// MARK: - Model key constants

private enum VADModelKeys {
  static let audio = "audio"
  static let h = "h"
  static let c = "c"
  static let prob = "prob"
  static let hOut = "h_out"
  static let cOut = "c_out"
}

/// Silero VAD inference wrapper. Stateful: LSTM hidden state is carried
/// across frames within an utterance. Call `reset()` at utterance boundaries
/// (handled automatically by `VADStreamer`).
final class SileroVAD {
  /// Number of audio samples per VAD frame.
  /// The bundled silero_vad.mlpackage expects shape [1, 576] for `audio`.
  static let frameSize = 576

  /// LSTM hidden-state dimension (matches Silero's architecture).
  private static let stateDim = 128

  /// Cache compiled model URL across instances so recompile only happens once per process.
  /// Protected by `compileLock`; nonisolated(unsafe) because the lock guarantees safety.
  private nonisolated(unsafe) static var cachedCompiledURL: URL?
  private static let compileLock = NSLock()

  private let model: MLModel
  private var hState: MLMultiArray
  private var cState: MLMultiArray

  init() throws {
    guard let url = VADResources.sileroVADURL else {
      throw TinyAudioError.vadModelMissing
    }
    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndNeuralEngine

    let compiledURL: URL
    Self.compileLock.lock()
    defer { Self.compileLock.unlock() }
    if let cached = Self.cachedCompiledURL, FileManager.default.fileExists(atPath: cached.path) {
      compiledURL = cached
    } else {
      do {
        compiledURL = try MLModel.compileModel(at: url)
        Self.cachedCompiledURL = compiledURL
      } catch {
        throw TinyAudioError.mlxModuleLoadFailed(name: "SileroVAD", underlying: AnyError(error))
      }
    }

    do {
      self.model = try MLModel(contentsOf: compiledURL, configuration: config)
    } catch {
      throw TinyAudioError.mlxModuleLoadFailed(name: "SileroVAD", underlying: AnyError(error))
    }

    self.hState = try Self.zeroState()
    self.cState = try Self.zeroState()
  }

  /// Reset LSTM state. Call at utterance boundaries.
  func reset() {
    hState = (try? Self.zeroState()) ?? hState
    cState = (try? Self.zeroState()) ?? cState
  }

  /// Run VAD on a frame of `frameSize` samples at 16 kHz. Returns speech probability.
  /// Updates internal LSTM state for the next call.
  func process(_ frame: [Float]) throws -> Float {
    precondition(
      frame.count == Self.frameSize,
      "SileroVAD expects exactly \(Self.frameSize) samples per call, got \(frame.count)")

    // Fill via the raw Float32 data pointer, no per-element NSNumber boxing.
    let audioArray = try MLMultiArray(
      shape: [1, NSNumber(value: Self.frameSize)], dataType: .float32)
    frame.withUnsafeBufferPointer { src in
      let dst = audioArray.dataPointer.assumingMemoryBound(to: Float32.self)
      dst.update(from: src.baseAddress!, count: Self.frameSize)
    }

    // Input keys: "audio", "h", "c"; outputs: "prob", "h_out", "c_out".
    // Verified via coremltools inspection of silero_vad.mlpackage.
    let input = SileroVADInput(audio: audioArray, h: hState, c: cState)
    let output: MLFeatureProvider
    do {
      output = try model.prediction(from: input)
    } catch {
      throw TinyAudioError.audioFormatUnsupported(reason: "SileroVAD prediction failed: \(error)")
    }

    // Extract prob.
    guard let probValue = output.featureValue(for: VADModelKeys.prob) else {
      throw TinyAudioError.audioFormatUnsupported(reason: "SileroVAD output missing 'prob'")
    }
    let prob: Float
    if let arr = probValue.multiArrayValue {
      prob = Float(truncating: arr[0])
    } else if probValue.type == .double {
      prob = Float(probValue.doubleValue)
    } else {
      throw TinyAudioError.audioFormatUnsupported(reason: "unexpected SileroVAD prob type")
    }

    // Update state.
    if let newH = output.featureValue(for: VADModelKeys.hOut)?.multiArrayValue {
      hState = newH
    }
    if let newC = output.featureValue(for: VADModelKeys.cOut)?.multiArrayValue {
      cState = newC
    }

    return prob
  }

  private static func zeroState() throws -> MLMultiArray {
    let arr = try MLMultiArray(shape: [1, NSNumber(value: stateDim)], dataType: .float32)
    // Fill via the raw Float32 data pointer, no per-element NSNumber boxing.
    let dst = arr.dataPointer.assumingMemoryBound(to: Float32.self)
    dst.initialize(repeating: 0, count: stateDim)
    return arr
  }
}

/// Manual MLFeatureProvider since we don't ship the auto-generated SileroVAD class.
private final class SileroVADInput: MLFeatureProvider {
  let audio: MLMultiArray
  let h: MLMultiArray
  let c: MLMultiArray

  init(audio: MLMultiArray, h: MLMultiArray, c: MLMultiArray) {
    self.audio = audio
    self.h = h
    self.c = c
  }

  var featureNames: Set<String> { [VADModelKeys.audio, VADModelKeys.h, VADModelKeys.c] }
  func featureValue(for featureName: String) -> MLFeatureValue? {
    switch featureName {
    case VADModelKeys.audio: return MLFeatureValue(multiArray: audio)
    case VADModelKeys.h: return MLFeatureValue(multiArray: h)
    case VADModelKeys.c: return MLFeatureValue(multiArray: c)
    default: return nil
    }
  }
}
