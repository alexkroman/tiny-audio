// swift/Sources/TinyAudio/VAD/SileroVAD.swift
import CoreML
import Foundation

/// Silero VAD inference wrapper. Stateful: LSTM hidden state is carried
/// across frames within an utterance. Call `reset()` at utterance boundaries
/// (handled automatically by `VADStreamer`).
final class SileroVAD {
    /// Number of audio samples per VAD frame.
    /// The bundled silero_vad.mlpackage expects shape [1, 576] for `audio`.
    static let frameSize = 576

    /// LSTM hidden-state dimension (matches Silero's architecture).
    private static let stateDim = 128

    private let model: MLModel
    private var hState: MLMultiArray
    private var cState: MLMultiArray

    init() throws {
        guard let url = VADResources.sileroVADURL else {
            throw TinyAudioError.vadModelMissing
        }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        do {
            // .mlpackage must be compiled to .mlmodelc before loading.
            let compiledURL = try MLModel.compileModel(at: url)
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
        precondition(frame.count == Self.frameSize,
                     "SileroVAD expects exactly \(Self.frameSize) samples per call, got \(frame.count)")

        // Model expects audio as [1, frameSize].
        let audioArray = try MLMultiArray(shape: [1, NSNumber(value: Self.frameSize)], dataType: .float32)
        for i in 0 ..< Self.frameSize {
            audioArray[i] = NSNumber(value: frame[i])
        }

        // Input keys: "audio", "h", "c"; outputs: "prob", "h_out", "c_out".
        // Verified via coremltools inspection of silero_vad.mlpackage.
        let input = SileroVADInput(audio: audioArray, h: hState, c: cState)
        let output: MLFeatureProvider
        do {
            output = try model.prediction(from: input)
        } catch {
            throw TinyAudioError.mlxModuleLoadFailed(name: "SileroVAD.process", underlying: AnyError(error))
        }

        // Extract prob.
        guard let probValue = output.featureValue(for: "prob") else {
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
        if let newH = output.featureValue(for: "h_out")?.multiArrayValue {
            hState = newH
        }
        if let newC = output.featureValue(for: "c_out")?.multiArrayValue {
            cState = newC
        }

        return prob
    }

    private static func zeroState() throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1, NSNumber(value: stateDim)], dataType: .float32)
        for i in 0 ..< stateDim {
            arr[i] = NSNumber(value: 0.0)
        }
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

    var featureNames: Set<String> { ["audio", "h", "c"] }
    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "audio": return MLFeatureValue(multiArray: audio)
        case "h": return MLFeatureValue(multiArray: h)
        case "c": return MLFeatureValue(multiArray: c)
        default: return nil
        }
    }
}
