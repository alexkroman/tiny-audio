// swift/Tests/TinyAudioTests/EncoderParityTests.swift
import Foundation
import MLX
import MLXNN
import Testing
@testable import TinyAudio

@Suite("EncoderParity")
struct EncoderParityTests {
    @Test func encoderMatchesPythonReference() throws {
        guard let bundle = BundleResolver.locate(repoID: "mazesmazes/tiny-audio-mlx") else {
            print("Skipping EncoderParityTests: tiny-audio-mlx bundle not cached.")
            return
        }

        // Load mel fixture as the encoder input. Shape: [1, 128, 601].
        let melShape = try FixtureLoader.shape(of: "mel")
        let melFloats = try FixtureLoader.loadFloat32(name: "reference_mel.bin")
        // GLMASREncoder.callAsFunction expects [B, nMels, T_mel] (NCL).
        let mel = MLXArray(melFloats, melShape)

        // Read encoder config from bundle/config.json.
        let configURL = bundle.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONSerialization.jsonObject(with: configData) as! [String: Any]
        let encDict = config["encoder"] as! [String: Any]

        let encConfig = GLMASREncoderConfig(
            nMels: encDict["n_mels"] as! Int,
            encoderDim: encDict["encoder_dim"] as! Int,
            numLayers: encDict["num_layers"] as! Int,
            numHeads: encDict["num_heads"] as! Int,
            headDim: encDict["head_dim"] as! Int,
            intermediateSize: encDict["intermediate_size"] as! Int,
            ropeTheta: Float(encDict["rope_theta"] as! Double)
        )

        // Build encoder and quantize BEFORE loading weights.
        // encoder.safetensors contains 4-bit quantized weights (group_size=64, bits=4).
        // quantize(model:groupSize:bits:) mutates in place, replacing Linear/Embedding
        // with QuantizedLinear/QuantizedEmbedding and adding .scales/.biases keys.
        let encoder = GLMASREncoder(encConfig)
        quantize(model: encoder, groupSize: 64, bits: 4)

        // Load weights from encoder.safetensors.
        let weightsURL = bundle.appendingPathComponent("encoder.safetensors")
        let weights = try MLX.loadArrays(url: weightsURL)
        try encoder.update(parameters: ModuleParameters.unflattened(weights), verify: .all)

        // Forward pass: [B, nMels, T_mel] -> [B, T_enc, encoderDim].
        let out = encoder(mel)
        MLX.eval(out)
        let outFloats = out.asArray(Float.self)

        // Load reference output. Shape: [1, 301, 1280].
        let refShape = try FixtureLoader.shape(of: "encoder_out")
        let refFloats = try FixtureLoader.loadFloat32(name: "reference_encoder_out.bin")

        #expect(out.shape == refShape, "encoder shape mismatch: swift=\(out.shape) ref=\(refShape)")
        #expect(outFloats.count == refFloats.count, "element count mismatch: \(outFloats.count) vs \(refFloats.count)")

        var maxDiff: Float = 0
        for (a, b) in zip(outFloats, refFloats) {
            maxDiff = max(maxDiff, abs(a - b))
        }
        // atol=2e-3: fp16 / 4-bit quantized tolerance.
        #expect(maxDiff < 2e-3, "encoder parity diff exceeds tolerance: \(maxDiff)")
    }
}
