// swift/Tests/TinyAudioTests/ProjectorParityTests.swift
import Foundation
import MLX
import MLXNN
import Testing

@testable import TinyAudio

@Suite("ProjectorParity")
struct ProjectorParityTests {
  @Test func projectorMatchesPythonReference() throws {
    guard let bundle = BundleResolver.locate() else {
      print("Skipping ProjectorParityTests: tiny-audio-mlx bundle not cached.")
      return
    }

    // Load encoder-out fixture as the projector input.
    let encShape = try FixtureLoader.shape(of: "encoder_out")
    let encFloats = try FixtureLoader.loadFloat32(name: "reference_encoder_out.bin")

    // Build projector from the bundle's config.json + load weights.
    let configURL = bundle.appendingPathComponent("config.json")
    let configData = try Data(contentsOf: configURL)
    let config = try JSONSerialization.jsonObject(with: configData) as! [String: Any]
    let projConfig = config["projector"] as! [String: Any]

    let projector = MLPProjector(
      encoderDim: projConfig["encoder_dim"] as! Int,
      llmDim: projConfig["llm_dim"] as! Int,
      hiddenDim: projConfig["hidden_dim"] as! Int,
      poolStride: projConfig["pool_stride"] as! Int
    )

    // Load weights from projector.safetensors.
    let weights = try MLX.loadArrays(url: bundle.appendingPathComponent("projector.safetensors"))
    try projector.update(parameters: ModuleParameters.unflattened(weights), verify: .all)

    // Build the input array from the float fixture.
    let encOut = MLXArray(encFloats, encShape)

    // Run the projector.
    let out = projector(encOut)

    // Load the reference projector output.
    let projShape = try FixtureLoader.shape(of: "projector_out")
    let referenceFloats = try FixtureLoader.loadFloat32(name: "reference_projector_out.bin")

    // Force eager evaluation before extracting float values.
    MLX.eval(out)
    let outFlat = out.asArray(Float.self)

    #expect(out.shape == projShape, "projector shape mismatch: swift=\(out.shape) ref=\(projShape)")
    #expect(outFlat.count == referenceFloats.count, "element count mismatch")

    var maxDiff: Float = 0
    for (a, b) in zip(outFlat, referenceFloats) {
      maxDiff = max(maxDiff, abs(a - b))
    }
    #expect(maxDiff < 1e-3, "projector parity diff exceeds tolerance: \(maxDiff)")
  }
}
