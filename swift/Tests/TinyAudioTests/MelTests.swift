// swift/Tests/TinyAudioTests/MelTests.swift
import AVFoundation
import Testing
@testable import TinyAudio

@Suite("Mel")
struct MelTests {
    @Test func melMatchesPythonReference() throws {
        let url = Bundle.module.url(forResource: "librispeech_sample", withExtension: "wav", subdirectory: "Fixtures")!
        let samples = try AudioDecoder.decode(.file(url))

        let mel = try LogMelSpectrogram.loadDefault()
        let (swiftFloats, swiftShape) = mel.computeFloats(samples)

        let referenceShape = try FixtureLoader.shape(of: "mel")
        let referenceFloats = try FixtureLoader.loadFloat32(name: "reference_mel.bin")

        #expect(swiftShape == referenceShape, "mel shape mismatch: swift=\(swiftShape) ref=\(referenceShape)")
        #expect(swiftFloats.count == referenceFloats.count, "element count mismatch: \(swiftFloats.count) vs \(referenceFloats.count)")

        var maxDiff: Float = 0
        for (a, b) in zip(swiftFloats, referenceFloats) {
            maxDiff = max(maxDiff, abs(a - b))
        }
        #expect(maxDiff < 1e-4, "mel max abs diff exceeds tolerance: \(maxDiff)")
    }
}
