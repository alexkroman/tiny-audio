// swift/Tests/TinyAudioTests/SpliceTests.swift
import Foundation
import MLX
import Testing
@testable import TinyAudio

@Suite("Splice")
struct SpliceTests {
    /// Synthetic test: build a known text-embed tensor and audio-embed tensor,
    /// splice at known positions, assert positional correctness.
    @Test func spliceLandsAtCorrectPositions() {
        // text_embeds [1, T=5, D=3] = [[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
        let textRows: [Float] = [0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4]
        let textEmbeds = MLXArray(textRows, [1, 5, 3])

        // audio_embeds [N=2, D=3] = [[10,10,10],[20,20,20]] at positions [1, 3]
        let audioRows: [Float] = [10,10,10, 20,20,20]
        let audioEmbeds = MLXArray(audioRows, [2, 3])

        let result = AudioEmbeddingSplice.splice(
            textEmbeds: textEmbeds,
            audioEmbeds: audioEmbeds,
            audioPositions: [1, 3]
        )

        let expected: [Float] = [0,0,0, 10,10,10, 2,2,2, 20,20,20, 4,4,4]
        MLX.eval(result)
        let resultFlat = result.asArray(Float.self)
        #expect(resultFlat == expected, "splice produced unexpected values: \(resultFlat)")
    }
}
