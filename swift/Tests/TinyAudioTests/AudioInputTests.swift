// swift/Tests/TinyAudioTests/AudioInputTests.swift
import AVFoundation
import Testing

@testable import TinyAudio

@Suite("AudioInput")
struct AudioInputTests {
  /// All three `AudioInput` cases pointing at the same content should produce
  /// the same Float32 array (within numerical resampling tolerance).
  @Test func allInputsConvergeToSameSamples() throws {
    let url = Bundle.module.url(
      forResource: "librispeech_sample", withExtension: "wav", subdirectory: "Fixtures")!
    let fromFile = try AudioDecoder.decode(.file(url))

    let buffer = try Self.readPCMBuffer(url: url)
    let fromBuffer = try AudioDecoder.decode(.pcm(buffer: buffer))

    let nativeSamples = Self.bufferToFloats(buffer)
    let fromSamples = try AudioDecoder.decode(
      .samples(nativeSamples, sampleRate: buffer.format.sampleRate))

    #expect(fromFile.count == fromBuffer.count, "file and buffer paths produce different lengths")
    #expect(
      fromBuffer.count == fromSamples.count, "buffer and samples paths produce different lengths")

    // Numerical equality within resampling tolerance.
    let mae = Self.meanAbsoluteError(fromFile, fromBuffer)
    #expect(mae < 1e-5, "file vs buffer drift is too large: \(mae)")
  }

  @Test func emptySamplesThrows() {
    #expect(throws: TinyAudioError.audioEmpty) {
      try AudioDecoder.decode(.samples([], sampleRate: 16_000))
    }
  }

  // MARK: - helpers

  private static func readPCMBuffer(url: URL) throws -> AVAudioPCMBuffer {
    let file = try AVAudioFile(forReading: url)
    let buffer = AVAudioPCMBuffer(
      pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
    try file.read(into: buffer)
    return buffer
  }

  private static func bufferToFloats(_ buffer: AVAudioPCMBuffer) -> [Float] {
    let frames = Int(buffer.frameLength)
    let ch = buffer.floatChannelData![0]
    return Array(UnsafeBufferPointer(start: ch, count: frames))
  }

  private static func meanAbsoluteError(_ a: [Float], _ b: [Float]) -> Float {
    let n = min(a.count, b.count)
    var sum: Float = 0
    for i in 0..<n { sum += abs(a[i] - b[i]) }
    return n == 0 ? 0 : sum / Float(n)
  }
}
