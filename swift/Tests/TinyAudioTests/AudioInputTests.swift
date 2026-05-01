// swift/Tests/TinyAudioTests/AudioInputTests.swift
import AVFoundation
import Testing

@testable import TinyAudio

@Suite("AudioInput")
struct AudioInputTests {
  /// `.file` and `.samples` pointing at the same audio should produce the
  /// same Float32 array (within numerical resampling tolerance).
  @Test func fileAndSamplesAgree() throws {
    let url = Bundle.module.url(
      forResource: "librispeech_sample", withExtension: "wav", subdirectory: "Fixtures")!
    let fromFile = try AudioDecoder.decode(.file(url))

    let buffer = try Self.readPCMBuffer(url: url)
    let nativeSamples = Self.bufferToFloats(buffer)
    let fromSamples = try AudioDecoder.decode(
      .samples(nativeSamples, sampleRate: buffer.format.sampleRate))

    #expect(fromFile.count == fromSamples.count, "file and samples paths produce different lengths")
    let mae = Self.meanAbsoluteError(fromFile, fromSamples)
    #expect(mae < 1e-5, "file vs samples drift is too large: \(mae)")
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
