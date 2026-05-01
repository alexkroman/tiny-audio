// swift/Tests/TinyAudioTests/VADTests.swift
import Foundation
import Testing

@testable import TinyAudio

@Suite("VAD")
struct VADTests {
  @Test func sileroDetectsSilenceVsSpeechLikeSignal() throws {
    let vad = try SileroVAD()
    let silence = [Float](repeating: 0, count: SileroVAD.frameSize)
    let probSilence = try vad.process(silence[...])
    #expect(probSilence < 0.1, "silence prob too high: \(probSilence)")

    // Reset state, then feed a modulated speech-like signal.
    vad.reset()
    let speech = makeSpeechLikeFrame()
    // Feed the same frame ~10x to let LSTM state warm up — Silero on a single
    // frame can be cold.
    var probs: [Float] = []
    for _ in 0..<10 {
      probs.append(try vad.process(speech[...]))
    }
    let maxProb = probs.max() ?? 0
    #expect(
      maxProb > probSilence, "speech-like signal should beat silence prob (got max=\(maxProb))")
  }

  @Test func streamerEmitsOnsetThenOffsetOnSilencePattern() throws {
    let vad = try SileroVAD()
    var config = VADConfig.default
    config.minSpeechDurationMs = 64  // 2 frames of confirmation
    config.minSilenceDurationMs = 192  // 6 frames of silence
    config.preSpeechPaddingMs = 96
    config.speechThreshold = 0.05  // permissive — synthetic isn't real speech

    let streamer = VADStreamer(vad: vad, config: config)

    let silenceFrame = [Float](repeating: 0, count: SileroVAD.frameSize)
    let speechFrame = makeSpeechLikeFrame()

    var allEvents: [VADStreamer.Event] = []
    // 5 silence frames -> 15 speech frames -> 10 silence frames.
    for _ in 0..<5 {
      allEvents.append(contentsOf: try streamer.process(silenceFrame[...]))
    }
    for _ in 0..<15 {
      allEvents.append(contentsOf: try streamer.process(speechFrame[...]))
    }
    for _ in 0..<10 {
      allEvents.append(contentsOf: try streamer.process(silenceFrame[...]))
    }

    let hasOnset = allEvents.contains {
      if case .onset = $0 { return true }
      return false
    }
    let hasOffset = allEvents.contains {
      if case .offset = $0 { return true }
      return false
    }
    #expect(hasOnset, "expected .onset, got: \(allEvents)")
    #expect(hasOffset, "expected .offset, got: \(allEvents)")
  }

  // MARK: - helpers

  private func makeSpeechLikeFrame() -> [Float] {
    var frame = [Float](repeating: 0, count: SileroVAD.frameSize)
    // AM-modulated tone: 200 Hz carrier × 5 Hz envelope. Closer to speech
    // than a pure tone but still synthetic.
    for i in 0..<frame.count {
      let t = Float(i) / 16_000
      frame[i] = 0.4 * sin(2 * .pi * 200 * t) * (0.5 + 0.5 * sin(2 * .pi * 5 * t))
    }
    return frame
  }
}
