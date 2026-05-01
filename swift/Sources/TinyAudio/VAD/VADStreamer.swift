// swift/Sources/TinyAudio/VAD/VADStreamer.swift
import Foundation

/// Stateful endpointer. Consumes `SileroVAD.frameSize`-sample frames at 16 kHz, runs
/// them through `SileroVAD`, tracks running speech/silence durations, and
/// emits `.onset` / `.offset(audio:)` events on transitions.
final class VADStreamer {
  enum Event {
    /// Speech onset detected.
    case onset
    /// Speech offset detected; the buffered utterance audio is delivered.
    /// Audio includes pre-speech padding + the speech region.
    case offset(audio: [Float])
  }

  private let vad: SileroVAD
  private let config: VADConfig
  private var state: State = .idle
  private var currentUtterance: [Float] = []
  private var preSpeechRing: [Float]
  private var preSpeechRingHead = 0

  private var silenceMs = 0
  private var speechMs = 0

  /// Frame duration in ms (36 ms for 576 samples at 16 kHz).
  private static let frameMs = (SileroVAD.frameSize * 1000) / 16_000

  /// Capacity hint for `currentUtterance` at onset — sized for ~10 s of
  /// 16 kHz mono speech so typical utterances avoid mid-stream resizing.
  /// Beyond this the array grows naturally; this is not a hard cap.
  private static let utteranceCapacityHint = 16_000 * 10

  init(vad: SileroVAD, config: VADConfig) {
    self.vad = vad
    self.config = config
    let preSpeechSamples = (config.preSpeechPaddingMs * 16_000) / 1000
    self.preSpeechRing = [Float](repeating: 0, count: max(0, preSpeechSamples))
  }

  /// Feed one VAD frame (`SileroVAD.frameSize` samples at 16 kHz). Returns 0 or 1 events.
  func process(_ frame: ArraySlice<Float>) throws -> [Event] {
    let prob = try vad.process(frame)
    let isSpeech = prob >= config.speechThreshold

    var events: [Event] = []
    switch state {
    case .idle:
      writePreSpeechRing(frame)
      if isSpeech {
        speechMs += Self.frameMs
        if speechMs >= config.minSpeechDurationMs {
          // Onset confirmed. Seed utterance with pre-speech padding plus
          // this frame, then reserve room for typical-utterance growth.
          let preSpeech = readPreSpeechRing()
          currentUtterance.removeAll(keepingCapacity: true)
          currentUtterance.reserveCapacity(
            preSpeech.count + frame.count + Self.utteranceCapacityHint)
          currentUtterance.append(contentsOf: preSpeech)
          currentUtterance.append(contentsOf: frame)
          silenceMs = 0
          state = .speaking
          events.append(.onset)
        }
      } else {
        speechMs = 0
      }

    case .speaking:
      currentUtterance.append(contentsOf: frame)
      if isSpeech {
        silenceMs = 0
      } else {
        silenceMs += Self.frameMs
        if silenceMs >= config.minSilenceDurationMs {
          let audio = currentUtterance
          currentUtterance.removeAll(keepingCapacity: true)
          speechMs = 0
          silenceMs = 0
          state = .idle
          vad.reset()  // reset LSTM state at utterance boundary
          events.append(.offset(audio: audio))
        }
      }
    }
    return events
  }

  /// Write `frame` into the pre-speech ring using up to two contiguous
  /// slice copies — never a per-sample modulo loop.
  private func writePreSpeechRing(_ frame: ArraySlice<Float>) {
    let n = preSpeechRing.count
    guard n > 0 else { return }
    let frameStart = frame.startIndex
    if frame.count >= n {
      // Frame larger than (or equal to) ring — only the last `n` samples survive.
      preSpeechRing.replaceSubrange(0..<n, with: frame.suffix(n))
      preSpeechRingHead = 0
      return
    }
    let head = preSpeechRingHead
    let firstChunk = min(frame.count, n - head)
    preSpeechRing.replaceSubrange(
      head..<head + firstChunk,
      with: frame[frameStart..<frameStart + firstChunk])
    let remaining = frame.count - firstChunk
    if remaining > 0 {
      preSpeechRing.replaceSubrange(
        0..<remaining,
        with: frame[(frameStart + firstChunk)..<(frameStart + firstChunk + remaining)])
    }
    preSpeechRingHead = (head + frame.count) % n
  }

  /// Read the pre-speech ring as a linear array using two contiguous copies
  /// (head..end, then 0..head) — no per-sample modulo.
  private func readPreSpeechRing() -> [Float] {
    let n = preSpeechRing.count
    guard n > 0 else { return [] }
    var out = [Float]()
    out.reserveCapacity(n)
    out.append(contentsOf: preSpeechRing[preSpeechRingHead..<n])
    if preSpeechRingHead > 0 {
      out.append(contentsOf: preSpeechRing[0..<preSpeechRingHead])
    }
    return out
  }

  private enum State {
    case idle
    case speaking
  }
}
