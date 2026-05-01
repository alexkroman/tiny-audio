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

    init(vad: SileroVAD, config: VADConfig) {
        self.vad = vad
        self.config = config
        let preSpeechSamples = (config.preSpeechPaddingMs * 16_000) / 1000
        self.preSpeechRing = [Float](repeating: 0, count: max(0, preSpeechSamples))
    }

    /// Feed one VAD frame (`SileroVAD.frameSize` samples at 16 kHz). Returns 0 or 1 events.
    func process(_ frame: [Float]) throws -> [Event] {
        let prob = try vad.process(frame)
        let isSpeech = prob >= config.speechThreshold

        var events: [Event] = []
        switch state {
        case .idle:
            // Track pre-speech audio in a ring buffer using two contiguous slice
            // writes instead of 576 per-sample modulo operations.
            if !preSpeechRing.isEmpty {
                let n = preSpeechRing.count
                let head = preSpeechRingHead
                let firstChunk = min(frame.count, n - head)
                if firstChunk > 0 {
                    preSpeechRing.replaceSubrange(head ..< head + firstChunk, with: frame[..<firstChunk])
                }
                let remaining = frame.count - firstChunk
                if remaining > 0 {
                    if remaining <= n {
                        preSpeechRing.replaceSubrange(0 ..< remaining, with: frame[firstChunk ..< firstChunk + remaining])
                    } else {
                        // Frame larger than ring — only the last `n` samples survive.
                        preSpeechRing.replaceSubrange(0 ..< n, with: frame[(frame.count - n)...])
                    }
                }
                preSpeechRingHead = (head + frame.count) % n
            }
            if isSpeech {
                speechMs += Self.frameMs
                if speechMs >= config.minSpeechDurationMs {
                    // Onset confirmed. Seed utterance with pre-speech padding.
                    currentUtterance = readPreSpeechRing()
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

    private func readPreSpeechRing() -> [Float] {
        guard !preSpeechRing.isEmpty else { return [] }
        let n = preSpeechRing.count
        var out: [Float] = []
        out.reserveCapacity(n)
        for i in 0 ..< n {
            out.append(preSpeechRing[(preSpeechRingHead + i) % n])
        }
        return out
    }

    private enum State {
        case idle
        case speaking
    }
}
