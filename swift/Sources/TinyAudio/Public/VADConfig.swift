// swift/Sources/TinyAudio/Public/VADConfig.swift
import Foundation

/// Silero Voice Activity Detection endpointer tuning for ``MicrophoneTranscriber``.
///
/// The defaults are calibrated for typical conversational speech at 16 kHz.
/// Increase ``minSilenceDurationMs`` to avoid splitting long natural pauses
/// into separate utterances; decrease it for faster endpointing in interview
/// or command-and-control scenarios.
///
/// ```swift
/// let config = VADConfig(
///     speechThreshold: 0.6,
///     minSilenceDurationMs: 800,
///     minSpeechDurationMs: 150,
///     preSpeechPaddingMs: 300
/// )
/// let mic = try MicrophoneTranscriber(transcriber: transcriber, vad: config)
/// ```
public struct VADConfig: Sendable {
    /// Silero confidence score above which a frame is classified as speech.
    ///
    /// Range `[0, 1]`.  The standard Silero recommendation is `0.5`.  Raise
    /// toward `0.7–0.8` in noisy environments to reduce false positives.
    public var speechThreshold: Float = 0.5

    /// Minimum consecutive silence (in milliseconds) before an utterance is
    /// declared complete.
    ///
    /// Shorter values produce faster endpointing but may split utterances at
    /// natural pauses.  Longer values tolerate pauses between words but add
    /// latency before transcription begins.
    public var minSilenceDurationMs: Int = 500

    /// Minimum consecutive speech (in milliseconds) before an onset is declared.
    ///
    /// Filters brief non-speech sounds (clicks, pops) from triggering an
    /// utterance.  Raise this value in environments with frequent impulsive
    /// noise.
    public var minSpeechDurationMs: Int = 200

    /// Audio captured *before* the detected speech onset to include in the
    /// utterance, in milliseconds.
    ///
    /// Prevents the first syllable of an utterance from being clipped.
    /// Requires ``minSpeechDurationMs`` of lookahead to be buffered.
    public var preSpeechPaddingMs: Int = 200

    /// Create a `VADConfig` with explicit field values.
    ///
    /// - Parameters:
    ///   - speechThreshold: Silero score threshold.  Defaults to `0.5`.
    ///   - minSilenceDurationMs: Silence duration before endpointing.  Defaults to `500`.
    ///   - minSpeechDurationMs: Speech duration before onset.  Defaults to `200`.
    ///   - preSpeechPaddingMs: Pre-roll audio before onset.  Defaults to `200`.
    public init(
        speechThreshold: Float = 0.5,
        minSilenceDurationMs: Int = 500,
        minSpeechDurationMs: Int = 200,
        preSpeechPaddingMs: Int = 200
    ) {
        self.speechThreshold = speechThreshold
        self.minSilenceDurationMs = minSilenceDurationMs
        self.minSpeechDurationMs = minSpeechDurationMs
        self.preSpeechPaddingMs = preSpeechPaddingMs
    }

    /// Default endpointer config: 0.5 threshold, 500 ms silence, 200 ms speech,
    /// 200 ms pre-roll.
    public static let `default` = VADConfig()
}
