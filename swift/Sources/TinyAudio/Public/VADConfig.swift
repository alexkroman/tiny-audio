// swift/Sources/TinyAudio/Public/VADConfig.swift
import Foundation

/// Endpointer tuning. Defaults are sensible for typical conversational ASR.
public struct VADConfig: Sendable {
    /// Silero VAD output > this is considered "speech" for the current frame.
    /// Default 0.5 matches Silero's standard threshold.
    public var speechThreshold: Float = 0.5

    /// Minimum continuous silence (ms) before declaring an utterance has ended.
    public var minSilenceDurationMs: Int = 500

    /// Minimum continuous speech (ms) before declaring an utterance has started.
    public var minSpeechDurationMs: Int = 200

    /// Audio captured before the detected onset, in ms.
    public var preSpeechPaddingMs: Int = 200

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

    public static let `default` = VADConfig()
}
