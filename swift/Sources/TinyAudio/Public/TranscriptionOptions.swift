import Foundation

/// Knobs for a single transcribe call.
public struct TranscriptionOptions: Sendable {
    /// Cap on tokens generated. Default mirrors the Python `transcribe()` default.
    public var maxNewTokens: Int = 96
    /// Optional system prompt prepended to the chat template.
    public var systemPrompt: String? = nil

    public init(maxNewTokens: Int = 96, systemPrompt: String? = nil) {
        self.maxNewTokens = maxNewTokens
        self.systemPrompt = systemPrompt
    }

    public static let `default` = TranscriptionOptions()
}
