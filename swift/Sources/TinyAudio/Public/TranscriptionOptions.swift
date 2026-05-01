import Foundation

/// Per-call tuning options for ``Transcriber/transcribe(_:options:)`` and
/// ``Transcriber/transcribeStream(_:options:)``.
///
/// Create a custom value when you need more tokens or a specific system
/// prompt; otherwise pass ``default``:
///
/// ```swift
/// let options = TranscriptionOptions(maxNewTokens: 256, systemPrompt: "Transcribe medical terminology accurately.")
/// let text = try await transcriber.transcribe(.file(url), options: options)
/// ```
///
/// The model always uses greedy decoding (`do_sample=false`, `num_beams=1`).
/// Temperature, top-p, and top-k are not supported.
public struct TranscriptionOptions: Sendable {
  /// Hard cap on the number of new tokens the decoder may generate.
  ///
  /// Default (96) mirrors the Python `transcribe()` default and is
  /// sufficient for utterances up to roughly 30–40 words.  Increase for
  /// longer recordings.
  public var maxNewTokens: Int = 96

  /// An optional system prompt prepended to the Qwen3 chat template.
  ///
  /// Use this to prime the model with domain-specific instructions (e.g.
  /// "Transcribe verbatim, preserving filler words.").  `nil` uses the
  /// model's default template without a system turn.
  public var systemPrompt: String? = nil

  /// Create a `TranscriptionOptions` value with explicit settings.
  ///
  /// - Parameters:
  ///   - maxNewTokens: Maximum tokens to generate.  Defaults to 96.
  ///   - systemPrompt: Optional system-turn text.  Defaults to `nil`.
  public init(maxNewTokens: Int = 96, systemPrompt: String? = nil) {
    self.maxNewTokens = maxNewTokens
    self.systemPrompt = systemPrompt
  }

  /// Default options: 96 max tokens, no system prompt.
  public static let `default` = TranscriptionOptions()
}
