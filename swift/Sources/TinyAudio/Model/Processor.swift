import Foundation
import MLX
import Tokenizers

/// Audio-token-count + Qwen3 chat-template prompt construction. Mirrors
/// `tiny_audio/mlx/processor.py`.
enum Processor {
  static let audioToken = "<audio>"
  /// Suffix appended after the `<audio>` placeholders in the user turn.
  /// Must match `tiny_audio/mlx/processor.py:TRANSCRIBE_PROMPT` and the
  /// training-time prompt in `scripts/train.py:TRANSCRIBE_PROMPTS` — the
  /// model only learns to transcribe when this exact suffix follows the
  /// audio tokens.
  static let transcribePrompt = "Transcribe the speech to text"

  /// Apply the encoder's conv layers in order. Each layer:
  ///   `out = (in + 2p - (k-1) - 1) / s + 1`
  /// Default GLM-ASR conv layers: `[(p=1, k=3, s=1), (p=1, k=3, s=2)]`.
  static func encoderOutputLength(
    melLength: Int,
    convLayers: [(padding: Int, kernel: Int, stride: Int)]
  ) -> Int {
    var len = melLength
    for layer in convLayers {
      len = (len + 2 * layer.padding - (layer.kernel - 1) - 1) / layer.stride + 1
    }
    return len
  }

  /// Total number of `<audio>` placeholder tokens for a given audio length.
  /// Pipeline: audio_len → mel_frames (audio_len / hop_length) → encoder_frames
  /// (conv formula) → projector_output_length.
  static func numAudioTokens(
    audioLength: Int,
    convLayers: [(padding: Int, kernel: Int, stride: Int)],
    projector: MLPProjector,
    hopLength: Int = 160
  ) -> Int {
    let melLen = audioLength / hopLength
    let encLen = encoderOutputLength(melLength: melLen, convLayers: convLayers)
    return projector.outputLength(encLen)
  }

  /// Render the Qwen3 chat-template prompt with N `<audio>` placeholders.
  /// Returns input_ids as `MLXArray[1, T]` of Int32.
  ///
  /// Qwen3's chat template requires `enable_thinking=False` to suppress
  /// `<think>...</think>` reasoning blocks. ASR responses must use this.
  static func buildPromptInputIds(
    tokenizer: any Tokenizer,
    numAudioTokens: Int,
    systemPrompt: String? = nil
  ) throws -> MLXArray {
    let placeholder = String(repeating: audioToken, count: numAudioTokens)
    let userContent = placeholder + " " + transcribePrompt
    var messages: [Message] = []
    if let systemPrompt, !systemPrompt.isEmpty {
      messages.append(["role": "system", "content": systemPrompt])
    }
    messages.append(["role": "user", "content": userContent])

    // Use additionalContext to pass enable_thinking=false to the Jinja engine.
    // swift-transformers' applyChatTemplate supports additionalContext in
    // PreTrainedTokenizer; the protocol extension falls back gracefully.
    let ids = try tokenizer.applyChatTemplate(
      messages: messages,
      tools: nil,
      additionalContext: ["enable_thinking": false]
    )

    let int32 = ids.map { Int32($0) }
    return MLXArray(int32).expandedDimensions(axis: 0)
  }

  /// Hardcoded Qwen3 chat template that mirrors the Jinja template's
  /// `enable_thinking=False` output. Used as a fallback when the tokenizer
  /// does not have a chat template configured.
  static func fallbackQwen3PromptIds(
    tokenizer: any Tokenizer,
    numAudioTokens: Int,
    systemPrompt: String? = nil
  ) -> MLXArray {
    let placeholder = String(repeating: audioToken, count: numAudioTokens)
    let userContent = placeholder + " " + transcribePrompt
    var prompt = ""
    if let systemPrompt, !systemPrompt.isEmpty {
      prompt += "<|im_start|>system\n\(systemPrompt)<|im_end|>\n"
    }
    prompt += "<|im_start|>user\n\(userContent)<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    let ids = tokenizer.encode(text: prompt)
    let int32 = ids.map { Int32($0) }
    return MLXArray(int32).expandedDimensions(axis: 0)
  }
}
