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

  struct PromptParts {
    /// Token IDs that come before the first `<audio>` placeholder. Constant
    /// for a given system prompt — safe to pre-prefill once.
    let prefixIds: [Int32]
    /// Token IDs that come after the last `<audio>` placeholder, terminating
    /// in the assistant `<think>...</think>` block. Length is constant for
    /// a fixed transcribe prompt.
    let suffixIds: [Int32]
  }

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

  /// Tokenize the constant chat-template prefix and suffix once. Audio
  /// embeddings sit between them; the full sequence is
  /// `prefix ++ <audio>×N ++ suffix`.
  static func buildPromptParts(
    tokenizer: any Tokenizer,
    systemPrompt: String? = nil
  ) -> PromptParts {
    var prefixText = ""
    if let systemPrompt, !systemPrompt.isEmpty {
      prefixText += "<|im_start|>system\n\(systemPrompt)<|im_end|>\n"
    }
    prefixText += "<|im_start|>user\n"

    let suffixText =
      " \(transcribePrompt)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    let prefixIds = tokenizer.encode(text: prefixText).map { Int32($0) }
    let suffixIds = tokenizer.encode(text: suffixText).map { Int32($0) }
    return PromptParts(prefixIds: prefixIds, suffixIds: suffixIds)
  }

  /// Render the full input_ids tensor with N `<audio>` placeholders, by
  /// concatenating prefix + audio + suffix. Mirrors the legacy
  /// `buildPromptInputIds` byte-for-byte; kept for callers that don't
  /// want the split form.
  static func buildPromptInputIds(
    tokenizer: any Tokenizer,
    numAudioTokens: Int,
    systemPrompt: String? = nil
  ) throws -> MLXArray {
    let parts = buildPromptParts(tokenizer: tokenizer, systemPrompt: systemPrompt)
    let audioId = tokenizer.convertTokenToId(audioToken)!
    let audioRun = [Int32](repeating: Int32(audioId), count: numAudioTokens)
    let all = parts.prefixIds + audioRun + parts.suffixIds
    return MLXArray(all).expandedDimensions(axis: 0)
  }
}
