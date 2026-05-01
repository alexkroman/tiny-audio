import Foundation
import MLX

enum AudioEmbeddingSplice {
  /// Replace `textEmbeds` rows at `audioPositions` with `audioEmbeds`.
  ///
  /// - Parameters:
  ///   - textEmbeds: `[1, T, D]` — text-embedded prompt sequence.
  ///   - audioEmbeds: `[N, D]` — projected audio frames; one row per `<audio>` token.
  ///   - audioPositions: 1-D Int positions where `audioEmbeds` should land.
  /// - Returns: `[1, T, D]` with audio embeddings spliced in.
  ///
  /// Mirrors `tiny_audio/mlx/model.py::splice_audio_embeds`. Builds an index
  /// gather + boolean mask so the whole splice stays on the GPU and chains
  /// into the prefill without forcing a host sync of `audioEmbeds`.
  static func splice(
    textEmbeds: MLXArray,
    audioEmbeds: MLXArray,
    audioPositions: [Int32]
  ) -> MLXArray {
    let t = textEmbeds.dim(1)
    let d = textEmbeds.dim(2)
    let n = audioPositions.count

    // idx[p] = i if p == positions[i] else n  (sentinel pointing to a zero row).
    // Built on CPU once per call; tiny.
    var idxNp = [Int32](repeating: Int32(n), count: t)
    for (i, pos) in audioPositions.enumerated() { idxNp[Int(pos)] = Int32(i) }
    let idx = MLXArray(idxNp)

    // [N+1, D]: append a zero row; positions not in `positions` index into it.
    let zeroRow = MLXArray.zeros([1, d], dtype: textEmbeds.dtype)
    let audioWithSentinel = MLX.concatenated(
      [audioEmbeds.asType(textEmbeds.dtype), zeroRow],
      axis: 0
    )

    // Gather: audioWithSentinel[idx] → [T, D], then add batch dim → [1, T, D]
    let audioFull = audioWithSentinel[idx].expandedDimensions(axis: 0)

    // Boolean mask: True at audio positions, False elsewhere.
    var maskNp = [Bool](repeating: false, count: t)
    for pos in audioPositions { maskNp[Int(pos)] = true }
    let mask = MLXArray(maskNp).reshaped([1, t, 1])

    return MLX.`where`(mask, audioFull, textEmbeds)
  }
}
