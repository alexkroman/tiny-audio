import Foundation
import MLX
import MLXAudioCore

/// 128-mel log-mel spectrogram for our model. Thin wrapper around
/// `MLXAudioCore.computeMelSpectrogram`, the same function used by the
/// `GLMASR` model in mlx-audio-swift, with our model's hyperparameters
/// (16 kHz, n_fft=400, hop_length=160, n_mels=128).
///
/// Output shape: `MLXArray[1, nMels, T_mel]` (matches the encoder's expected
/// input — note that `computeMelSpectrogram` returns `[T_mel, nMels]`, which
/// we transpose and unsqueeze).
struct LogMelSpectrogram {
  static let nMels = 128
  static let nFFT = 400
  static let hopLength = 160
  static let sampleRate = 16_000

  func compute(_ samples: [Float]) -> MLXArray {
    let audio = MLXArray(samples)
    let mel = MLXAudioCore.computeMelSpectrogram(
      audio: audio,
      sampleRate: Self.sampleRate,
      nFft: Self.nFFT,
      hopLength: Self.hopLength,
      nMels: Self.nMels
    )
    // mel is [T_mel, nMels]. Transpose and add batch dim -> [1, nMels, T_mel].
    return mel.transposed().expandedDimensions(axis: 0)
  }

  /// CPU-friendly variant: returns the flat float array + shape so tests
  /// can compare without forcing Metal eval indirectly. Internally this
  /// still goes through MLX (Metal) because mlx-audio-swift's mel is
  /// MLX-based; the caller pays one eval at the boundary.
  func computeFloats(_ samples: [Float]) -> ([Float], shape: [Int]) {
    let mel = compute(samples)
    MLX.eval(mel)
    let floats = mel.asArray(Float.self)
    return (floats, shape: mel.shape)
  }

  /// Compatibility shim -- older code called `LogMelSpectrogram.loadDefault()`.
  /// Now that the filterbank is built inside `computeMelSpectrogram`, no
  /// resource is loaded. Returns a fresh instance.
  static func loadDefault() -> LogMelSpectrogram {
    LogMelSpectrogram()
  }
}
