import Foundation
import MLX
import MLXAudioCore

/// Whisper-compatible log-mel spectrogram (16 kHz, n_fft=400, hop=160, n_mels=128).
///
/// Built directly on top of mlx-audio-swift's `stft` + `melFilters` rather than
/// `computeMelSpectrogram` so the mel matches `WhisperFeatureExtractor` exactly:
///
/// - **Slaney mel scale** (piecewise linear + log above 1 kHz). `computeMelSpectrogram`
///   defaults to HTK, which produces a different filterbank above 1 kHz and was the
///   single biggest source of acoustic drift vs the trained-against Whisper mel.
/// - **Periodic Hann window** (`denom = N`). `MLXAudioCore.hanningWindow` is
///   symmetric (`denom = N − 1`); `torch.hann_window(N)` defaults to periodic.
/// - **Drop the last STFT frame**.
///   `WhisperFeatureExtractor._torch_extract_fbank_features` does
///   `stft[..., :-1].abs() ** 2`. Without the drop Swift's mel is 1 frame
///   longer than Whisper's for the same audio.
///
/// Output shape: `MLXArray[1, nMels, T_mel]`.
struct LogMelSpectrogram {
  static let nMels = 128
  static let nFFT = 400
  static let hopLength = 160
  static let sampleRate = 16_000

  /// Pre-built `[nFreqs, nMels]` Slaney filterbank — same for every call.
  private let filters: MLXArray
  /// Pre-built periodic Hann window of length `nFFT`.
  private let window: MLXArray

  init() {
    self.filters = melFilters(
      sampleRate: Self.sampleRate,
      nFft: Self.nFFT,
      nMels: Self.nMels,
      norm: "slaney",
      melScale: .slaney
    )
    self.window = Self.periodicHannWindow(size: Self.nFFT)
  }

  func compute(_ samples: [Float]) -> MLXArray {
    let audio = MLXArray(samples)
    let freqs = stft(
      audio: audio, window: window, nFft: Self.nFFT, hopLength: Self.hopLength
    )  // [numFrames, nFft/2 + 1]

    // Whisper drops the last STFT frame before squaring.
    let nFrames = freqs.shape[0]
    let trimmed = freqs[0..<(nFrames - 1)]

    let magnitudes = MLX.abs(trimmed).square()
    var melSpec = MLX.matmul(magnitudes, filters)  // [numFrames-1, nMels]

    // Whisper-style log + clip + normalize:
    //   log10(max(mel, 1e-10)) → max(., max - 8) → (. + 4) / 4
    melSpec = MLX.maximum(melSpec, MLXArray(Float(1e-10)))
    melSpec = MLX.log10(melSpec)
    let maxVal = melSpec.max()
    melSpec = MLX.maximum(melSpec, maxVal - MLXArray(Float(8.0)))
    melSpec = (melSpec + MLXArray(Float(4.0))) / MLXArray(Float(4.0))

    // [T_mel, nMels] → [1, nMels, T_mel]
    return melSpec.transposed().expandedDimensions(axis: 0)
  }

  /// Periodic Hann window matching `torch.hann_window(N)` (denom = N).
  /// `MLXAudioCore.hanningWindow` is symmetric (denom = N − 1).
  private static func periodicHannWindow(size: Int) -> MLXArray {
    var w = [Float](repeating: 0, count: size)
    let denom = Float(size)
    for n in 0..<size {
      w[n] = 0.5 * (1 - cos(2 * Float.pi * Float(n) / denom))
    }
    return MLXArray(w)
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
}
