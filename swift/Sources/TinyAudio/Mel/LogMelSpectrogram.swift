import Accelerate
import Foundation
import MLX

/// Compute the 128-mel log-mel spectrogram of a 16 kHz mono `[Float]` array.
///
/// Matches `transformers.WhisperFeatureExtractor` with 30-second implicit
/// zero-padding: reflect-pads by `n_fft/2` on the **left** only, zero-pads on
/// the **right**, then drops the last frame. This reproduces the reference
/// fixture exactly (atol < 1e-4 vs Python).
///
/// Background: the Python feature extractor zero-pads audio to 30 seconds before
/// STFT. `torch.stft(center=True, pad_mode='reflect')` then reflect-pads the
/// zero-padded signal, so the right center-pad is all zeros. We apply the same
/// asymmetric padding so the mel output matches without needing 30-second audio.
///
/// Because `n_fft=400` is not a power of 2, vDSP's DFT routines cannot be used.
/// Instead we precompute the real DFT matrix (cosine rows) and imaginary DFT
/// matrix (sine rows) and evaluate each frame with two `cblas_sgemv` calls.
/// For 400×201 bins × 600 frames this is fast on Apple Silicon.
///
/// Output shape: `MLXArray[1, n_mels, T_mel]` where
/// `T_mel = (samples + 2*(n_fft/2) - n_fft) / hop_length + 1 - 1`
/// (the trailing `- 1` mirrors `stft[..., :-1]` in the Whisper Python path).
struct LogMelSpectrogram {
    let nMels: Int
    let nFFT: Int
    let hopLength: Int
    private let melFilters: [Float]   // [nMels × nBins] row-major
    private let window: [Float]       // un-normalized Hann, length nFFT
    /// Cosine DFT matrix: [nBins × nFFT] row-major.
    private let dftCos: [Float]
    /// Sine DFT matrix (negative):  −sin, [nBins × nFFT] row-major.
    private let dftNegSin: [Float]

    init(resourcePath: URL) throws {
        let data = try Data(contentsOf: resourcePath)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        self.nMels = json["n_mels"] as! Int
        self.nFFT = json["n_fft"] as! Int
        self.hopLength = json["hop_length"] as! Int
        let raw = json["filters"] as! [Double]
        self.melFilters = raw.map { Float($0) }

        // --- Hann window (un-normalized, sum = nFFT/2, matches torch.hann_window) ---
        var w = [Float](repeating: 0, count: nFFT)
        vDSP_hann_window(&w, vDSP_Length(nFFT), Int32(vDSP_HANN_DENORM))
        self.window = w

        // --- Precompute the real DFT matrix ---
        // For each output frequency bin k (0 ..< nBins) and each input sample n:
        //   cos_mat[k, n] =  cos(2π·k·n / nFFT)   → real part
        //   neg_sin_mat[k, n] = -sin(2π·k·n / nFFT) → imaginary part (negated for FORWARD DFT)
        let nBins = nFFT / 2 + 1
        var cosMatrix = [Float](repeating: 0, count: nBins * nFFT)
        var negSinMatrix = [Float](repeating: 0, count: nBins * nFFT)
        let twoPiOverN = 2.0 * Double.pi / Double(nFFT)
        for k in 0 ..< nBins {
            for n in 0 ..< nFFT {
                let angle = twoPiOverN * Double(k) * Double(n)
                cosMatrix[k * nFFT + n] = Float(cos(angle))
                negSinMatrix[k * nFFT + n] = Float(-sin(angle))
            }
        }
        self.dftCos = cosMatrix
        self.dftNegSin = negSinMatrix
    }

    static func loadDefault() throws -> LogMelSpectrogram {
        guard let url = Bundle.module.url(forResource: "MelFilterbank", withExtension: "json") else {
            throw TinyAudioError.audioFormatUnsupported(reason: "MelFilterbank.json missing from bundle")
        }
        return try LogMelSpectrogram(resourcePath: url)
    }

    /// Run the spectrogram. Returns raw `([Float], shape: [Int])` without Metal.
    ///
    /// This is the core computation. `compute(_:)` wraps this in an `MLXArray`.
    /// Tests that run without a Metal device can call this directly.
    ///
    /// Python equivalent:
    /// ```python
    /// stft = torch.stft(waveform, n_fft, hop_length, window=hann_window, return_complex=True)
    /// magnitudes = stft[..., :-1].abs() ** 2          # drop last frame
    /// mel_spec = mel_filters.T @ magnitudes
    /// log_spec = clamp(mel_spec, min=1e-10).log10()
    /// log_spec = maximum(log_spec, log_spec.max() - 8.0)
    /// log_spec = (log_spec + 4.0) / 4.0
    /// ```
    func computeFloats(_ samples: [Float]) -> ([Float], shape: [Int]) {
        let nBins = nFFT / 2 + 1
        let pad = nFFT / 2

        // --- Asymmetric center padding (matches the WhisperFeatureExtractor reference) ---
        //
        // The Python feature extractor zero-pads audio to 30 seconds, then calls
        // torch.stft(center=True, pad_mode='reflect').  The 'reflect' pad at the
        // right end of the 30-second signal reflects zeros → stays zero.
        //
        // We reproduce this without allocating 30s: left-reflect the audio start,
        // right-zero-pad.  This gives identical mel values within atol=1e-4.
        let N = samples.count
        let totalLen = N + 2 * pad
        // Initialised to zero — right pad is implicitly zero.
        var padded = [Float](repeating: 0, count: totalLen)
        // Copy interior samples.
        for i in 0 ..< N {
            padded[pad + i] = samples[i]
        }
        // Left reflect: padded[pad - 1 - i] = samples[i + 1] for i in 0..<pad.
        // (numpy 'reflect' mode excludes the edge sample itself.)
        for i in 0 ..< pad {
            padded[pad - 1 - i] = samples[i + 1]
        }
        // Right pad stays zero (already initialised above).

        // --- Frame count: Python: num_frames = 1 + floor((padded_len - n_fft) / hop) ---
        // Then [:, :-1] drops the last frame.
        let nFramesRaw = max(0, (totalLen - nFFT) / hopLength + 1)
        let nFrames = max(0, nFramesRaw - 1)   // mirror stft[..., :-1]

        // melOut stored as [nMels, nFrames] row-major.
        var melOut = [Float](repeating: 0, count: nMels * nFrames)

        // Per-frame buffers.
        var frame = [Float](repeating: 0, count: nFFT)   // windowed samples
        var realPart = [Float](repeating: 0, count: nBins)
        var imagPart = [Float](repeating: 0, count: nBins)
        var power = [Float](repeating: 0, count: nBins)

        // --- DFT via BLAS sgemv: realPart = dftCos @ frame,  imagPart = dftNegSin @ frame ---
        // Matrix is [nBins × nFFT], vector is [nFFT], result is [nBins].
        dftCos.withUnsafeBufferPointer { cosPtr in
            dftNegSin.withUnsafeBufferPointer { sinPtr in
                melFilters.withUnsafeBufferPointer { filterPtr in
                    for f in 0 ..< nFrames {
                        let start = f * hopLength

                        // Apply window.
                        for i in 0 ..< nFFT {
                            frame[i] = padded[start + i] * window[i]
                        }

                        // Real part: cosine matrix × frame.
                        cblas_sgemv(
                            CblasRowMajor, CblasNoTrans,
                            Int32(nBins), Int32(nFFT),
                            1.0, cosPtr.baseAddress!, Int32(nFFT),
                            frame, 1,
                            0.0, &realPart, 1
                        )

                        // Imaginary part: (−sine) matrix × frame.
                        cblas_sgemv(
                            CblasRowMajor, CblasNoTrans,
                            Int32(nBins), Int32(nFFT),
                            1.0, sinPtr.baseAddress!, Int32(nFFT),
                            frame, 1,
                            0.0, &imagPart, 1
                        )

                        // Power spectrum.
                        for k in 0 ..< nBins {
                            power[k] = realPart[k] * realPart[k] + imagPart[k] * imagPart[k]
                        }

                        // Mel filterbank: melOut[m, f] = sum_k filters[m, k] * power[k]
                        for m in 0 ..< nMels {
                            var acc: Float = 0
                            let row = m * nBins
                            for k in 0 ..< nBins { acc += filterPtr[row + k] * power[k] }
                            melOut[m * nFrames + f] = acc
                        }
                    }
                }
            }
        }

        // --- Log + Whisper normalization ---
        // log10(max(mel, 1e-10)), clamp to (max - 8), scale to (x + 4) / 4.
        for i in 0 ..< melOut.count {
            melOut[i] = log10(max(melOut[i], 1e-10))
        }
        var maxVal: Float = -Float.greatestFiniteMagnitude
        vDSP_maxv(melOut, 1, &maxVal, vDSP_Length(melOut.count))
        let floorVal = maxVal - 8.0
        for i in 0 ..< melOut.count {
            melOut[i] = (max(melOut[i], floorVal) + 4.0) / 4.0
        }

        return (melOut, shape: [1, nMels, nFrames])
    }

    /// Run the spectrogram. Returns an `MLXArray[1, nMels, T_mel]`.
    ///
    /// Requires Metal (GPU). Call `computeFloats(_:)` instead when Metal is
    /// not available (e.g. unit tests running outside Xcode).
    func compute(_ samples: [Float]) -> MLXArray {
        let (floats, shape) = computeFloats(samples)
        return MLXArray(floats, shape)
    }
}
