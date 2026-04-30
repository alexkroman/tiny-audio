import Accelerate
import Foundation
import MLX

/// Compute the 128-mel log-mel spectrogram of a 16 kHz mono `[Float]` array.
/// Matches `transformers.WhisperFeatureExtractor` (no 30-second padding) so the
/// number of frames reflects actual audio duration.
///
/// Output shape: `MLXArray[1, n_mels, T_mel]` with `T_mel = 1 + (samples - n_fft) / hop_length`
/// (truncated; samples are zero-padded by hop_length//2 on both sides like librosa.stft center=True).
struct LogMelSpectrogram {
    let nMels: Int
    let nFFT: Int
    let hopLength: Int
    private let melFilters: [Float]      // [nMels * (nFFT/2 + 1)] row-major
    private let window: [Float]          // Hann
    private let dftSetup: vDSP_DFT_Setup

    init(resourcePath: URL) throws {
        let data = try Data(contentsOf: resourcePath)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        self.nMels = json["n_mels"] as! Int
        self.nFFT = json["n_fft"] as! Int
        self.hopLength = json["hop_length"] as! Int
        let raw = json["filters"] as! [Double]
        self.melFilters = raw.map { Float($0) }

        // Symmetric Hann window of length nFFT.
        var w = [Float](repeating: 0, count: nFFT)
        vDSP_hann_window(&w, vDSP_Length(nFFT), Int32(vDSP_HANN_NORM))
        self.window = w

        guard let setup = vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(nFFT), .FORWARD) else {
            throw TinyAudioError.audioFormatUnsupported(reason: "vDSP_DFT setup failed for nFFT=\(nFFT)")
        }
        self.dftSetup = setup
    }

    static func loadDefault() throws -> LogMelSpectrogram {
        guard let url = Bundle.module.url(forResource: "MelFilterbank", withExtension: "json") else {
            throw TinyAudioError.audioFormatUnsupported(reason: "MelFilterbank.json missing from bundle")
        }
        return try LogMelSpectrogram(resourcePath: url)
    }

    /// Run the spectrogram. Returns an `MLXArray[1, nMels, nFrames]`.
    func compute(_ samples: [Float]) -> MLXArray {
        let nBins = nFFT / 2 + 1
        let pad = nFFT / 2
        let total = samples.count + 2 * pad
        var padded = [Float](repeating: 0, count: total)
        samples.withUnsafeBufferPointer {
            padded.replaceSubrange(pad ..< pad + samples.count, with: UnsafeBufferPointer(start: $0.baseAddress, count: samples.count))
        }

        let nFrames = max(0, (total - nFFT) / hopLength + 1)
        var melOut = [Float](repeating: 0, count: nMels * nFrames)

        var realIn = [Float](repeating: 0, count: nFFT)
        var imagIn = [Float](repeating: 0, count: nFFT)
        var realOut = [Float](repeating: 0, count: nFFT)
        var imagOut = [Float](repeating: 0, count: nFFT)
        var power = [Float](repeating: 0, count: nBins)

        for f in 0 ..< nFrames {
            let start = f * hopLength
            for i in 0 ..< nFFT {
                realIn[i] = padded[start + i] * window[i]
                imagIn[i] = 0
            }
            vDSP_DFT_Execute(dftSetup, realIn, imagIn, &realOut, &imagOut)
            for k in 0 ..< nBins {
                power[k] = realOut[k] * realOut[k] + imagOut[k] * imagOut[k]
            }
            // Mel filterbank: melOut[m, f] = sum_k filters[m, k] * power[k]
            for m in 0 ..< nMels {
                var sum: Float = 0
                let row = m * nBins
                for k in 0 ..< nBins { sum += melFilters[row + k] * power[k] }
                melOut[m * nFrames + f] = sum
            }
        }

        // Log + Whisper normalization: log10(max(mel, 1e-10)); clamp(max - 8.0); / 4 + 1.
        for i in 0 ..< melOut.count {
            melOut[i] = log10(max(melOut[i], 1e-10))
        }
        var maxVal: Float = -Float.greatestFiniteMagnitude
        vDSP_maxv(melOut, 1, &maxVal, vDSP_Length(melOut.count))
        let floorVal = maxVal - 8.0
        for i in 0 ..< melOut.count {
            melOut[i] = (max(melOut[i], floorVal) + 4.0) / 4.0
        }

        return MLXArray(melOut, [1, nMels, nFrames])
    }
}
