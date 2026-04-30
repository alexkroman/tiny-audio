import AVFoundation
import Foundation

/// What the SDK accepts as audio. The SDK normalizes everything to 16 kHz
/// mono Float32 internally before running the model.
///
/// - Note: `AVAudioPCMBuffer` lacks a formal `Sendable` conformance in AVFoundation,
///   but is safe to pass between actors when the caller does not mutate the buffer
///   after handing it off.
public enum AudioInput: @unchecked Sendable {
    /// A file URL pointing to any format `AVAudioFile` can decode (wav, mp3, m4a, caf, flac).
    case file(URL)
    /// A pre-decoded PCM buffer at any sample rate / channel layout.
    case pcm(buffer: AVAudioPCMBuffer)
    /// Raw `Float32` samples at a known sample rate. Mono if 1D, deinterleaved-mono if 2D.
    case samples([Float], sampleRate: Double)
}
