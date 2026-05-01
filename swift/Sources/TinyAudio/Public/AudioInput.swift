import AVFoundation
import Foundation

/// The audio data formats accepted by ``Transcriber`` and ``MicrophoneTranscriber``.
///
/// All cases are normalised to 16 kHz mono Float32 by the SDK before running
/// the model.  Choose the case that best matches how your audio is already
/// represented:
///
/// - ``file(_:)`` — simplest for recorded files; the SDK handles decoding.
/// - ``pcm(buffer:)`` — use when you already have an `AVAudioPCMBuffer` from
///   `AVAudioEngine` or `AVAudioFile`.
/// - ``samples(_:sampleRate:)`` — use when you have raw samples from another
///   framework (e.g. SFSpeechRecognizer, CoreML, custom DSP).
///
/// ## Sendability
///
/// `AudioInput` is declared `@unchecked Sendable` because `AVAudioPCMBuffer`
/// does not conform to `Sendable` in AVFoundation.  The conformance is safe
/// provided the caller does not mutate the buffer after passing it to the SDK.
public enum AudioInput: @unchecked Sendable {
    /// A file URL whose audio will be decoded on the fly.
    ///
    /// Accepts any container that `AVAudioFile` can open: WAV, MP3, M4A,
    /// CAF, FLAC, AIFF, and more.  Decoding is performed on the calling thread
    /// during the first `await` inside ``Transcriber/transcribe(_:options:)``.
    case file(URL)
    /// A pre-decoded PCM buffer at any sample rate and channel layout.
    ///
    /// The SDK resamples and down-mixes to 16 kHz mono Float32 via
    /// `AVAudioConverter` if needed.  Do not mutate the buffer after passing
    /// it to the SDK.
    case pcm(buffer: AVAudioPCMBuffer)
    /// Raw mono Float32 samples at a caller-specified sample rate.
    ///
    /// The SDK resamples from `sampleRate` to 16 kHz if `sampleRate` differs
    /// from 16 000.  The array is copied on first access, so it is safe to
    /// modify or release after calling ``Transcriber/transcribe(_:options:)``.
    case samples([Float], sampleRate: Double)
}
