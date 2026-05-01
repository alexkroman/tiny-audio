import Foundation

/// The audio formats accepted by ``Transcriber/transcribe(_:)`` and
/// ``MicrophoneTranscriber``.
///
/// All cases are normalised to 16 kHz mono Float32 by the SDK before running
/// the model.
public enum AudioInput: Sendable {
  /// A file URL whose audio will be decoded on the fly.
  ///
  /// Accepts any container that `AVAudioFile` can open: WAV, MP3, M4A,
  /// CAF, FLAC, AIFF, and more.
  case file(URL)

  /// Raw mono Float32 samples at a caller-specified sample rate.
  ///
  /// The SDK resamples from `sampleRate` to 16 kHz if it differs from
  /// 16 000.  The array is copied on first access, so it is safe to modify
  /// or release after the call returns.
  case samples([Float], sampleRate: Double)
}
