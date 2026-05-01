import Foundation

/// Errors thrown by TinyAudio operations.
///
/// Cases are grouped by lifecycle phase:
///
/// - **Load-time** errors (`weightDownloadFailed`, `manifestMismatch`,
///   `formatVersionUnsupported`, `mlxModuleLoadFailed`) are thrown only from
///   ``Transcriber/load(from:progress:)``.
/// - **Runtime** errors (`audioFormatUnsupported`, `audioEmpty`,
///   `promptAudioTokenMismatch`, `vadModelMissing`) are thrown during
///   transcription.
/// - **Microphone** errors (`micPermissionDenied`,
///   `audioSessionConfigurationFailed`) are thrown from
///   ``MicrophoneTranscriber/start()``.
///
/// `TinyAudioError` is `Sendable` so it can be carried across actor
/// boundaries (e.g. forwarded through `AsyncStream<Event>.Continuation`).
/// Associated `Error` values are wrapped in ``AnyError`` to preserve
/// `Sendable` conformance without requiring every underlying error to be
/// `Sendable`.
public enum TinyAudioError: Error, Sendable {

  // MARK: Load-time

  /// The HuggingFace Hub download failed.
  ///
  /// Inspect `underlying` for the network or file-system error.
  case weightDownloadFailed(underlying: AnyError)

  /// A file in the downloaded bundle has a SHA-256 hash that does not match
  /// the manifest.
  ///
  /// - Parameters:
  ///   - file: The relative path of the mismatched file inside the bundle.
  ///   - expected: The SHA-256 hex string from `manifest.json`.
  ///   - actual: The SHA-256 hex string computed from the file on disk.
  case manifestMismatch(file: String, expected: String, actual: String)

  /// The bundle's format version is outside the range this SDK version supports.
  ///
  /// - Parameters:
  ///   - found: The version integer read from `manifest.json`.
  ///   - supported: The closed range of versions this SDK can load.
  case formatVersionUnsupported(found: Int, supported: ClosedRange<Int>)

  /// An MLX model component (encoder, projector, or decoder) could not be
  /// constructed or had its weights applied.
  ///
  /// - Parameters:
  ///   - name: The component name (`"encoder"`, `"projector"`, `"decoder"`,
  ///     or `"tokenizer"`).
  ///   - underlying: The root cause from MLX or the Swift runtime.
  case mlxModuleLoadFailed(name: String, underlying: AnyError)

  // MARK: Runtime

  /// The audio container or codec is not supported by `AVAudioFile`.
  ///
  /// - Parameter reason: A human-readable description of the failure.
  case audioFormatUnsupported(reason: String)

  /// The decoded audio contained no samples.
  ///
  /// Thrown when the input file is empty, the PCM buffer has zero frames,
  /// or the `samples` array is empty.
  case audioEmpty

  /// The number of `<audio>` placeholder tokens in the prompt does not match
  /// the number of frames the projector emitted.
  ///
  /// This is an internal consistency check; it should not be reachable in
  /// production builds.
  ///
  /// - Parameters:
  ///   - prompt: Number of `<audio>` tokens found in the prompt.
  ///   - projector: Number of frames the projector produced.
  case promptAudioTokenMismatch(prompt: Int, projector: Int)

  /// The Silero VAD `.mlpackage` is missing from the SDK bundle.
  ///
  /// This should only occur in incorrectly assembled app bundles.
  case vadModelMissing

  // MARK: Microphone

  /// The user denied microphone access, or permission has never been granted.
  ///
  /// Direct the user to **Settings → Privacy → Microphone** to re-enable
  /// access.
  case micPermissionDenied

  /// `AVAudioEngine` or `AVAudioSession` configuration failed.
  ///
  /// - Parameter underlying: The AVFoundation or system error that caused
  ///   the failure.
  case audioSessionConfigurationFailed(underlying: AnyError)
}

/// A `Sendable` type-erased wrapper for any `Error`.
///
/// `TinyAudioError` cases that carry an associated `Error` (for example
/// ``TinyAudioError/weightDownloadFailed(underlying:)`` and
/// ``TinyAudioError/mlxModuleLoadFailed(name:underlying:)``) need to
/// themselves be `Sendable` so they can be transported across actor
/// boundaries.  Because arbitrary `Error` values are not `Sendable`,
/// those associated values are wrapped in `AnyError`, which uses
/// `@unchecked Sendable` storage internally.
///
/// Access the original error through ``underlying``.
///
/// ```swift
/// do {
///     let t = try await Transcriber.load()
/// } catch let TinyAudioError.weightDownloadFailed(anyErr) {
///     print(anyErr.underlying)  // the root URLError or similar
/// }
/// ```
public struct AnyError: Error, Sendable, CustomStringConvertible {
  private let _underlying: any Error

  /// Wrap an arbitrary error value.
  ///
  /// - Parameter error: The error to wrap.  The value is stored by reference;
  ///   no copying occurs.
  public init(_ error: any Error) {
    self._underlying = error
  }

  /// A human-readable description of the wrapped error.
  public var description: String { return String(describing: _underlying) }

  /// The original, unwrapped error value.
  public var underlying: any Error { return _underlying }
}

extension TinyAudioError: Equatable {
  public static func == (lhs: TinyAudioError, rhs: TinyAudioError) -> Bool {
    switch (lhs, rhs) {
    case (.weightDownloadFailed, .weightDownloadFailed): return false
    case (.manifestMismatch(let lf, let le, let la), .manifestMismatch(let rf, let re, let ra)):
      return lf == rf && le == re && la == ra
    case (.formatVersionUnsupported(let lf, let ls), .formatVersionUnsupported(let rf, let rs)):
      return lf == rf && ls == rs
    case (.mlxModuleLoadFailed, .mlxModuleLoadFailed): return false
    case (.audioFormatUnsupported(let lr), .audioFormatUnsupported(let rr)):
      return lr == rr
    case (.audioEmpty, .audioEmpty): return true
    case (.promptAudioTokenMismatch(let lp, let lpr), .promptAudioTokenMismatch(let rp, let rpr)):
      return lp == rp && lpr == rpr
    case (.vadModelMissing, .vadModelMissing): return true
    case (.micPermissionDenied, .micPermissionDenied): return true
    case (.audioSessionConfigurationFailed, .audioSessionConfigurationFailed): return false
    default: return false
    }
  }
}

extension TinyAudioError: CustomStringConvertible {
  public var description: String {
    switch self {
    case .weightDownloadFailed(let err): return "weight download failed: \(err)"
    case .manifestMismatch(let file, let expected, let actual):
      return "manifest sha256 mismatch for \(file): expected \(expected), got \(actual)"
    case .formatVersionUnsupported(let found, let supported):
      return "MLX format version \(found) not supported; SDK supports \(supported)"
    case .mlxModuleLoadFailed(let name, let err): return "failed to load \(name): \(err)"
    case .audioFormatUnsupported(let reason): return "audio format unsupported: \(reason)"
    case .audioEmpty: return "audio is empty"
    case .promptAudioTokenMismatch(let prompt, let projector):
      return "prompt has \(prompt) <audio> placeholders but projector emitted \(projector) frames"
    case .vadModelMissing: return "Silero VAD mlpackage is missing from the bundle"
    case .micPermissionDenied: return "microphone permission denied"
    case .audioSessionConfigurationFailed(let err):
      return "audio session configuration failed: \(err)"
    }
  }
}
