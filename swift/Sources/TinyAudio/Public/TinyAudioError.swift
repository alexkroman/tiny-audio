import Foundation

/// Errors thrown by TinyAudio operations.
///
/// `TinyAudioError` is `Sendable` so it can be carried across actor
/// boundaries (e.g. forwarded through `AsyncStream<Event>.Continuation`).
/// Associated `Error` values are wrapped in ``AnyError`` to preserve
/// `Sendable` conformance without requiring every underlying error to be
/// `Sendable`.
public enum TinyAudioError: Error, Sendable {

  /// An MLX model component (encoder, projector, or decoder) could not be
  /// constructed or had its weights applied.
  ///
  /// - Parameters:
  ///   - name: The component name (`"encoder"`, `"projector"`, `"decoder"`,
  ///     or `"tokenizer"`).
  ///   - underlying: The root cause from MLX or the Swift runtime.
  case mlxModuleLoadFailed(name: String, underlying: AnyError)

  /// The audio container or codec is not supported by `AVAudioFile`.
  ///
  /// - Parameter reason: A human-readable description of the failure.
  case audioFormatUnsupported(reason: String)

  /// The decoded audio contained no samples.
  case audioEmpty

  /// The Silero VAD `.mlpackage` is missing from the SDK bundle.
  case vadModelMissing

  /// The user denied microphone access, or permission has never been granted.
  case micPermissionDenied

  /// `AVAudioEngine` or `AVAudioSession` configuration failed.
  case audioSessionConfigurationFailed(underlying: AnyError)

  /// The caller passed an empty or whitespace-only prompt to a text-generation
  /// API. The empty prompt would render to a degenerate chat template and the
  /// model's behavior would be unpredictable.
  case promptEmpty

  /// The caller passed an invalid argument value (e.g. `maxNewTokens <= 0`)
  /// that the SDK rejects before doing any model work.
  case invalidArgument(reason: String)
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
  /// Cases that carry an `AnyError` payload compare equal when the cases match,
  /// since the wrapped error has no meaningful equality.  Cases with primitive
  /// payloads compare structurally.
  public static func == (lhs: TinyAudioError, rhs: TinyAudioError) -> Bool {
    switch (lhs, rhs) {
    case (.mlxModuleLoadFailed(let l, _), .mlxModuleLoadFailed(let r, _)): return l == r
    case (.audioFormatUnsupported(let lr), .audioFormatUnsupported(let rr)): return lr == rr
    case (.audioEmpty, .audioEmpty): return true
    case (.vadModelMissing, .vadModelMissing): return true
    case (.micPermissionDenied, .micPermissionDenied): return true
    case (.audioSessionConfigurationFailed, .audioSessionConfigurationFailed): return true
    case (.promptEmpty, .promptEmpty): return true
    case (.invalidArgument(let lr), .invalidArgument(let rr)): return lr == rr
    default: return false
    }
  }
}

extension TinyAudioError: CustomStringConvertible {
  public var description: String {
    switch self {
    case .mlxModuleLoadFailed(let name, let err): return "failed to load \(name): \(err)"
    case .audioFormatUnsupported(let reason): return "audio format unsupported: \(reason)"
    case .audioEmpty: return "audio is empty"
    case .vadModelMissing: return "Silero VAD mlpackage is missing from the bundle"
    case .micPermissionDenied: return "microphone permission denied"
    case .audioSessionConfigurationFailed(let err):
      return "audio session configuration failed: \(err)"
    case .promptEmpty:
      return "Prompt was empty or whitespace-only."
    case .invalidArgument(let reason):
      return "Invalid argument: \(reason)"
    }
  }
}
