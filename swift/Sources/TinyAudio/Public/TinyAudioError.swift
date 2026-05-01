import Foundation

/// Errors thrown by `TinyAudio`.
public enum TinyAudioError: Error, Sendable {
    // Load-time
    case weightDownloadFailed(underlying: AnyError)
    case manifestMismatch(file: String, expected: String, actual: String)
    case formatVersionUnsupported(found: Int, supported: ClosedRange<Int>)
    case mlxModuleLoadFailed(name: String, underlying: AnyError)

    // Runtime
    case audioFormatUnsupported(reason: String)
    case audioEmpty
    case promptAudioTokenMismatch(prompt: Int, projector: Int)
    case vadModelMissing
}

/// Sendable wrapper for any `Error`, allowing `TinyAudioError` to be `Sendable`
/// without requiring associated `Error` values to themselves be `Sendable`.
public struct AnyError: Error, Sendable, CustomStringConvertible {
    private let _underlying: any Error

    public init(_ error: any Error) {
        self._underlying = error
    }

    public var description: String { return String(describing: _underlying) }

    public var underlying: any Error { return _underlying }
}

extension TinyAudioError: Equatable {
    public static func == (lhs: TinyAudioError, rhs: TinyAudioError) -> Bool {
        switch (lhs, rhs) {
        case (.weightDownloadFailed, .weightDownloadFailed): return false
        case let (.manifestMismatch(lf, le, la), .manifestMismatch(rf, re, ra)):
            return lf == rf && le == re && la == ra
        case let (.formatVersionUnsupported(lf, ls), .formatVersionUnsupported(rf, rs)):
            return lf == rf && ls == rs
        case (.mlxModuleLoadFailed, .mlxModuleLoadFailed): return false
        case let (.audioFormatUnsupported(lr), .audioFormatUnsupported(rr)):
            return lr == rr
        case (.audioEmpty, .audioEmpty): return true
        case let (.promptAudioTokenMismatch(lp, lpr), .promptAudioTokenMismatch(rp, rpr)):
            return lp == rp && lpr == rpr
        case (.vadModelMissing, .vadModelMissing): return true
        default: return false
        }
    }
}

extension TinyAudioError: CustomStringConvertible {
    public var description: String {
        switch self {
        case let .weightDownloadFailed(err): return "weight download failed: \(err)"
        case let .manifestMismatch(file, expected, actual): return "manifest sha256 mismatch for \(file): expected \(expected), got \(actual)"
        case let .formatVersionUnsupported(found, supported): return "MLX format version \(found) not supported; SDK supports \(supported)"
        case let .mlxModuleLoadFailed(name, err): return "failed to load \(name): \(err)"
        case let .audioFormatUnsupported(reason): return "audio format unsupported: \(reason)"
        case .audioEmpty: return "audio is empty"
        case let .promptAudioTokenMismatch(prompt, projector): return "prompt has \(prompt) <audio> placeholders but projector emitted \(projector) frames"
        case .vadModelMissing: return "Silero VAD mlpackage is missing from the bundle"
        }
    }
}
