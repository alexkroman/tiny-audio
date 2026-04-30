import Foundation

/// Errors thrown by `TinyAudio`. Full case list lands in Task 22; this stub
/// covers what early tasks need.
public enum TinyAudioError: Error {
    case audioFormatUnsupported(reason: String)
    case audioEmpty
}
