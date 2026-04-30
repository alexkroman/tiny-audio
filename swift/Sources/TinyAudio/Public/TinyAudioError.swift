import Foundation

/// Errors thrown by `TinyAudio`. Full case list lands in Task 22; this stub
/// covers what early tasks need.
public enum TinyAudioError: Error, Equatable {
    case audioFormatUnsupported(reason: String)
    case audioEmpty
    /// Prompt `<audio>` placeholder count does not match projector output length.
    case promptAudioTokenMismatch(prompt: Int, projector: Int)
}
