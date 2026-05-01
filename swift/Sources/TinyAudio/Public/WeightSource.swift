import Foundation

/// Where the SDK should source model weights from.
public enum WeightSource: Sendable {
    /// Default Hub bundle (`mazesmazes/tiny-audio-mlx`).
    case defaultHub
    /// Explicit Hub repo (and optional revision).
    case hub(repoID: String, revision: String?)
    /// Pre-staged local directory containing the bundle's files.
    case localDirectory(URL)
}

extension WeightSource {
    static let defaultRepoID = "mazesmazes/tiny-audio-mlx"
}
