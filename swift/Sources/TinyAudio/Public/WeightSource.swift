import Foundation

/// The location from which ``Transcriber/load(from:progress:)`` sources model weights.
///
/// The three cases cover the common deployment scenarios:
///
/// - ``defaultHub`` — production default; reads the bundled model weights
///   directly from `Bundle.module` — no download required.
/// - ``hub(repoID:revision:)`` — use a custom or forked Hub repository, or
///   pin to a specific git revision for reproducibility.
/// - ``localDirectory(_:)`` — use a pre-staged bundle; suitable for offline
///   use, unit tests, or embedding weights inside the app bundle.
public enum WeightSource: Sendable {
  /// The bundled model weights — no download required.
  ///
  /// Loads from the SDK's resource bundle (`Bundle.module/Resources/Model/`).
  /// Use ``hub(repoID:revision:)`` if you want a different model revision or
  /// a custom Hub repo.
  case defaultHub

  /// An explicit HuggingFace Hub repository, with an optional git revision.
  ///
  /// Pass a `revision` string (branch name, tag, or full commit SHA) to pin
  /// to an immutable snapshot.  `nil` resolves to the repository's `main`
  /// branch.
  ///
  /// - Parameters:
  ///   - repoID: The `owner/repo` identifier on HuggingFace Hub.
  ///   - revision: A branch name, tag, or commit SHA.  `nil` uses `main`.
  case hub(repoID: String, revision: String?)

  /// A local directory that already contains all required bundle files.
  ///
  /// The directory must contain `config.json`, `encoder.safetensors`,
  /// `projector.safetensors`, `decoder.safetensors`, `decoder_config.json`,
  /// `tokenizer.json`, `tokenizer_config.json`, and `manifest.json`.
  ///
  /// - Parameter _: A `file://` URL pointing to the bundle directory.
  case localDirectory(URL)
}

extension WeightSource {
  static let defaultRepoID = "mazesmazes/tiny-audio-mlx"
}
