import Foundation

enum VADResources {
  /// URL of the bundled Silero VAD Core ML model.
  ///
  /// Xcode's iOS build pipeline auto-compiles `.mlpackage` → `.mlmodelc`,
  /// while SwiftPM's `.copy` (used by `swift build` / `swift test` on macOS)
  /// preserves the `.mlpackage` directory verbatim. Try both extensions.
  /// Returns nil if neither is present — callers should throw a sensible error.
  static var sileroVADURL: URL? {
    Bundle.module.url(forResource: "silero_vad", withExtension: "mlmodelc")
      ?? Bundle.module.url(forResource: "silero_vad", withExtension: "mlpackage")
  }

  /// Whether the resolved URL is a precompiled `.mlmodelc` (true) or a raw
  /// `.mlpackage` that needs `MLModel.compileModel(at:)` (false).
  static func isPrecompiled(_ url: URL) -> Bool {
    url.pathExtension == "mlmodelc"
  }
}
