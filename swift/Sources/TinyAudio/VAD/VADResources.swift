import Foundation

enum VADResources {
    /// URL of the bundled Silero VAD Core ML model. Lives in TinyAudio's
    /// resource bundle so `Bundle.module` (which resolves resources for the
    /// declaring target) finds it. Returns nil if the resource is missing —
    /// callers should throw a sensible error.
    static var sileroVADURL: URL? {
        Bundle.module.url(forResource: "silero_vad", withExtension: "mlpackage")
    }
}
