import Foundation
import Hub

struct HubLoader {
    private static let requiredGlobs: [String] = [
        "encoder.safetensors",
        "projector.safetensors",
        "decoder.safetensors",
        "tokenizer*",
        "config.json",
        "decoder_config.json",
        "manifest.json",
    ]

    /// Materialize a `WeightSource` into a local directory containing the
    /// expected files. Forwards download progress to the callback (if any).
    static func materialize(
        _ source: WeightSource,
        progress: ((Progress) -> Void)?
    ) async throws -> URL {
        switch source {
        case .defaultHub:
            return try await fetchHub(repoID: WeightSource.defaultRepoID, revision: nil, progress: progress)
        case let .hub(repoID, revision):
            return try await fetchHub(repoID: repoID, revision: revision, progress: progress)
        case let .localDirectory(url):
            return url
        }
    }

    private static func fetchHub(
        repoID: String,
        revision: String?,
        progress: ((Progress) -> Void)?
    ) async throws -> URL {
        do {
            let hub = HubApi()
            let repo = Hub.Repo(id: repoID)
            let rev = revision ?? "main"
            let folder = try await hub.snapshot(
                from: repo,
                revision: rev,
                matching: requiredGlobs,
                progressHandler: { p in progress?(p) }
            )
            return folder
        } catch let error as TinyAudioError {
            throw error
        } catch {
            throw TinyAudioError.weightDownloadFailed(underlying: AnyError(error))
        }
    }
}
