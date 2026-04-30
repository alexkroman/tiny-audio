// swift/Tests/TinyAudioTests/BundleResolver.swift
import Foundation

enum BundleResolver {
    /// Best-effort path to a downloaded tiny-audio-mlx bundle. Checks the
    /// standard Python HF cache first (`~/.cache/huggingface/hub/...`), then
    /// the Apple-convention path (`~/Library/Caches/huggingface/hub/...`).
    /// Returns nil if neither has the bundle — caller should skip the test.
    static func locate(repoID: String = "mazesmazes/tiny-audio-mlx") -> URL? {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let prefix = "models--" + repoID.replacingOccurrences(of: "/", with: "--")
        let candidates = [
            home.appendingPathComponent(".cache/huggingface/hub").appendingPathComponent(prefix).appendingPathComponent("snapshots"),
            home.appendingPathComponent("Library/Caches/huggingface/hub").appendingPathComponent(prefix).appendingPathComponent("snapshots"),
        ]
        for snapshots in candidates {
            guard let entries = try? FileManager.default.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: [.contentModificationDateKey]) else { continue }
            // Pick most recently modified snapshot.
            if let latest = entries.sorted(by: { lhs, rhs in
                ((try? lhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast)
                  > ((try? rhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast)
            }).first {
                return latest
            }
        }
        return nil
    }
}
