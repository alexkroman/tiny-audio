import Foundation

/// Ensures the MLX `mlx.metallib` file is present next to the running executable.
///
/// SwiftPM's `swift build` does not auto-generate `mlx.metallib` for executable
/// targets — only test bundles get it. So `swift run`-launched apps fail on
/// first MLX use unless the metallib is already in place.
///
/// Call this once before any MLX use, ideally in your `@main` struct's `main()`
/// (or `init()`) before any model loading. Safe to call multiple times.
///
/// On macOS, this searches for the metallib in (in order):
/// - The current executable's directory (already there → no-op)
/// - The project's debug test bundle (`.build/<arch>-apple-macosx/debug/TinyAudioPackageTests.xctest/Contents/MacOS/mlx.metallib`)
/// - The project's release build directory (`.build/<arch>-apple-macosx/release/mlx.metallib`)
/// - Debug/release build directories up to 8 directory levels above the executable
///
/// Copies the first candidate found to sit alongside the running executable.
/// Failures are silently swallowed — if no metallib is found, MLX's own error
/// surfaces on first use.
public enum MLXBootstrap {
  /// Locate `mlx.metallib` in nearby SwiftPM build artifacts and copy it
  /// next to the running executable if it is not already present.
  ///
  /// This is a no-op when:
  /// - The file already exists next to the executable.
  /// - The platform is not macOS (iOS/visionOS apps bundle the metallib via Xcode).
  public static func ensureMetallibAvailable() {
    #if os(macOS)
      let executableURL =
        Bundle.main.executableURL
        ?? URL(fileURLWithPath: CommandLine.arguments[0])
      let executableDir = executableURL.deletingLastPathComponent()
      let target = executableDir.appendingPathComponent("mlx.metallib")

      guard !FileManager.default.fileExists(atPath: target.path) else {
        return  // Already in place — nothing to do.
      }

      let arch = currentArch()
      let candidates = buildCandidates(from: executableDir, arch: arch)
      for candidate in candidates {
        if FileManager.default.fileExists(atPath: candidate.path) {
          do {
            try FileManager.default.copyItem(at: candidate, to: target)
            return
          } catch {
            // Copy failed (e.g. permissions) — try the next candidate.
            continue
          }
        }
      }
    // No metallib found — let MLX's own error fire on first use.
    #endif
  }

  // MARK: - Private helpers

  #if os(macOS)
    /// Walk up the directory tree from `start`, collecting metallib candidate
    /// paths in each `.build/<arch>-apple-macosx/{debug,release}/` directory.
    private static func buildCandidates(from start: URL, arch: String) -> [URL] {
      var candidates: [URL] = []
      var dir = start
      for _ in 0..<8 {
        let buildDir = dir.appendingPathComponent(".build/\(arch)-apple-macosx")
        for sub in ["debug", "release"] {
          // Test bundle path — the most reliable source after `swift test`.
          candidates.append(
            buildDir
              .appendingPathComponent(sub)
              .appendingPathComponent("TinyAudioPackageTests.xctest/Contents/MacOS/mlx.metallib")
          )
          // Direct build artifact path.
          candidates.append(
            buildDir.appendingPathComponent(sub).appendingPathComponent("mlx.metallib")
          )
        }
        let parent = dir.deletingLastPathComponent()
        if parent.path == dir.path || parent.path == "/" { break }
        dir = parent
      }
      return candidates
    }

    /// Returns "arm64" on Apple Silicon, "x86_64" on Intel.
    private static func currentArch() -> String {
      var info = utsname()
      uname(&info)
      let machine = withUnsafePointer(to: &info.machine) {
        $0.withMemoryRebound(to: CChar.self, capacity: Int(_SYS_NAMELEN)) {
          String(cString: $0)
        }
      }
      return machine == "arm64" ? "arm64" : "x86_64"
    }
  #endif
}
