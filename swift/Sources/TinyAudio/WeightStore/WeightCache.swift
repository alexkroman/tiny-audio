import Foundation

/// Locates and validates a downloaded tiny-audio-mlx bundle directory.
///
/// We rely on swift-transformers' Hub to actually download files into its
/// standard cache layout (`~/Library/Caches/huggingface/hub/...` or
/// `~/.cache/huggingface/hub/...`). This type only resolves paths and
/// tracks our completion marker.
struct WeightCache {
  static let supportedFormatVersions: ClosedRange<Int> = 1...1
  static let completionMarker = ".mlx_complete"

  let directory: URL

  init(directory: URL) {
    self.directory = directory
  }

  var isComplete: Bool {
    return FileManager.default.fileExists(
      atPath: directory.appendingPathComponent(Self.completionMarker).path)
  }

  func markComplete() throws {
    let marker = directory.appendingPathComponent(Self.completionMarker)
    let tmp = directory.appendingPathComponent(Self.completionMarker + ".tmp")
    try Data().write(to: tmp)
    if FileManager.default.fileExists(atPath: marker.path) {
      try FileManager.default.removeItem(at: marker)
    }
    try FileManager.default.moveItem(at: tmp, to: marker)
  }

  func wipe() throws {
    if FileManager.default.fileExists(atPath: directory.path) {
      try FileManager.default.removeItem(at: directory)
    }
  }
}
