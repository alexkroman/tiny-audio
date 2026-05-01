import CryptoKit
import Foundation

struct ManifestVerifier {
  struct Manifest: Decodable {
    let formatVersion: Int
    let files: [String: FileEntry]

    private enum CodingKeys: String, CodingKey {
      case formatVersion = "format_version"
      case files
    }
  }

  struct FileEntry: Decodable {
    let sha256: String
    let size: Int
  }

  /// Validate a directory against its `manifest.json`.
  static func verify(directory: URL) throws {
    let manifestURL = directory.appendingPathComponent("manifest.json")
    let data = try Data(contentsOf: manifestURL)
    let manifest = try JSONDecoder().decode(Manifest.self, from: data)

    guard WeightCache.supportedFormatVersions.contains(manifest.formatVersion) else {
      throw TinyAudioError.formatVersionUnsupported(
        found: manifest.formatVersion,
        supported: WeightCache.supportedFormatVersions
      )
    }

    for (name, entry) in manifest.files {
      let fileURL = directory.appendingPathComponent(name)
      let actualSize =
        (try? FileManager.default.attributesOfItem(atPath: fileURL.path)[.size] as? Int) ?? -1
      guard actualSize == entry.size else {
        throw TinyAudioError.manifestMismatch(
          file: name, expected: "size=\(entry.size)", actual: "size=\(actualSize)")
      }
      let actual = try sha256(of: fileURL)
      guard actual == entry.sha256 else {
        throw TinyAudioError.manifestMismatch(file: name, expected: entry.sha256, actual: actual)
      }
    }
  }

  private static func sha256(of url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var hasher = SHA256()
    while true {
      let chunk = try handle.read(upToCount: 1 << 20) ?? Data()
      if chunk.isEmpty { break }
      hasher.update(data: chunk)
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
  }
}
