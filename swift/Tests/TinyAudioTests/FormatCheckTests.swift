import Foundation
import Testing

@Suite("FormatCheck")
struct FormatCheckTests {
  /// Runs `swift-format lint` over the SDK Sources and Tests directories
  /// and fails if it reports any issues. This is the canonical "is the
  /// code formatted?" check. CI also runs the same command via the
  /// pre-commit hook.
  @Test func swiftFormatLintsClean() throws {
    // Find the repo root by walking up from the test bundle.
    let testBundle = Bundle.module.bundleURL
    var dir = testBundle
    while !FileManager.default.fileExists(atPath: dir.appendingPathComponent("Package.swift").path)
    {
      let parent = dir.deletingLastPathComponent()
      if parent.path == dir.path {
        Issue.record("Could not find Package.swift walking up from test bundle.")
        return
      }
      dir = parent
    }
    // dir is now the swift package root (the swift/ dir).
    let swiftRoot = dir

    // Locate swift-format via xcrun if not on PATH.
    let swiftFormatPath: String
    if let envPath = ProcessInfo.processInfo.environment["PATH"],
      let found = envPath.split(separator: ":").map(String.init).first(where: {
        FileManager.default.fileExists(atPath: "\($0)/swift-format")
      })
    {
      swiftFormatPath = "\(found)/swift-format"
    } else {
      // Fall back to xcrun to locate swift-format bundled with Xcode.
      let xcrun = Process()
      xcrun.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
      xcrun.arguments = ["--find", "swift-format"]
      let xcrunPipe = Pipe()
      xcrun.standardOutput = xcrunPipe
      xcrun.standardError = Pipe()
      try xcrun.run()
      xcrun.waitUntilExit()
      let found =
        String(data: xcrunPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
        .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
      swiftFormatPath = found.isEmpty ? "swift-format" : found
    }

    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = [
      swiftFormatPath, "lint",
      "--strict",
      "--recursive",
      swiftRoot.appendingPathComponent("Sources").path,
      swiftRoot.appendingPathComponent("Tests").path,
    ]
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = pipe
    try process.run()
    process.waitUntilExit()
    if process.terminationStatus != 0 {
      let data = pipe.fileHandleForReading.readDataToEndOfFile()
      let output = String(data: data, encoding: .utf8) ?? "(empty)"
      Issue.record("swift-format lint found issues:\n\(output)")
    }
  }
}
