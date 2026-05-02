// swift-tools-version:6.0
import PackageDescription

let package = Package(
  name: "TinyAudioDemo",
  platforms: [
    // macOS only on purpose. Building this SwiftPM executableTarget for iOS
    // produces a bare binary with no Info.plist, which the iOS Simulator
    // launches without a bundle identifier — see BKSHIDEvent log noise and
    // missing-bundleID failures. The iOS demo MUST be built via the
    // TinyAudioDemo.xcodeproj's `TinyAudioDemo_iOS` scheme inside
    // swift/TinyAudio.xcworkspace.
    .macOS(.v14),
  ],
  dependencies: [
    // Path-based dep on the parent TinyAudio package — no need to publish.
    .package(path: "../..")
  ],
  targets: [
    .executableTarget(
      name: "TinyAudioDemo",
      dependencies: [
        .product(name: "TinyAudio", package: "swift")
      ],
      path: "Sources/TinyAudioDemo"
    )
  ]
)
