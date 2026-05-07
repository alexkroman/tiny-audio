// swift-tools-version:6.0
import PackageDescription

let package = Package(
  name: "CookbookDemo",
  platforms: [
    // macOS only on purpose (matches TinyAudioDemo). Cooking demo is voice-first
    // and big-screen by design.
    .macOS(.v14)
  ],
  dependencies: [
    .package(path: "../..")
  ],
  targets: [
    .executableTarget(
      name: "CookbookDemo",
      dependencies: [
        .product(name: "TinyAudio", package: "swift")
      ],
      path: "Sources/CookbookDemo",
      resources: [
        .copy("../../Resources/recipes.json")
      ]
    ),
    .testTarget(
      name: "CookbookDemoTests",
      dependencies: ["CookbookDemo"],
      path: "Tests/CookbookDemoTests"
    ),
  ]
)
