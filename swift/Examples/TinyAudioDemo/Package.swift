// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "TinyAudioDemo",
    platforms: [
        .macOS(.v14),
    ],
    dependencies: [
        // Path-based dep on the parent TinyAudio package — no need to publish.
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "TinyAudioDemo",
            dependencies: [
                .product(name: "TinyAudio", package: "swift"),
            ],
            path: "Sources/TinyAudioDemo"
        ),
    ]
)
