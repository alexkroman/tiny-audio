// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "TinyAudio",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "TinyAudio", targets: ["TinyAudio"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0"),
        .package(url: "https://github.com/Blaizzy/mlx-audio-swift", from: "0.1.2"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
    ],
    targets: [
        .target(
            name: "TinyAudio",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXAudioCore", package: "mlx-audio-swift"),
                .product(name: "MLXAudioSTT", package: "mlx-audio-swift"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/TinyAudio",
            exclude: [
                "Vendored/Qwen3/LICENSE",
                "Vendored/Qwen3/UPSTREAM.md",
            ]
        ),
        .testTarget(
            name: "TinyAudioTests",
            dependencies: ["TinyAudio"],
            path: "Tests/TinyAudioTests",
            resources: [
                .copy("Fixtures"),
            ]
        ),
    ]
)
