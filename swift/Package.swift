// swift-tools-version:5.9
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
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.20"),
    ],
    targets: [
        .target(
            name: "TinyAudio",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
                // NOTE: swift-transformers does not expose a standalone "Hub" product;
                // Hub functionality is bundled inside "Transformers". Removed "Hub" here.
                // TODO(Task 8+): Re-evaluate if a future swift-transformers version adds a Hub product.
            ],
            path: "Sources/TinyAudio"
            // TODO(Task 8): Re-enable once MelFilterbank.json is generated and committed.
            // resources: [
            //     .process("Mel/MelFilterbank.json"),
            // ]
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
