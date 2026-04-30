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
            ],
            path: "Sources/TinyAudio"
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
