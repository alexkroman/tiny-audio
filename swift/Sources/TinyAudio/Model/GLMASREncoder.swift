import Foundation
import MLX
import MLXNN
import MLXAudioSTT

/// Minimal config for the GLM-ASR audio encoder. Built from the bundle's config.json.
struct GLMASREncoderConfig: Codable {
    let nMels: Int
    let encoderDim: Int
    let numLayers: Int
    let numHeads: Int
    let headDim: Int
    let intermediateSize: Int
    let ropeTheta: Float

    enum CodingKeys: String, CodingKey {
        case nMels = "n_mels"
        case encoderDim = "encoder_dim"
        case numLayers = "num_layers"
        case numHeads = "num_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case ropeTheta = "rope_theta"
    }
}

/// GLM-ASR audio encoder: mlx-audio-swift's `WhisperEncoder` + a final `LayerNorm`.
///
/// The wrapper exists so the parameter keys match what `encoder.safetensors` ships:
///   `encoder.<inner whisper keys>` (conv1, conv2, embed_positions, layers.N.*)
///   `norm.weight`, `norm.bias`
///
/// Input:  `[B, nMels, T_mel]` — NCL convention (matches `LogMelSpectrogram.compute()`).
/// Output: `[B, T_enc, encoderDim]` — NLC, where `T_enc ≈ T_mel / 2` (conv2 stride=2).
final class GLMASREncoder: Module, UnaryLayer {
    @ModuleInfo(key: "encoder") var encoder: WhisperEncoder
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(_ config: GLMASREncoderConfig) {
        // Build WhisperConfig matching Python GLMASREncoder: RoPE on, Llama
        // half-split convention (ropeTraditional=false).
        let whisperConfig = WhisperConfig(
            dModel: config.encoderDim,
            encoderAttentionHeads: config.numHeads,
            encoderFfnDim: config.intermediateSize,
            encoderLayers: config.numLayers,
            numMelBins: config.nMels,
            ropeTraditional: false  // Llama half-split partial RoPE
        )
        self._encoder.wrappedValue = WhisperEncoder(config: whisperConfig, useRope: true)
        self._norm.wrappedValue = LayerNorm(dimensions: config.encoderDim)
        super.init()
    }

    /// Input: `[B, nMels, T_mel]` (NCL from LogMelSpectrogram).
    /// Output: `[B, T_enc, encoderDim]` (NLC, ready for the projector).
    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        // MLX Conv1d expects NLC [B, T, C]. Permute from NCL [B, C, T].
        let x = mel.transposed(0, 2, 1)  // [B, T_mel, nMels]
        return norm(encoder(x))
    }
}
