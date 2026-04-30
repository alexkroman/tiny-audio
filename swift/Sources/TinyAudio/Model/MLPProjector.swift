import Foundation
import MLX
import MLXNN

/// Frame-stack downsampler + 2-layer MLP. Mirrors `tiny_audio/mlx/projector.py`.
///
/// Input `x` has shape `[B, T, encoderDim]`. Each output frame absorbs `poolStride`
/// consecutive input frames concatenated along the feature dimension, then passes
/// through: `Linear → RMSNorm(eps=1e-6) → GELU → Linear`. No biases.
final class MLPProjector: Module, UnaryLayer {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "linear_2") var linear2: Linear

    let poolStride: Int

    init(encoderDim: Int, llmDim: Int, hiddenDim: Int, poolStride: Int = 4) {
        self.poolStride = poolStride
        self._linear1.wrappedValue = Linear(encoderDim * poolStride, hiddenDim, bias: false)
        self._norm.wrappedValue = RMSNorm(dimensions: hiddenDim, eps: 1e-6)
        self._linear2.wrappedValue = Linear(hiddenDim, llmDim, bias: false)
        super.init()
    }

    /// Output token count given an input frame count.
    ///
    /// Matches the Python formula: `(input_length - k) // k + 1`
    func outputLength(_ inputLength: Int) -> Int {
        return (inputLength - poolStride) / poolStride + 1
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, T, D]
        let batch = x.dim(0)
        let seq = x.dim(1)
        let dim = x.dim(2)
        let outLen = (seq - poolStride) / poolStride + 1
        let trimmed = x[0..., ..<(outLen * poolStride), 0...]
        var h = trimmed.reshaped([batch, outLen, dim * poolStride])
        h = linear1(h)
        h = norm(h)
        h = gelu(h)
        h = linear2(h)
        return h
    }
}
