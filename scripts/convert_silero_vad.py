"""Convert Silero VAD (TorchScript) to Core ML for the TinyAudio Swift SDK.

Run once. Produces swift/Sources/TinyAudio/Resources/silero_vad.mlpackage.

Usage:
    poetry run python scripts/convert_silero_vad.py

Architecture notes:
    Silero VAD (v5, 16 kHz) is a small neural VAD: STFT -> Conv encoder -> LSTM -> sigmoid.

    The upstream TorchScript model cannot be converted to Core ML directly because:
      1. The STFT submodule reads ``hop_length`` via ``prim::GetAttr``, which
         coremltools resolves as a string type (not int), causing a conv stride error.
      2. The decoder uses ``torch.len(state)`` for a null-state branch, which has
         no coremltools mapping.
      3. The LSTMCell uses ``unsafe_chunk`` internally, which is unsupported.

    Solution: pure-Python reimplementation with baked-in constants:
      * STFT: reproduce the forward transform with a constant ``stride=128`` instead
        of a module attribute lookup.
      * LSTM: expand the LSTM cell into explicit i/f/g/o gate computations using
        ``torch.nn.functional.linear``, which converts cleanly.
      * Decoder linear: reuse the TorchScript submodule directly (it has no
        unsupported ops after the LSTM is replaced).

    State representation:
      The original model stacks h and c into a single ``[2, 1, 128]`` tensor.
      We expose them as two separate inputs/outputs ``h`` and ``c`` (each
      ``[1, 128]``) to match Core ML's flat tensor model and simplify Swift.

    Inputs:
      - audio:  Float32 [1, 576]  -- context (64) + chunk (512) samples at 16 kHz.
                Zero-pad the context portion at the start of a new utterance.
      - h:      Float32 [1, 128]  -- LSTM hidden state (zeros to reset).
      - c:      Float32 [1, 128]  -- LSTM cell state (zeros to reset).

    Outputs:
      - prob:   Float16 [1, 1]    -- speech probability in [0, 1].
      - h_out:  Float16 [1, 128]  -- updated hidden state for the next frame.
      - c_out:  Float16 [1, 128]  -- updated cell state for the next frame.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import coremltools as ct
import torch

OUT_DIR = Path("swift/Sources/TinyAudio/Resources/silero_vad.mlpackage")

# Silero VAD 16 kHz constants.
CONTEXT_SAMPLES = 64  # context window prepended to each chunk
CHUNK_SAMPLES = 512  # 32 ms at 16 kHz
TOTAL_AUDIO = CONTEXT_SAMPLES + CHUNK_SAMPLES  # 576 total input samples
STFT_STRIDE = 128  # hop_length of the STFT filter bank
STFT_CUTOFF = 129  # filter_length // 2 + 1  (256 // 2 + 1)
HIDDEN_SIZE = 128  # LSTM hidden dimension


class SileroVADCoreML(torch.nn.Module):
    """Pure-Python Silero VAD, compatible with coremltools 9.0 / iOS 17.

    Reproduces the forward pass using only supported ops:
    torch.nn.functional.pad, conv1d, linear, sigmoid, tanh, sqrt.
    No torch.len, no prim::GetAttr on numeric attributes, no unsafe_chunk.
    """

    def __init__(
        self,
        forward_basis: torch.Tensor,
        encoder: torch.nn.Module,
        wi: torch.Tensor,
        wf: torch.Tensor,
        wg: torch.Tensor,
        wo: torch.Tensor,
        ui: torch.Tensor,
        uf: torch.Tensor,
        ug: torch.Tensor,
        uo: torch.Tensor,
        bxi: torch.Tensor,
        bxf: torch.Tensor,
        bxg: torch.Tensor,
        bxo: torch.Tensor,
        bhi: torch.Tensor,
        bhf: torch.Tensor,
        bhg: torch.Tensor,
        bho: torch.Tensor,
        conv_w: torch.Tensor,
        conv_b: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("forward_basis", forward_basis)
        self.encoder = encoder
        self.register_buffer("wi", wi)
        self.register_buffer("wf", wf)
        self.register_buffer("wg", wg)
        self.register_buffer("wo", wo)
        self.register_buffer("ui", ui)
        self.register_buffer("uf", uf)
        self.register_buffer("ug", ug)
        self.register_buffer("uo", uo)
        self.register_buffer("bxi", bxi)
        self.register_buffer("bxf", bxf)
        self.register_buffer("bxg", bxg)
        self.register_buffer("bxo", bxo)
        self.register_buffer("bhi", bhi)
        self.register_buffer("bhf", bhf)
        self.register_buffer("bhg", bhg)
        self.register_buffer("bho", bho)
        self.register_buffer("conv_w", conv_w)
        self.register_buffer("conv_b", conv_b)

    def forward(
        self, audio: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one VAD frame.

        Args:
            audio: Float32 [1, 576] -- context (64) prepended to current chunk (512).
            h:     Float32 [1, 128] -- LSTM hidden state; pass zeros to reset.
            c:     Float32 [1, 128] -- LSTM cell state; pass zeros to reset.

        Returns:
            prob:  Float [1, 1]   -- speech probability in [0, 1].
            h_out: Float [1, 128] -- updated hidden state for the next frame.
            c_out: Float [1, 128] -- updated cell state for the next frame.
        """
        # STFT: reflect-pad then apply learned filter bank with constant stride.
        x = torch.nn.functional.pad(audio, (0, 64), mode="reflect")
        x = torch.unsqueeze(x, 1)
        x = torch.nn.functional.conv1d(x, self.forward_basis, stride=STFT_STRIDE, padding=0)
        real = x[:, :STFT_CUTOFF, :].to(torch.float32)
        imag = x[:, STFT_CUTOFF:, :].to(torch.float32)
        x = torch.sqrt(real**2 + imag**2)  # magnitude spectrogram

        # Conv encoder (TorchScript sequential, no unsupported ops in this path).
        x = self.encoder(x)  # [1, 128, 1]
        x = torch.squeeze(x, -1)  # [1, 128]

        # LSTM cell via explicit gates (avoids unsafe_chunk).
        lin = torch.nn.functional.linear
        i_gate = torch.sigmoid(lin(x, self.wi, self.bxi) + lin(h, self.ui, self.bhi))
        f_gate = torch.sigmoid(lin(x, self.wf, self.bxf) + lin(h, self.uf, self.bhf))
        g_gate = torch.tanh(lin(x, self.wg, self.bxg) + lin(h, self.ug, self.bhg))
        o_gate = torch.sigmoid(lin(x, self.wo, self.bxo) + lin(h, self.uo, self.bho))
        c_out = f_gate * c + i_gate * g_gate
        h_out = o_gate * torch.tanh(c_out)

        # Output projection: ReLU -> 1x1 conv -> sigmoid.
        # The original decoder is Sequential[Dropout, ReLU, Conv1d(128->1), Sigmoid].
        # Dropout is a no-op at inference; we include ReLU explicitly here.
        y = torch.unsqueeze(h_out, -1).to(torch.float32)  # [1, 128, 1]
        y = torch.nn.functional.relu(y)
        y = torch.nn.functional.conv1d(y, self.conv_w, self.conv_b)
        y = torch.sigmoid(y)

        # Reproduce outer model mean pooling over the single frame.
        prob = torch.unsqueeze(torch.mean(torch.squeeze(y, 1), [1]), 1)  # [1, 1]
        return prob, h_out, c_out


def _split_lstm_weights(
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
) -> tuple:
    """Unpack fused LSTM weight matrix into per-gate tensors (i, f, g, o)."""
    hs = HIDDEN_SIZE
    wi = weight_ih[:hs]
    wf = weight_ih[hs : 2 * hs]
    wg = weight_ih[2 * hs : 3 * hs]
    wo = weight_ih[3 * hs :]
    ui = weight_hh[:hs]
    uf = weight_hh[hs : 2 * hs]
    ug = weight_hh[2 * hs : 3 * hs]
    uo = weight_hh[3 * hs :]
    bxi = bias_ih[:hs]
    bxf = bias_ih[hs : 2 * hs]
    bxg = bias_ih[2 * hs : 3 * hs]
    bxo = bias_ih[3 * hs :]
    bhi = bias_hh[:hs]
    bhf = bias_hh[hs : 2 * hs]
    bhg = bias_hh[2 * hs : 3 * hs]
    bho = bias_hh[3 * hs :]
    return (
        wi,
        wf,
        wg,
        wo,
        ui,
        uf,
        ug,
        uo,
        bxi,
        bxf,
        bxg,
        bxo,
        bhi,
        bhf,
        bhg,
        bho,
    )


def main() -> None:
    print("Loading Silero VAD from torch.hub...")
    outer, _ = torch.hub.load(
        "snakers4/silero-vad",
        "silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    outer.train(False)
    inner = outer._model
    inner.train(False)

    # Validate that STFT constants match the loaded model.
    actual_hop = int(inner.stft.hop_length)
    actual_filter = int(inner.stft.filter_length)
    expected_cutoff = actual_filter // 2 + 1
    if actual_hop != STFT_STRIDE or expected_cutoff != STFT_CUTOFF:
        raise ValueError(
            f"STFT mismatch: hop={actual_hop} (expected {STFT_STRIDE}), "
            f"cutoff={expected_cutoff} (expected {STFT_CUTOFF}). "
            "Update constants in this script."
        )

    # Extract weights from the loaded TorchScript model.
    forward_basis = inner.stft.forward_basis_buffer.detach().clone()
    encoder = inner.encoder
    rnn = inner.decoder.rnn
    conv_layer = list(inner.decoder.decoder.children())[2]
    conv_w = conv_layer.weight.detach().clone()
    conv_b = conv_layer.bias.detach().clone()
    lstm_gates = _split_lstm_weights(
        rnn.weight_ih.detach().clone(),
        rnn.weight_hh.detach().clone(),
        rnn.bias_ih.detach().clone(),
        rnn.bias_hh.detach().clone(),
    )

    model = SileroVADCoreML(forward_basis, encoder, *lstm_gates, conv_w, conv_b)
    model.eval()

    # Verify numerical agreement with the original model.
    audio = torch.zeros(1, TOTAL_AUDIO, dtype=torch.float32)
    h0 = torch.zeros(1, HIDDEN_SIZE, dtype=torch.float32)
    c0 = torch.zeros(1, HIDDEN_SIZE, dtype=torch.float32)
    state0 = torch.zeros(2, 1, HIDDEN_SIZE, dtype=torch.float32)

    with torch.no_grad():
        my_prob, _, _ = model(audio, h0, c0)
        orig_prob, _ = inner(audio, state0)

    max_diff = (my_prob - orig_prob).abs().max().item()
    print(f"Silence prob (reimplementation): {my_prob.item():.6f}")
    print(f"Silence prob (original):         {orig_prob.item():.6f}")
    print(f"Max absolute difference:         {max_diff:.2e}")
    if max_diff > 1e-3:
        raise RuntimeError(f"Reimplementation diverges by {max_diff:.4f}. Check weight extraction.")

    print(f"\nTracing model (audio={list(audio.shape)}, h={list(h0.shape)}, c={list(c0.shape)})...")
    traced = torch.jit.trace(model, (audio, h0, c0))

    with torch.no_grad():
        tp, _, _ = traced(audio, h0, c0)
    print(f"Traced silence prob: {tp.item():.6f}")

    print("Converting to Core ML (FP16, iOS 17+)...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="audio", shape=(1, TOTAL_AUDIO)),
            ct.TensorType(name="h", shape=(1, HIDDEN_SIZE)),
            ct.TensorType(name="c", shape=(1, HIDDEN_SIZE)),
        ],
        outputs=[
            ct.TensorType(name="prob"),
            ct.TensorType(name="h_out"),
            ct.TensorType(name="c_out"),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {OUT_DIR}...")
    mlmodel.save(str(OUT_DIR))

    total_bytes = sum(f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file())
    print(f"\nDone. Model size: {total_bytes / 1024 / 1024:.2f} MB")
    print(f"Output: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
