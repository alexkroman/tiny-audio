"""Audio projector modules for bridging encoder and decoder embeddings.

This module contains all projector architectures:
- MLPAudioProjector: Simple 2-layer MLP with frame stacking downsampling
"""

import math

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm

# =============================================================================
# MLP Projector
# =============================================================================


class MLPAudioProjector(nn.Module):
    """2-layer MLP projector with frame-stacking downsampling.

    Normalize the stacked encoder features, project, and let ``linear_2`` set
    the output scale. This is the layout the audio projectors in transformers
    5.x use -- GLM-ASR, Qwen3-ASR, Voxtral, AudioFlamingo3, Qwen2-Audio and
    Granite Speech all end on a bare ``nn.Linear``, and Gemma3's
    ``mm_soft_emb_norm`` / LFM2-VL's optional projector LayerNorm normalize the
    encoder features going in.

    Ending on a bare linear is the majority convention, not a universal one: a
    survey of ~72 projector classes in the 5.x tree found five families that DO
    end in a trailing normalization -- Gemma3n's ``embedding_post_projection_norm``
    (weightless), plus hunyuan_vl, ernie4_5_vl_moe, inkling and florence2, the
    last four with a learnable gain. All of them train the decoder jointly, so
    the degeneracy described below is evidently tolerated at scale. Gemma3n is
    the one design in the tree that deliberately pins the injected magnitude at
    ~1x its text-embedding scale, and it pays that degeneracy to do it.

    TRIED AND REJECTED, with numbers, so nobody spends another launch on it.
    Adding Gemma3n's trailing weightless RMSNorm here (``Qwen3_5RMSNorm`` with
    ``with_scale`` off, so no parameter and no state_dict key) does pin the
    scale perfectly -- ``output_rms_over_embed`` logs 1.0000000 for the whole
    run instead of climbing to 28x. It also destroys training: loss at step
    300 was **3.166 against 0.319** for the identical recipe without it, and
    the projector's noise->signal cliff never fired at all.
    Because a trailing norm does not pin the AGGREGATE scale, which is what
    drifts -- it pins EVERY TOKEN to the same magnitude. A trained projector's
    output carries per-token RMS with CV 0.373, p95/p5 = 3.67x and
    max/min = 9.4x, against CV 0.102 / 1.40x for the decoder's own
    ``embed_tokens`` rows. That spread is signal: silence against speech,
    confident frames against ambiguous ones. Normalising per token deletes it
    and leaves only direction.
    ``output_scale`` -- one scalar -- is the right SHAPE of intervention for
    this reason: it fixes the aggregate and leaves relative magnitudes alone.
    Its only weakness is being applied once at init. If the drift ever needs a
    brake, lower the projector LR (equilibrium ||W|| scales with lr) rather
    than clamping the output; and note the drift has no measured cost --
    dL/d(log c) came back +0.0005 +/- 0.0273, |t| = 0.05, and the completed
    granite_qwen run reached ~48x while producing the best WER on record here.

    A trailing ``norm_2`` RMSNorm used to follow ``linear_2``, with the other
    RMSNorm between ``linear_1`` and the activation. Both made the linear
    feeding them scale-invariant -- RMSNorm erases whatever magnitude a linear
    produces -- so the loss could not see their scale and it inflated without
    bound while the trailing gain shrank to compensate. Measured on
    granite_qwen (Granite-speech encoder -> Qwen3.5-2B) across its own
    checkpoints at steps 4k/6k/12k:

        ||linear_1||   92.6 -> 114.1 -> 160.7
        ||linear_2||   86.2 -> 111.5 -> 169.6
        norm_2 gain    0.85 ->  0.80 ->  0.69   (output RMS 57x -> 48x embed_tokens)

    Two costs. The optimizer spends capacity on a scale fight it can never
    settle, and for a scale-invariant weight under Adam the angular step goes
    as lr/||W||, so 1.85x norm growth had quietly decayed the projector's
    effective learning rate 1.85x on top of the cosine schedule.

    Biases are on, as in GLM-ASR / Qwen3-ASR / Qwen2-Audio / AudioFlamingo3 and
    in ``SimpleAdapter`` below: with no trailing norm to absorb an offset, the
    projector needs some way to place its output, and 6K bias params is the
    cheap way to give it one.

    No dropout. Every audio projector surveyed in transformers 5.x -- Voxtral,
    Qwen2-Audio, Qwen3-ASR, GLM-ASR, AudioFlamingo3, Granite Speech, Gemma3n,
    Gemma4, Phi-4-MM, FunASR-Nano, VibeVoice-ASR -- has none, and neither does
    any vision projector in the tree. The reason to care is that the dose was
    much larger than it looked: elementwise dropout at rate ``p`` injects
    ``sqrt(p/(1-p))`` relative RMS noise on the output, so the former 0.05 put
    **22.9%** relative noise on every one of the ~237 soft audio tokens, and
    the 0.1 before it 33%. NEFTune-style embedding noise is typically 5-15%,
    so this was well past that band, on the one module with no alternative
    path from encoder to decoder and a frozen encoder that cannot adapt to it.
    """

    def __init__(self, config):
        """Initialize MLP projector.

        Args:
            config: ASRConfig with encoder_dim, llm_dim, projector_pool_stride
        """
        super().__init__()

        encoder_dim = getattr(config, "encoder_dim", 768)
        llm_dim = getattr(config, "llm_dim", 2048)
        self.k = getattr(config, "projector_pool_stride", 4)

        # Frame stacking: concat k adjacent frames then project
        in_dim = encoder_dim * self.k
        # Hidden dim defaults to llm_dim, can be overridden via config
        hidden_dim = getattr(config, "projector_hidden_dim", None) or llm_dim
        self.input_norm = LlamaRMSNorm(in_dim, eps=1e-6)
        self.linear_1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(hidden_dim, llm_dim)

        # Fixed, non-learnable. Persistent so it round-trips with the
        # checkpoint -- recomputing it at load time would calibrate against the
        # freshly-initialized weights rather than the trained ones.
        self.register_buffer("output_scale", torch.ones(()), persistent=True)
        self._calibrate_output_scale(getattr(config, "projector_output_rms", None))

    def _calibrate_output_scale(self, target_rms) -> None:
        """Set ``output_scale`` so the projector emits at the decoder's embedding scale.

        Dropping the trailing norm let ``linear_2`` own the output scale, but
        nothing then *sets* it: PyTorch's default init puts the output at RMS
        ~0.198 against Qwen3.5's ``embed_tokens`` RMS of 0.0150, i.e. 13x hot.
        The decoder is pre-norm, so the loss barely sees this -- but the
        residual stream is unnormalized, so a 13x-hot prefix makes every
        sublayer's write at those ~237 audio positions 13x smaller in relative
        terms, and the audio passes through the stack close to untouched.

        The scale lives in a fixed buffer rather than being folded into
        ``linear_2``'s weights, and that distinction is load-bearing. Folding
        it in shrinks ``linear_2`` by the same 13x, which leaves its relative
        step ``lr/|w|`` 13x LARGER -- at lr 1e-3 against a post-scaling std of
        ~6.8e-4 that is an O(1) update per step, and the weights simply grow
        until the relative step becomes reasonable again. That is exactly the
        mechanism behind the drift this repo already measured, where a 0.029
        init climbed back to ~1.0. A constant multiplier sets the output scale
        while leaving ``linear_2`` at default-init magnitude, so its relative
        step stays sane.

        It is also not a normalizer: the buffer is fixed, so ``linear_2``'s own
        magnitude stays visible to the loss and the scale-invariance
        degeneracy that removing ``norm_2`` fixed does not come back.

        Calibrated by probe rather than closed form because the init -> output
        RMS mapping runs through GELU and depends on ``hidden_dim`` and
        ``pool_stride``. The output is exactly linear in the scale, so one
        probe is exact for any geometry.
        """
        # "auto" is resolved to a float by ASRModel._create_projector, the only
        # place that can see the decoder's embedding table. A projector built
        # standalone (deploy planning, unit tests) gets the unresolved sentinel
        # and keeps scale 1.0.
        if not isinstance(target_rms, (int, float)) or isinstance(target_rms, bool):
            return
        if target_rms <= 0:
            return

        measured = self.measure_output_rms()

        if not math.isfinite(measured) or measured <= 0.0:
            return
        self.output_scale.fill_(float(target_rms) / measured)

    def measure_output_rms(self) -> float:
        """RMS of this projector's output on a standard-normal probe.

        Used both to calibrate ``output_scale`` at init and by the trainer to
        log whether that calibration is holding as the weights move. Both need
        the same probe geometry, so it is defined once here rather than
        reconstructed from ``linear_1``/``linear_2`` internals by the caller.
        """
        with torch.no_grad():
            w = self.linear_2.weight
            enc_dim = self.linear_1.in_features // self.k
            probe = torch.randn(1, 64 * self.k, enc_dim, dtype=w.dtype, device=w.device)
            return self.forward(probe).float().pow(2).mean().sqrt().item()

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input length (matches GLM-ASR)."""
        return _frame_stack_length(input_length, self.k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project audio features to LLM embedding space.

        Args:
            x: Audio encoder output of shape [batch, seq_len, encoder_dim]

        Returns:
            Projected features of shape [batch, (seq_len - k) // k + 1, llm_dim]
        """
        x = _frame_stack(x, self.k)
        x = self.input_norm(x)
        x = self.linear_1(x)
        x = self.act(x)
        return self.linear_2(x) * self.output_scale


# =============================================================================
# Frame stacking helpers
# =============================================================================


def _frame_stack_length(length, k: int):
    """Frames left after stacking k adjacent frames (GLM-ASR's rule).

    Trailing frames that don't fill a complete k-frame window are dropped.
    Works for Python ints and torch tensors alike.
    """
    return (length - k) // k + 1


def _frame_stack(x: torch.Tensor, k: int) -> torch.Tensor:
    """Stack k adjacent frames along the feature dim."""
    batch, seq, dim = x.shape
    out_len = _frame_stack_length(seq, k)
    return x[:, : out_len * k, :].reshape(batch, out_len, dim * k)


# =============================================================================
# Projector Registry
# =============================================================================

PROJECTOR_CLASSES = {
    "mlp": MLPAudioProjector,
}
