"""Audio projector modules for bridging encoder and decoder embeddings.

This module contains all projector architectures:
- MLPAudioProjector: Simple 2-layer MLP with frame stacking downsampling
- MOSAProjector: MOSA-style dense mixture of experts
- SharedMoEAudioProjector: Shared expert + sparse routed experts
- QFormerAudioProjector: BLIP-2 QFormer with learnable queries (Granite-style)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from transformers import AutoModel, Blip2QFormerConfig
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
    encoder features going in. It also matches ``MoEAudioProjector`` below,
    which has always normalized its stacked input.

    Ending on a bare linear is the majority convention, not a universal one: a
    survey of ~72 projector classes in the 5.x tree found five families that DO
    end in a trailing normalization -- Gemma3n's ``embedding_post_projection_norm``
    (weightless), plus hunyuan_vl, ernie4_5_vl_moe, inkling and florence2, the
    last four with a learnable gain. All of them train the decoder jointly, so
    the degeneracy described below is evidently tolerated at scale. Gemma3n is
    the one design in the tree that deliberately pins the injected magnitude at
    ~1x its text-embedding scale, and it pays that degeneracy to do it.

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
        # Set by the backward hook in `_maybe_probe_scale_gradient`, drained by
        # the trainer. Not a parameter and not a buffer, so it stays out of the
        # optimizer and out of the checkpoint.
        self.scale_grad: torch.Tensor | None = None
        self.scale_grad_count: int = 0
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

        with torch.no_grad():
            enc_dim = self.linear_1.in_features // self.k
            probe = torch.randn(
                1,
                64 * self.k,
                enc_dim,
                dtype=self.linear_2.weight.dtype,
                device=self.linear_2.weight.device,
            )
            measured = self.forward(probe).pow(2).mean().sqrt().item()

        if not math.isfinite(measured) or measured <= 0.0:
            return
        self.output_scale.fill_(float(target_rms) / measured)

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
        out = self.linear_2(x) * self.output_scale
        self._maybe_probe_scale_gradient(out)
        return out

    def _maybe_probe_scale_gradient(self, out: torch.Tensor) -> None:
        """Accumulate dL/d(log c), where c is a hypothetical output-scale multiplier.

        This answers the one question the output-scale story actually turns on:
        does the LOSS want a different injection magnitude, or is the observed
        drift just Adam's step size interacting with weight norm?

        Measured on a live run, the ratio of projector output RMS to the
        decoder's embedding RMS starts at 1.0 by construction and climbs to
        ~47x by step 12k, then plateaus. Two readings, with opposite
        implications:
          - Adam artifact. The per-parameter step is ~lr regardless of
            gradient, so weights grow until lr/|w| lands in a usable band, and
            the output scale is dragged along. Then the scale is cosmetic and
            pinning it is free.
          - The loss wants it. A large-magnitude write reserves residual
            bandwidth and survives the stack for later positions to attend to,
            and attention reads are scale-invariant (q/k/v all derive from an
            RMSNorm'd hidden state). Then pinning the scale costs quality.

        `dL/dc` at `c = 1` is `sum(dL/dout * out)` by the chain rule, and
        because `d/d(log c) = c * d/dc`, at `c = 1` the two coincide. Computing
        it from a backward hook on `out` avoids materializing a second copy of
        a `[B, T, llm_dim]` tensor, which an explicit probe multiplier would.

        Sign is what matters: persistently negative means the loss is pushing
        the scale up and the drift is intentional; hovering around zero means
        the loss is indifferent and the drift is optimizer noise.
        """
        if not self.training or not out.requires_grad:
            return
        detached = out.detach()

        def _hook(grad: torch.Tensor) -> None:
            contribution = (grad.detach() * detached).sum()
            prev = self.scale_grad
            self.scale_grad = contribution if prev is None else prev + contribution
            self.scale_grad_count += 1

        out.register_hook(_hook)


# =============================================================================
# MoE Projector (MOSA-style)
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


class SimpleAdapter(nn.Module):
    """Simple 2-layer GELU adapter (from MOSA paper)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MOSAProjector(nn.Module):
    """MOSA-Base projector: simple 2-layer ReLU router with 4 simple adapters.

    Based on "MOSA: Mixtures of Simple Adapters" (arXiv:2508.18998).
    Uses softmax gating over all experts (dense MoE) with only cross-entropy loss.
    Uses Conv1d for downsampling (2 layers, stride 2 each = 4x total).
    """

    ADAPTER_HIDDEN_DIM = 4096
    ROUTER_HIDDEN_DIM = 512
    CONV_KERNEL = 3
    CONV_STRIDE = 2
    CONV_PADDING = 1

    def __init__(self, config):
        """Initialize MOSA projector.

        Args:
            config: ASRConfig with encoder_dim, llm_dim, num_experts
        """
        super().__init__()
        self.encoder_dim = getattr(config, "encoder_dim", None) or 1280
        self.llm_dim = getattr(config, "llm_dim", None) or 2048
        self.num_experts = getattr(config, "num_experts", None) or 4  # MOSA-Base uses 4

        conv_kwargs = {
            "kernel_size": self.CONV_KERNEL,
            "stride": self.CONV_STRIDE,
            "padding": self.CONV_PADDING,
        }
        self.downsampler = nn.Sequential(
            nn.Conv1d(self.encoder_dim, self.encoder_dim, **conv_kwargs),
            nn.GELU(),
            nn.Conv1d(self.encoder_dim, self.llm_dim, **conv_kwargs),
            nn.GELU(),
        )

        self.router = nn.Sequential(
            nn.Linear(self.llm_dim, self.ROUTER_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(self.ROUTER_HIDDEN_DIM, self.num_experts),
        )

        self.experts = nn.ModuleList(
            [
                SimpleAdapter(self.llm_dim, self.ADAPTER_HIDDEN_DIM, self.llm_dim)
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project audio features using mixture of experts.

        Args:
            x: Audio encoder output of shape [batch, seq_len, encoder_dim]

        Returns:
            Projected features of shape [batch, out_len, llm_dim]
        """
        x = self.downsampler(x.transpose(1, 2)).transpose(1, 2)

        routing_weights = F.softmax(self.router(x), dim=-1)  # (B, out_len, num_experts)

        # Accumulate weighted expert outputs without materializing all experts at once.
        output = self.experts[0](x) * routing_weights[..., 0:1]
        for i, expert in enumerate(self.experts[1:], start=1):
            output = output + expert(x) * routing_weights[..., i : i + 1]
        return output

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length after Conv1d downsampling (4x reduction)."""
        length = input_length
        for _ in range(2):
            length = (length + 2 * self.CONV_PADDING - self.CONV_KERNEL) // self.CONV_STRIDE + 1
        return length


# =============================================================================
# MoE Projector (Pure PyTorch with Shared Expert)
# =============================================================================


class MoEAudioProjector(nn.Module):
    """MoE projector with shared expert (DeepSeek-style), pure PyTorch implementation.

    Uses 4 sparse experts with top-2 routing plus a shared expert that processes all tokens.
    No external dependencies (megablocks removed).

    Architecture matches main branch: norm → experts(in_dim → hidden → out_dim)
    """

    def __init__(self, config):
        """Initialize MoE projector.

        Args:
            config: ASRConfig with encoder_dim, llm_dim, num_experts, num_experts_per_tok
        """
        super().__init__()

        self.k = getattr(config, "projector_pool_stride", 4)
        self.aux_coef = getattr(config, "router_aux_loss_coef", 0.01)

        # Stability coefficients
        self.router_z_loss_coef = getattr(
            config, "router_z_loss_coef", 1e-4
        )  # Prevents logit explosion
        self.router_jitter_noise = getattr(
            config, "router_jitter_noise", 0.01
        )  # Prevents expert collapse

        in_dim = config.encoder_dim * self.k
        out_dim = config.llm_dim

        # Expert hidden dim (default = output dim)
        hidden_dim = getattr(config, "projector_hidden_dim", None) or out_dim

        # Number of experts and top-k selection
        self.num_experts = getattr(config, "num_experts", 4)
        self.top_k = getattr(config, "num_experts_per_tok", 2)

        # A. Normalize stacked input (like main branch SharedMoEBlock)
        self.norm = LlamaRMSNorm(in_dim, eps=1e-6)

        # B. Router (operates on stacked input)
        self.router = nn.Linear(in_dim, self.num_experts, bias=False)

        # C. Experts: simple 2-layer MLP (same as MLPAudioProjector)
        self.experts = nn.ModuleList(
            [SimpleAdapter(in_dim, hidden_dim, out_dim) for _ in range(self.num_experts)]
        )

        # D. Shared Expert (same architecture)
        self.shared_expert = SimpleAdapter(in_dim, hidden_dim, out_dim)

        # E. Initialize weights for stable training
        self._init_weights()

        self.last_aux_loss = torch.tensor(0.0)

    def _init_weights(self):
        """Initialize weights for stable training start."""
        with torch.no_grad():
            # Router: small weights -> uniform probability
            nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

            # Experts: xavier for fc1, small for fc2 (output)
            for expert in [self.shared_expert, *self.experts]:
                nn.init.xavier_uniform_(expert.fc1.weight)
                nn.init.normal_(expert.fc2.weight, mean=0.0, std=0.01)  # Small init

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input length (matches MLP projector)."""
        return _frame_stack_length(input_length, self.k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project audio features using shared + sparse MoE.

        Args:
            x: Audio encoder output of shape [batch, seq_len, encoder_dim]

        Returns:
            Projected features of shape [batch, out_len, llm_dim]
        """
        x = _frame_stack(x, self.k)
        batch, out_len, _ = x.shape

        # Normalize stacked input (like main branch SharedMoEBlock)
        x = self.norm(x)
        flat_x = x.view(-1, x.size(-1))  # [tokens, in_dim]

        # 3. Shared Expert (compute first, creates output tensor)
        output = self.shared_expert(flat_x)

        # 4. Sparse Experts (in-place add to shared output)
        self.last_aux_loss = self._forward_sparse(flat_x, output)

        return output.view(batch, out_len, -1)

    def _forward_sparse(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Stability-hardened sparse expert dispatch (in-place add to output).

        Args:
            x: Flattened input of shape [tokens, dim]
            output: Output tensor to add sparse expert results into (in-place)

        Returns:
            Auxiliary loss tensor
        """
        # A. Router Logic with Jitter
        logits = self.router(x)

        if self.training and self.router_jitter_noise > 0:
            # Jitter: multiply by uniform noise (1-eps, 1+eps) to shake decision boundary
            # Prevents router from getting stuck on one expert early in training
            noise = torch.empty_like(logits).uniform_(
                1.0 - self.router_jitter_noise, 1.0 + self.router_jitter_noise
            )
            logits = logits * noise

        # Force float32 for softmax (bf16/fp16 exponentials can overflow)
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(x)

        # B. Top-K Selection
        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # Normalize weights so they sum to 1.0
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # C. Aux Loss + Z-Loss
        aux_loss = torch.tensor(0.0, device=x.device)

        if self.training:
            # Load balancing loss (batch-size invariant)
            prob_per_expert = probs.mean(0)  # [num_experts]
            target = 1.0 / self.num_experts
            balance_loss = (
                self.aux_coef * ((prob_per_expert - target) ** 2).mean() * self.num_experts
            )

            # Z-loss: penalty on large logits to prevent softmax saturation
            z_loss = self.router_z_loss_coef * torch.logsumexp(logits, dim=-1).pow(2).mean()

            aux_loss = balance_loss + z_loss

        # D. Dispatch Loop (in-place add to output)
        for i, expert in enumerate(self.experts):
            # Create boolean mask for tokens that selected Expert 'i'
            mask = top_k_indices == i

            if mask.any():
                # token_idx = which tokens, k_idx = 1st or 2nd choice
                token_idx, k_idx = torch.where(mask)

                # Gather inputs and compute
                expert_input = x[token_idx]
                expert_output = expert(expert_input)

                # Apply routing weight
                weight = top_k_weights[token_idx, k_idx].unsqueeze(-1)
                weighted_output = (expert_output * weight).type_as(output)

                # Scatter back in-place (index_add_ is atomic and deterministic)
                output.index_add_(0, token_idx, weighted_output)

        return aux_loss

    def get_aux_loss(self) -> torch.Tensor:
        """Return auxiliary load balancing loss."""
        return self.last_aux_loss


# =============================================================================
# QFormer Projector (Granite-style)
# =============================================================================


class QFormerAudioProjector(nn.Module):
    """
    BLIP-2 QFormer projector with learnable queries.

    Based on GraniteSpeechEncoderProjector - uses a QFormer model with learnable
    query embeddings to compress and project audio encoder outputs. The audio
    sequence is processed in windows and downsampled via cross-attention.
    """

    def __init__(self, config):
        """Initialize QFormer projector.

        Args:
            config: ASRConfig with encoder_dim, llm_dim, qformer_* settings
        """
        super().__init__()

        encoder_dim = config.encoder_dim
        llm_dim = config.llm_dim

        # Window and downsampling parameters (Granite defaults: window=15, downsample=5)
        self.window_size = getattr(config, "qformer_window_size", 15)
        self.downsample_rate = getattr(config, "downsample_rate", 5)
        self.num_queries = self.window_size // self.downsample_rate

        # QFormer hidden size (matches encoder for cross-attention)
        qformer_hidden = getattr(config, "qformer_hidden_size", None) or encoder_dim
        qformer_num_layers = getattr(config, "qformer_num_layers", 2)
        qformer_num_heads = getattr(config, "qformer_num_heads", 16)
        qformer_intermediate = getattr(config, "qformer_intermediate_size", None) or (
            qformer_hidden * 4
        )

        # Learnable query embeddings (Granite uses std=1.0)
        self.query = nn.Parameter(torch.zeros(1, self.num_queries, qformer_hidden))
        self.query.data.normal_(mean=0.0, std=1.0)

        # Optional projection if encoder dim != qformer hidden
        if encoder_dim != qformer_hidden:
            self.encoder_proj = nn.Linear(encoder_dim, qformer_hidden, bias=False)
        else:
            self.encoder_proj = None

        # Configure QFormer to match Granite's exact config
        qformer_config = Blip2QFormerConfig(
            hidden_size=qformer_hidden,
            num_hidden_layers=qformer_num_layers,
            num_attention_heads=qformer_num_heads,
            intermediate_size=qformer_intermediate,
            encoder_hidden_size=qformer_hidden,
            cross_attention_frequency=1,
            # Granite-specific settings
            hidden_act="gelu",
            attention_probs_dropout_prob=0.1,
            hidden_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            initializer_range=0.02,
        )
        self.qformer = AutoModel.from_config(qformer_config)

        # Final projection to LLM dimension (Granite uses bias=True)
        self.linear = nn.Linear(qformer_hidden, llm_dim)

    def get_output_length(self, input_length):
        """Calculate output sequence length given input length.

        Accepts either Python ints or torch tensors; uses ceiling division so
        the formula is identical for both — math.ceil would block tensors.
        """
        nblocks = (input_length + self.window_size - 1) // self.window_size
        return nblocks * self.num_queries

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, encoder_dim]

        Returns:
            projected: [batch_size, num_output_tokens, llm_dim]
        """
        batch_size, seq_len, dim = hidden_states.size()

        # Ensure float dtype for QFormer
        target_dtype = self.query.dtype
        if hidden_states.dtype != target_dtype:
            hidden_states = hidden_states.to(target_dtype)

        # Optional encoder projection
        if self.encoder_proj is not None:
            hidden_states = self.encoder_proj(hidden_states)

        # Compute number of windows and pad to fit
        nblocks = math.ceil(seq_len / self.window_size)
        pad = nblocks * self.window_size - seq_len
        if pad > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad), "constant", 0)

        # Reshape to process each window: [batch*nblocks, window_size, dim]
        effective_batch = batch_size * nblocks
        hidden_states = hidden_states.view(effective_batch, self.window_size, -1)

        # Expand queries to match batch size
        query_embeds = self.query.expand(effective_batch, -1, -1)

        # QFormer cross-attention
        query_output = self.qformer(
            query_embeds=query_embeds,
            encoder_hidden_states=hidden_states,
            return_dict=True,
        )

        # Reshape back: [batch, nblocks * num_queries, hidden]
        output_tokens = nblocks * self.num_queries
        query_proj = query_output.last_hidden_state.view(batch_size, output_tokens, -1)

        # Project to LLM dimension
        return self.linear(query_proj)


# =============================================================================
# Projector Registry
# =============================================================================

PROJECTOR_CLASSES = {
    "mlp": MLPAudioProjector,
    "mosa": MOSAProjector,
    "moe": MoEAudioProjector,
    "qformer": QFormerAudioProjector,
}
