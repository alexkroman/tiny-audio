"""Flow matching audio head for speech-to-speech.

Generates audio from LLM hidden states via flow matching:
  LLM hidden -> llm_proj -> flow_net (LSD decode) -> Mimi latents -> Mimi decoder -> audio

All components are trained from scratch for S2S.
"""

import logging
from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from .modules.mlp import SimpleMLPAdaLN

logger = logging.getLogger(__name__)


def lsd_decode(
    v_t,
    x_0: torch.Tensor,
    num_steps: int = 1,
) -> torch.Tensor:
    """Lagrangian Self-Distillation decoding.

    Iteratively refines noise into latents using the flow velocity network.

    Args:
        v_t: Velocity function v(s, t, x) -> velocity
        x_0: Initial noise, shape [N, latent_dim]
        num_steps: Number of integration steps

    Returns:
        Decoded latents, shape [N, latent_dim]
    """
    current = x_0
    for i in range(num_steps):
        s = i / num_steps
        t = (i + 1) / num_steps
        s_tensor = torch.full_like(x_0[..., :1], s)
        t_tensor = torch.full_like(x_0[..., :1], t)
        flow_dir = v_t(s_tensor, t_tensor, current)
        current = current + flow_dir / num_steps
    return current


class AudioHead(nn.Module):
    """Flow matching head: LLM hidden -> Mimi latents -> audio.

    Architecture:
        - llm_proj: Linear projection from LLM hidden dim to flow conditioning
        - latent_proj_in/out: Project between Mimi 512-dim and flow 32-dim
        - flow_net: SimpleMLPAdaLN that predicts flow velocity
        - Mimi decoder for latent -> audio

    Args:
        config: ASRConfig with:
            - llm_dim: LLM hidden dimension (default: 2048)
            - lsd_decode_steps: Number of LSD integration steps (default: 1)
            - flow_temperature: Sampling temperature for noise (default: 1.0)
    """

    # Architecture dimensions
    COND_DIM = 1024  # Conditioning dimension
    LATENT_DIM = 32  # Flow latent dimension (matches Mimi's 32 codebooks)
    MIMI_DIM = 512  # Mimi encoder output dimension
    FLOW_DIM = 512  # Flow network hidden dimension
    FLOW_DEPTH = 6  # Number of residual blocks

    def __init__(self, config, llm_dim: int = None):
        super().__init__()
        # llm_dim can be passed directly or from config
        self.llm_dim = llm_dim or getattr(config, "llm_dim", None) or 2048
        self.cond_dim = self.COND_DIM
        self.latent_dim = self.LATENT_DIM
        self.mimi_dim = self.MIMI_DIM
        self.lsd_steps = getattr(config, "lsd_decode_steps", 1)
        self.temp = getattr(config, "flow_temperature", 1.0)

        # LLM -> conditioning projection
        self.llm_proj = nn.Linear(self.llm_dim, self.cond_dim, bias=False)

        # Mimi embedding projections
        # Projects 512-dim Mimi embeddings to 32-dim flow latents and back
        self.latent_proj_in = nn.Linear(self.mimi_dim, self.latent_dim, bias=False)
        self.latent_proj_out = nn.Linear(self.latent_dim, self.mimi_dim, bias=False)

        # Flow network
        self.flow_net = SimpleMLPAdaLN(
            in_channels=self.latent_dim,
            model_channels=self.FLOW_DIM,
            out_channels=self.latent_dim,
            cond_channels=self.cond_dim,
            num_res_blocks=self.FLOW_DEPTH,
            num_time_conds=2,
        )

        # Mimi decoder components (loaded separately via load_mimi_decoder)
        self.mimi = None

    def load_mimi_decoder(self, device: torch.device = None, dtype: torch.dtype = None):
        """Load Mimi model for decoding latents to audio."""
        from transformers import MimiModel

        self.mimi = MimiModel.from_pretrained("kyutai/mimi")
        self.mimi.requires_grad_(False)
        self.mimi.eval()

        if device is not None:
            self.mimi = self.mimi.to(device)
        if dtype is not None:
            self.mimi = self.mimi.to(dtype)

        logger.info("Loaded Mimi decoder from kyutai/mimi")

    def forward(
        self,
        hidden_states: torch.Tensor,
        latent_targets: Optional[torch.Tensor] = None,
        latent_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for training or inference.

        Args:
            hidden_states: LLM hidden states, shape [batch, seq_len, llm_dim]
            latent_targets: Target Mimi latents for training, shape [batch, seq_len, 512]
            latent_lengths: Actual lengths per sample, shape [batch]

        Returns:
            Training: scalar flow matching loss
            Inference: generated Mimi latents, shape [batch, seq_len, 512]
        """
        # Project LLM hidden states to conditioning
        cond = self.llm_proj(hidden_states)

        if latent_targets is not None:
            return self._compute_loss(cond, latent_targets, latent_lengths)
        return self._generate(cond)

    def _compute_loss(
        self,
        cond: torch.Tensor,
        targets: torch.Tensor,
        lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute flow matching loss.

        Args:
            cond: Conditioning from LLM, shape [batch, cond_seq_len, cond_dim]
            targets: Mimi embeddings, shape [batch, target_seq_len, 512]
            lengths: Optional lengths for masking
        """
        # Debug: check inputs for NaN/Inf
        if torch.isnan(cond).any() or torch.isinf(cond).any():
            logger.warning(
                f"NaN/Inf in cond! shape={cond.shape}, nan={torch.isnan(cond).sum()}, inf={torch.isinf(cond).sum()}"
            )
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            logger.warning(f"NaN/Inf in targets! shape={targets.shape}")

        batch, cond_seq_len, _ = cond.shape
        target_seq_len = targets.shape[1]
        device = cond.device

        # Project 512-dim Mimi embeddings to 32-dim flow latents
        targets_proj = self.latent_proj_in(targets)

        # Interpolate targets to match conditioning sequence length
        if target_seq_len != cond_seq_len:
            targets_proj = targets_proj.transpose(1, 2)
            targets_proj = torch.nn.functional.interpolate(
                targets_proj, size=cond_seq_len, mode="linear", align_corners=False
            )
            targets_proj = targets_proj.transpose(1, 2).contiguous()

            if lengths is not None:
                scale = cond_seq_len / target_seq_len
                lengths = (lengths.float() * scale).long()

        seq_len = cond_seq_len
        x_1 = targets_proj

        # Random timesteps for each sample/position
        t = torch.rand(batch, seq_len, 1, device=device)

        # Sample noise
        x_0 = torch.randn_like(x_1)

        # Linear interpolation: x_t = (1-t) * x_0 + t * x_1
        x_t = (1 - t) * x_0 + t * x_1

        # Target velocity: dx/dt = x_1 - x_0
        v_target = x_1 - x_0

        # Flatten for flow_net: [batch * seq_len, dim]
        cond_flat = cond.view(-1, self.cond_dim)
        t_flat = t.view(-1, 1)
        x_t_flat = x_t.view(-1, self.latent_dim)

        # Predict velocity
        v_pred = self.flow_net(cond_flat, t_flat, t_flat, x_t_flat)
        v_pred = v_pred.view(batch, seq_len, -1)

        # Compute masked MSE loss
        if lengths is not None:
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            mask = positions < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).expand_as(v_pred)
            loss = ((v_pred - v_target) ** 2)[mask].mean()
        else:
            loss = ((v_pred - v_target) ** 2).mean()

        return loss

    def _generate(self, cond: torch.Tensor) -> torch.Tensor:
        """Generate Mimi embeddings via LSD decoding.

        Args:
            cond: Conditioning from LLM, shape [batch, seq_len, cond_dim]

        Returns:
            Generated Mimi embeddings, shape [batch, seq_len, 512]
        """
        batch, seq_len, _ = cond.shape
        device = cond.device
        dtype = cond.dtype

        latents = []
        for t in range(seq_len):
            cond_t = cond[:, t]

            # Sample initial noise in 32-dim flow space
            noise = torch.randn(batch, self.latent_dim, device=device, dtype=dtype)
            noise = noise * (self.temp**0.5)

            def velocity_fn(cond_fixed, s, t, x):
                return self.flow_net(cond_fixed, s, t, x)

            conditioned_flow = partial(velocity_fn, cond_t)
            latent = lsd_decode(conditioned_flow, noise, self.lsd_steps)
            latents.append(latent)

        latents = torch.stack(latents, dim=1)

        # Project back to 512-dim Mimi embedding space
        return self.latent_proj_out(latents)

    def decode_to_audio(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode Mimi latents to audio waveform.

        Note: HuggingFace MimiModel.decode() expects discrete codes, not continuous
        embeddings. We bypass the quantizer and call upsample → decoder_transformer
        → decoder directly to decode from continuous latents.

        Args:
            latents: Mimi latents, shape [batch, seq_len, 512]

        Returns:
            Audio waveform, shape [batch, samples]
        """
        if self.mimi is None:
            raise RuntimeError("Mimi decoder not loaded. Call load_mimi_decoder() first.")

        # [batch, seq, 512] → [batch, 512, seq]
        latents = latents.transpose(1, 2)

        with torch.no_grad():
            # Upsample latents (2x temporal upsampling)
            emb = self.mimi.upsample(latents)

            # Decoder transformer expects [batch, seq, dim]
            emb = emb.transpose(1, 2)
            decoder_out = self.mimi.decoder_transformer(emb)
            emb = getattr(decoder_out, "last_hidden_state", decoder_out[0])

            # Final decoder expects [batch, dim, seq]
            emb = emb.transpose(1, 2)
            audio = self.mimi.decoder(emb)

        return audio.squeeze(1)

    def get_output_length(self, input_length: int) -> int:
        """Estimate output audio frames from input hidden state length.

        For Mimi at 12.5 Hz frame rate with 24kHz audio:
        Each latent frame = 24000 / 12.5 = 1920 audio samples
        """
        return input_length * 1920
