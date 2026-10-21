"""Tests for tiny_audio/asr_packing.py and ASRModel's padding-free forward."""

from typing import Any, cast

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from tiny_audio.asr_modeling import ASRModel
from tiny_audio.asr_packing import pack_rows

# Two left-padded rows, as the training collator builds them: 2 and 0 pads.
MASK = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
LABELS = torch.tensor([[-100, -100, -100, 11, 12, 13], [-100, -100, 21, 22, 23, 24]])


class TestPackRows:
    """Real tokens are concatenated; positions and row boundaries come along."""

    def test_layout(self) -> None:
        embeds = torch.arange(12, dtype=torch.float).reshape(2, 6, 1)
        packed = pack_rows(embeds, MASK, LABELS)
        assert packed.inputs_embeds.flatten().tolist() == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        assert packed.position_ids.tolist() == [[0, 1, 2, 3, 0, 1, 2, 3, 4, 5]]
        assert packed.cu_seq_lens.tolist() == [0, 4, 10]
        assert packed.max_length == 6

    def test_row_starts_are_never_targets(self) -> None:
        """No row learns its first token from the previous row's last one."""
        labels = LABELS.clone()
        labels[1, 0] = 99  # a labelled first token would otherwise cross the boundary
        packed = pack_rows(torch.zeros(2, 6, 1), MASK, labels)
        assert packed.labels.tolist() == [[-100, 11, 12, 13, -100, -100, 21, 22, 23, 24]]


def _block_causal_mask(position_ids: torch.Tensor) -> torch.Tensor:
    """What FA2's varlen kernels compute, as a 4D mask: causal within each row only."""
    segment = (position_ids[0] == 0).cumsum(0)
    n = segment.shape[0]
    allowed = (segment[:, None] == segment[None, :]) & torch.ones(n, n, dtype=torch.bool).tril()
    mask = torch.zeros(n, n).masked_fill(~allowed, torch.finfo(torch.float).min)
    return mask[None, None]


def test_packed_loss_equals_padded_loss() -> None:
    """Packing changes no loss: same tokens, same targets, RoPE is relative."""
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    lm = LlamaForCausalLM(config).eval()
    lm.set_attn_implementation("eager")  # honours the explicit 4D mask
    embeds = torch.randn(2, 6, 32)
    # Each row's first real token is a prompt token, unlabelled, as in every
    # training batch. (Labelled, a left-padded row would "predict" it from a
    # fully masked pad position: noise that packing rightly drops.)
    labels = LABELS.clone()

    with torch.no_grad():
        padded = lm(inputs_embeds=embeds, attention_mask=MASK, labels=labels).loss
        packed = pack_rows(embeds, MASK, labels)
        flat = lm(
            inputs_embeds=packed.inputs_embeds,
            attention_mask=_block_causal_mask(packed.position_ids),
            position_ids=packed.position_ids,
            labels=packed.labels,
        ).loss
    assert padded is not None
    assert flat is not None
    torch.testing.assert_close(flat, padded, rtol=1e-5, atol=1e-5)


class _CapturingLM(nn.Module):
    """Wraps the real decoder, recording what ASRModel.forward hands it."""

    def __init__(self, real: Any) -> None:
        super().__init__()
        self.real = real
        self.config = real.config
        self.seen: dict[str, Any] = {}

    def get_input_embeddings(self) -> nn.Module:
        return self.real.get_input_embeddings()

    def forward(self, **kwargs: Any) -> CausalLMOutputWithPast:
        self.seen = kwargs
        return CausalLMOutputWithPast(loss=cast(torch.FloatTensor, torch.zeros(())))


class TestForwardPacking:
    """ASRModel packs only labelled, uncached batches, and only under FA2."""

    def _forward(
        self, model: ASRModel, monkeypatch: pytest.MonkeyPatch, attn: str
    ) -> dict[str, Any]:
        capture = _CapturingLM(model.language_model)
        monkeypatch.setattr(model, "language_model", capture)
        monkeypatch.setattr(model, "_lm_accepts_skip_logits", True)

        def active(_config: object) -> str:
            return attn

        monkeypatch.setattr("tiny_audio.asr_modeling.active_attn_implementation", active)
        input_ids = torch.randint(10, 100, (2, 6))
        model(input_ids=input_ids, attention_mask=MASK, labels=LABELS)
        return capture.seen

    def test_fa2_batches_are_packed(
        self, base_asr_model: ASRModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = self._forward(base_asr_model, monkeypatch, "flash_attention_2")
        assert seen["attention_mask"] is None
        assert seen["inputs_embeds"].shape[:2] == (1, 10)
        assert seen["cu_seq_lens_q"].tolist() == [0, 4, 10]

    def test_sdpa_batches_stay_padded(
        self, base_asr_model: ASRModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = self._forward(base_asr_model, monkeypatch, "sdpa")
        assert seen["inputs_embeds"].shape[:2] == (2, 6)
        assert "cu_seq_lens_q" not in seen

    def test_switched_off_by_config(
        self, base_asr_model: ASRModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(base_asr_model.config, "pack_sequences", False)
        seen = self._forward(base_asr_model, monkeypatch, "flash_attention_2")
        assert seen["inputs_embeds"].shape[:2] == (2, 6)
