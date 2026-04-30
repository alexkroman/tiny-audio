"""Tests for _gather_audio_embeds — the helper used by ASRModel._encode_audio."""

import torch

from tiny_audio.asr_modeling import _gather_audio_embeds


def _gather_reference(audio_embeds: torch.Tensor, token_counts: torch.Tensor) -> torch.Tensor:
    """Per-sample slice + cat — the implementation we are replacing."""
    batch_size, _, hidden_dim = audio_embeds.shape
    parts = []
    for i in range(batch_size):
        count = int(token_counts[i].item())
        sample = audio_embeds[i, :count, :]
        if sample.shape[0] < count:
            pad = torch.zeros(
                count - sample.shape[0],
                hidden_dim,
                device=audio_embeds.device,
                dtype=audio_embeds.dtype,
            )
            sample = torch.cat([sample, pad], dim=0)
        parts.append(sample)
    return torch.cat(parts, dim=0) if parts else torch.zeros(0, hidden_dim)


class TestGatherAudioEmbeds:
    def test_matches_reference_balanced_batch(self):
        torch.manual_seed(0)
        embeds = torch.randn(4, 10, 8)
        counts = torch.tensor([10, 5, 7, 3])
        out_ref = _gather_reference(embeds, counts)
        out_vec = _gather_audio_embeds(embeds, counts)
        assert out_vec.shape == out_ref.shape == (25, 8)
        torch.testing.assert_close(out_vec, out_ref)

    def test_zero_count_sample(self):
        embeds = torch.randn(3, 6, 4)
        counts = torch.tensor([6, 0, 2])
        out_ref = _gather_reference(embeds, counts)
        out_vec = _gather_audio_embeds(embeds, counts)
        assert out_vec.shape == (8, 4)
        torch.testing.assert_close(out_vec, out_ref)

    def test_count_exceeds_max_len_pads_with_zero(self):
        embeds = torch.ones(2, 4, 3)
        counts = torch.tensor([4, 6])  # second sample wants 2 more than available
        out_ref = _gather_reference(embeds, counts)
        out_vec = _gather_audio_embeds(embeds, counts)
        assert out_vec.shape == (10, 3)
        torch.testing.assert_close(out_vec, out_ref)
        assert torch.equal(out_vec[-2:], torch.zeros(2, 3))

    def test_all_zero_counts(self):
        embeds = torch.randn(2, 5, 4)
        counts = torch.tensor([0, 0])
        out_vec = _gather_audio_embeds(embeds, counts)
        assert out_vec.shape == (0, 4)
