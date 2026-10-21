"""Tests for tiny_audio/qwen3_asr_encoder.py: unpacking, length arithmetic and weight loading."""

from pathlib import Path

import pytest
import torch
from huggingface_hub.errors import EntryNotFoundError
from safetensors.torch import save_file
from transformers.models.qwen3_asr import Qwen3ASRConfig, Qwen3ASREncoder, Qwen3ASREncoderConfig

from tiny_audio.asr_config import QWEN3_ASR_CHUNK_FRAMES, QWEN3_ASR_CONV_LAYERS
from tiny_audio.qwen3_asr_encoder import Qwen3ASRAudioEncoder

REPO = "Qwen/Qwen3-ASR-1.7B-hf"


def tiny_config() -> Qwen3ASREncoderConfig:
    """A two-layer encoder with the real chunking (n_window 50, 8 s windows), tiny widths."""
    return Qwen3ASREncoderConfig(
        num_mel_bins=16,
        encoder_layers=2,
        encoder_attention_heads=2,
        encoder_ffn_dim=64,
        d_model=32,
        downsample_hidden_size=8,
        output_dim=32,
        n_window=50,
        n_window_infer=800,
    )


@pytest.fixture
def encoder() -> Qwen3ASRAudioEncoder:
    """A randomly initialised tiny encoder in eval mode."""
    torch.manual_seed(0)
    return Qwen3ASRAudioEncoder(Qwen3ASREncoder(tiny_config())).eval()


def _mask(lengths: list[int], frames: int) -> torch.Tensor:
    return (torch.arange(frames)[None, :] < torch.tensor(lengths)[:, None]).long()


class TestForward:
    """The packed output is split back into a padded batch of the right lengths."""

    def test_lengths_follow_chunked_arithmetic(self, encoder: Qwen3ASRAudioEncoder) -> None:
        # 300 frames = three full 1 s chunks (13 each); 150 = one chunk + a 50-frame
        # remainder (50 -> 25 -> 13 -> 7).
        with torch.no_grad():
            out = encoder(torch.randn(2, 16, 300), _mask([300, 150], 300))
        assert out.last_hidden_state is not None
        assert out.attention_mask is not None
        assert out.last_hidden_state.shape == (2, 39, 32)
        assert out.attention_mask.sum(dim=-1).tolist() == [39, 20]

    def test_batched_row_matches_the_clip_alone(self, encoder: Qwen3ASRAudioEncoder) -> None:
        """Padding is dropped by the encoder, so a clip encodes the same in any batch."""
        feats = torch.randn(2, 16, 300)
        with torch.no_grad():
            batched = encoder(feats, _mask([300, 150], 300)).last_hidden_state
            alone = encoder(feats[1:, :, :200], _mask([150], 200)).last_hidden_state
        assert batched is not None
        assert alone is not None
        torch.testing.assert_close(batched[1, :20], alone[0, :20], rtol=1e-4, atol=1e-5)

    def test_constants_match_the_encoder(self) -> None:
        assert tiny_config().n_window * 2 == QWEN3_ASR_CHUNK_FRAMES
        assert len(QWEN3_ASR_CONV_LAYERS) == 3


class TestFromPretrained:
    """Only `model.audio_tower.*` tensors are read, and every one is required."""

    def _checkpoint(self, root: Path, source: Qwen3ASRAudioEncoder, drop: str = "") -> None:
        tensors = {
            f"model.audio_tower.{k}": v.contiguous() for k, v in source.encoder.state_dict().items()
        }
        tensors.pop(drop, None)
        tensors["model.language_model.embed_tokens.weight"] = torch.zeros(2, 2)
        save_file(tensors, str(root / "model.safetensors"))

    def _patch_hub(self, monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
        def download(repo_id: str, filename: str) -> str:
            assert repo_id == REPO
            if filename.endswith(".index.json"):
                raise EntryNotFoundError(filename)
            return str(root / filename)

        def config(*_args: object, **_kwargs: object) -> Qwen3ASRConfig:
            return Qwen3ASRConfig(audio_config=tiny_config().to_dict())

        monkeypatch.setattr("tiny_audio.qwen3_asr_encoder.hf_hub_download", download)
        monkeypatch.setattr(Qwen3ASRConfig, "from_pretrained", config)

    def test_loads_encoder_tensors(
        self, encoder: Qwen3ASRAudioEncoder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._checkpoint(tmp_path, encoder)
        self._patch_hub(monkeypatch, tmp_path)
        loaded = Qwen3ASRAudioEncoder.from_pretrained(REPO, torch.float32, None).eval()
        feats, mask = torch.randn(1, 16, 200), _mask([180], 200)
        with torch.no_grad():
            torch.testing.assert_close(
                loaded(feats, mask).last_hidden_state, encoder(feats, mask).last_hidden_state
            )

    def test_missing_tensor_fails_loudly(
        self, encoder: Qwen3ASRAudioEncoder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._checkpoint(tmp_path, encoder, drop="model.audio_tower.ln_post.weight")
        self._patch_hub(monkeypatch, tmp_path)
        with pytest.raises(RuntimeError, match=r"ln_post\.weight"):
            Qwen3ASRAudioEncoder.from_pretrained(REPO, torch.float32, None)
