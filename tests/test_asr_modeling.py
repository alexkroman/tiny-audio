"""Tests for ASRModel — projector dispatch, tokenizer init, embeddings, audio token counting."""

import pytest
import torch


class TestProjectorDispatch:
    """_create_projector should dispatch on projector_type."""

    def test_default_mlp_projector(self, base_asr_model):
        from tiny_audio.projectors import MLPAudioProjector

        assert isinstance(base_asr_model.projector, MLPAudioProjector)

    def test_unknown_projector_type_raises(self):
        from tiny_audio.asr_config import ASRConfig
        from tiny_audio.asr_modeling import ASRModel

        bad_config = ASRConfig(
            audio_model_id="openai/whisper-tiny",
            text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            attn_implementation="eager",
            model_dtype="float32",
            projector_type="not_a_real_projector",
        )
        with pytest.raises(ValueError, match="Unknown projector_type"):
            ASRModel(bad_config)


class TestTokenizerInit:
    """_init_tokenizer adds <audio> token and resizes embeddings."""

    def test_audio_token_is_in_tokenizer(self, base_asr_model):
        audio_id = base_asr_model.tokenizer.convert_tokens_to_ids("<audio>")
        assert audio_id is not None
        assert audio_id != base_asr_model.tokenizer.unk_token_id

    def test_audio_token_id_attribute(self, base_asr_model):
        assert base_asr_model.audio_token_id == base_asr_model.tokenizer.convert_tokens_to_ids(
            "<audio>"
        )

    def test_pad_token_set(self, base_asr_model):
        assert base_asr_model.tokenizer.pad_token is not None
        assert base_asr_model.tokenizer.pad_token_id is not None

    def test_padding_side_is_right(self, base_asr_model):
        assert base_asr_model.tokenizer.padding_side == "right"

    def test_embedding_resized_to_tokenizer_length(self, base_asr_model):
        embed = base_asr_model.language_model.get_input_embeddings()
        assert embed.num_embeddings >= len(base_asr_model.tokenizer)

    def test_generation_config_eos_synced(self, base_asr_model):
        eos_ids = base_asr_model.generation_config.eos_token_id
        assert eos_ids is None or all(e is not None for e in eos_ids)


class TestEmbeddings:
    """get_input_embeddings / set_input_embeddings / get_output_embeddings."""

    def test_get_input_embeddings_returns_module(self, base_asr_model):
        embed = base_asr_model.get_input_embeddings()
        assert isinstance(embed, torch.nn.Module)

    def test_get_output_embeddings_returns_module(self, base_asr_model):
        out = base_asr_model.get_output_embeddings()
        assert isinstance(out, torch.nn.Module)


class TestEncoderOutputLengths:
    """_compute_encoder_output_lengths and _get_num_audio_tokens use conv formula."""

    def test_compute_encoder_output_lengths_shape(self, base_asr_model):
        # whisper-tiny defaults: mel_len 3000 -> 1500 after conv
        attention_mask = torch.ones(2, 3000)
        lengths = base_asr_model._compute_encoder_output_lengths(attention_mask)
        assert lengths.shape == (2,)
        assert lengths[0].item() == 1500

    def test_get_num_audio_tokens_matches_projector(self, base_asr_model):
        attention_mask = torch.ones(1, 3000)
        n = base_asr_model._get_num_audio_tokens(attention_mask)
        assert n > 0
        assert isinstance(n, int)


class TestFeatureExtractor:
    """_create_feature_extractor returns a usable extractor."""

    def test_feature_extractor_attached(self, base_asr_model):
        assert base_asr_model.feature_extractor is not None
        assert base_asr_model.feature_extractor.sampling_rate == 16000


class TestStateDict:
    """state_dict only contains projector when LM is frozen."""

    def test_state_dict_only_has_projector_keys(self, base_asr_model):
        sd = base_asr_model.state_dict()
        assert all(k.startswith("projector.") for k in sd)

    def test_state_dict_includes_lm_when_unfrozen(self):
        from tiny_audio.asr_config import ASRConfig
        from tiny_audio.asr_modeling import ASRModel

        cfg = ASRConfig(
            audio_model_id="openai/whisper-tiny",
            text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            attn_implementation="eager",
            model_dtype="float32",
            freeze_language_model=False,
        )
        model = ASRModel(cfg)
        sd = model.state_dict()
        assert any(k.startswith("language_model.") for k in sd)
        assert any(k.startswith("projector.") for k in sd)


class TestLoadAudioEncoder:
    """_load_audio_encoder dispatches based on audio_model_id substring."""

    def test_whisper_branch_loads_encoder_only(self, base_asr_model):
        # base_asr_model uses whisper-tiny → audio_tower should be Whisper's encoder
        # (not the full WhisperModel)
        from transformers.models.whisper.modeling_whisper import WhisperEncoder

        assert isinstance(base_asr_model.audio_tower, WhisperEncoder)

    def test_audio_encoder_is_frozen(self, base_asr_model):
        for p in base_asr_model.audio_tower.parameters():
            assert p.requires_grad is False

    def test_audio_encoder_in_eval_mode(self, base_asr_model):
        assert base_asr_model.audio_tower.training is False

    def test_glm_branch_uses_audio_tower(self, monkeypatch):
        """Verify GLM dispatch path without downloading the real GLM model."""
        from unittest.mock import MagicMock

        from tiny_audio.asr_modeling import ASRModel

        # Mock AutoModelForSeq2SeqLM.from_pretrained to return a mock with audio_tower
        mock_full = MagicMock()
        mock_full.audio_tower = MagicMock(spec=torch.nn.Module)
        mock_full.audio_tower.requires_grad_ = MagicMock()
        mock_full.audio_tower.eval = MagicMock()

        with monkeypatch.context() as m:
            mock_loader = MagicMock(return_value=mock_full)
            m.setattr("transformers.AutoModelForSeq2SeqLM.from_pretrained", mock_loader)

            cfg = MagicMock()
            cfg.audio_model_id = "zai-org/GLM-ASR-something"
            cfg.attn_implementation = "eager"

            encoder = ASRModel._load_audio_encoder(cfg, torch.float32)

            # Should have called the GLM loader, not WhisperModel
            mock_loader.assert_called_once()
            assert encoder is mock_full.audio_tower
            mock_full.audio_tower.requires_grad_.assert_called_with(False)
            mock_full.audio_tower.eval.assert_called_once()


class TestLoadLanguageModel:
    """_load_language_model freezes LM by default."""

    def test_lm_frozen_by_default(self, base_asr_model):
        for p in base_asr_model.language_model.parameters():
            assert p.requires_grad is False

    def test_use_cache_synced(self, base_asr_model):
        assert base_asr_model.language_model.config.use_cache is True


class TestLoRASetup:
    """_setup_lora wraps language_model with PEFT when use_lora=True."""

    def test_lora_model_has_peft_config(self, lora_asr_model):
        assert hasattr(lora_asr_model.language_model, "peft_config")

    def test_lora_target_modules_applied(self, lora_asr_model):
        # PEFT replaces target Linear layers with LoraLayer wrappers
        from peft.tuners.lora import LoraLayer

        has_lora = any(isinstance(m, LoraLayer) for m in lora_asr_model.language_model.modules())
        assert has_lora

    def test_non_lora_model_has_no_peft(self, base_asr_model):
        assert not hasattr(base_asr_model.language_model, "peft_config")


class TestFreezeProjector:
    """freeze_projector=True freezes projector params."""

    def test_freeze_projector_disables_grad(self):
        from tiny_audio.asr_config import ASRConfig
        from tiny_audio.asr_modeling import ASRModel

        cfg = ASRConfig(
            audio_model_id="openai/whisper-tiny",
            text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            attn_implementation="eager",
            model_dtype="float32",
            freeze_projector=True,
        )
        model = ASRModel(cfg)
        for p in model.projector.parameters():
            assert p.requires_grad is False


class TestForward:
    """forward() handles text-only and audio+text inputs."""

    def test_forward_text_only(self, base_asr_model):
        """Forward without audio inputs (pure text path)."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            out = base_asr_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        assert out.logits.shape[0] == 1
        assert out.logits.shape[1] == 5

    def test_forward_with_labels_returns_loss(self, base_asr_model):
        """Loss is computed when labels are passed."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        labels = input_ids.clone()
        attention_mask = torch.ones_like(input_ids)

        out = base_asr_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        assert out.loss is not None


class TestAudioTokenDropout:
    """_maybe_drop_audio_tokens zeros whole encoder frames during training."""

    def test_disabled_when_dropout_zero(self, base_asr_model):
        base_asr_model.config.audio_token_dropout = 0.0
        base_asr_model.train(True)
        try:
            x = torch.randn(2, 10, 16)
            out = base_asr_model._maybe_drop_audio_tokens(x)
            torch.testing.assert_close(out, x)
        finally:
            base_asr_model.train(False)

    def test_disabled_when_not_training(self, base_asr_model):
        base_asr_model.config.audio_token_dropout = 0.5
        base_asr_model.train(False)
        x = torch.randn(2, 10, 16)
        out = base_asr_model._maybe_drop_audio_tokens(x)
        torch.testing.assert_close(out, x)

    def test_zeros_whole_frames_in_train_mode(self, base_asr_model):
        base_asr_model.config.audio_token_dropout = 0.5
        base_asr_model.train(True)
        try:
            torch.manual_seed(0)
            x = torch.randn(4, 100, 8)
            out = base_asr_model._maybe_drop_audio_tokens(x)
            # When a frame is dropped, ALL feature dims for that time step
            # are zero (broadcast mask). Surviving frames are unchanged.
            frame_norms = out.abs().sum(dim=-1)
            zero_frames = (frame_norms == 0).float().mean().item()
            # 0.5 drop rate plus noise on a 4x100 grid; well within bounds.
            assert 0.3 < zero_frames < 0.7
            # Surviving frames preserve magnitude (no rescaling).
            survivors = frame_norms > 0
            torch.testing.assert_close(out[survivors], x[survivors])
        finally:
            base_asr_model.train(False)


class TestGenerate:
    """generate() validates inputs and returns generated tokens."""

    def test_generate_requires_input_features(self, base_asr_model):
        with pytest.raises(ValueError, match="input_features"):
            base_asr_model.generate(input_ids=torch.tensor([[1, 2, 3]]))

    def test_generate_requires_audio_attention_mask(self, base_asr_model):
        with pytest.raises(ValueError, match="audio_attention_mask"):
            base_asr_model.generate(
                input_features=torch.randn(1, 80, 3000),
                input_ids=torch.tensor([[1, 2, 3]]),
            )

    def test_generate_returns_tokens(self, base_asr_model):
        """Generate end-to-end with synthetic mel features (whisper-tiny)."""
        # Whisper-tiny expects 80 mel bins, 3000 frames (30s of audio at 16kHz)
        input_features = torch.zeros(1, 80, 3000)
        audio_attention_mask = torch.ones(1, 3000, dtype=torch.long)

        out = base_asr_model.generate(
            input_features=input_features,
            audio_attention_mask=audio_attention_mask,
            max_new_tokens=4,
        )
        # Returns only generated tokens (input is stripped)
        assert out.dim() == 2
        assert out.shape[0] == 1
        assert out.shape[1] <= 4


class TestGenerateStreaming:
    """generate_streaming yields partial transcript pieces."""

    def test_streaming_yields_strings(self, base_asr_model):
        input_features = torch.zeros(1, 80, 3000)
        audio_attention_mask = torch.ones(1, 3000, dtype=torch.long)

        outputs = list(
            base_asr_model.generate_streaming(
                input_features=input_features,
                audio_attention_mask=audio_attention_mask,
                max_new_tokens=4,
            )
        )
        # Each yielded piece is a string (possibly empty)
        for piece in outputs:
            assert isinstance(piece, str)


class TestSavePretrained:
    """save_pretrained writes config, weights, tokenizer, and source files."""

    def test_save_creates_expected_files(self, base_asr_model, tmp_path):
        save_dir = tmp_path / "model"
        base_asr_model.save_pretrained(save_dir)

        assert (save_dir / "config.json").exists()
        assert (save_dir / "model.safetensors").exists()
        assert (save_dir / "tokenizer_config.json").exists()
        # asr_*.py files copied for auto-loading
        assert (save_dir / "asr_modeling.py").exists()
        assert (save_dir / "asr_config.py").exists()
        assert (save_dir / "asr_processing.py").exists()
        assert (save_dir / "asr_pipeline.py").exists()
        assert (save_dir / "projectors.py").exists()
        assert (save_dir / "alignment.py").exists()
        assert (save_dir / "diarization.py").exists()

    def test_save_then_load_round_trip(self, base_asr_model, tmp_path):
        from tiny_audio.asr_modeling import ASRModel

        save_dir = tmp_path / "model"
        base_asr_model.save_pretrained(save_dir)

        loaded = ASRModel.from_pretrained(str(save_dir))

        # Check projector weights match
        original_proj = dict(base_asr_model.projector.state_dict())
        loaded_proj = dict(loaded.projector.state_dict())
        assert original_proj.keys() == loaded_proj.keys()
        for k, v in original_proj.items():
            assert torch.allclose(v, loaded_proj[k])

    def test_save_lora_writes_adapter_config(self, lora_asr_model, tmp_path):
        save_dir = tmp_path / "lora_model"
        lora_asr_model.save_pretrained(save_dir)

        # PEFT writes these
        assert (save_dir / "adapter_config.json").exists()
        assert (save_dir / "adapter_model.safetensors").exists()

    def test_save_lora_clears_base_model_path_when_no_repo_id(self, lora_asr_model, tmp_path):
        import json

        save_dir = tmp_path / "lora_model"
        lora_asr_model.save_pretrained(save_dir)

        with (save_dir / "adapter_config.json").open() as f:
            adapter_cfg = json.load(f)

        # Should be empty string (not None / "None") when no repo_id is given
        assert adapter_cfg["base_model_name_or_path"] == ""

    def test_save_lora_uses_repo_id_when_provided(self, lora_asr_model, tmp_path):
        import json

        save_dir = tmp_path / "lora_model_with_repo"
        lora_asr_model.save_pretrained(save_dir, repo_id="alex/test-model")

        with (save_dir / "adapter_config.json").open() as f:
            adapter_cfg = json.load(f)

        assert adapter_cfg["base_model_name_or_path"] == "alex/test-model"


class TestProcessor:
    """get_processor wires together feature extractor, tokenizer, projector."""

    def test_get_processor_returns_asrprocessor(self, base_asr_model):
        from tiny_audio.asr_processing import ASRProcessor

        proc = base_asr_model.get_processor()
        assert isinstance(proc, ASRProcessor)
        assert proc.feature_extractor is base_asr_model.feature_extractor
        assert proc.tokenizer is base_asr_model.tokenizer
