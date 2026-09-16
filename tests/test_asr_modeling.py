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

    @pytest.mark.parametrize("layout", ["nested", "flat"])
    def test_glm_branch_uses_audio_tower(self, monkeypatch, layout):
        """Verify GLM dispatch path without downloading the real GLM model.

        Both submodule layouts must work. transformers 5.x makes
        GlmAsrForConditionalGeneration a wrapper holding a GlmAsrModel at
        `.model`, and that inner model owns audio_tower; older versions hung
        audio_tower off the top-level model.
        """
        from unittest.mock import MagicMock

        from tiny_audio.asr_modeling import ASRModel

        tower = MagicMock(spec=torch.nn.Module)
        tower.requires_grad_ = MagicMock()
        tower.train = MagicMock()
        # _load_audio_encoder applies an idempotent dtype cast post-load
        # (`encoder = encoder.to(dtype=dtype)`), so the returned encoder is
        # whatever audio_tower.to() yields. Pin .to() to return the tower
        # itself so the identity assertion below still describes the
        # logical "encoder == the mocked audio_tower".
        tower.to = MagicMock(return_value=tower)

        # Mock AutoModelForSeq2SeqLM.from_pretrained to return a mock whose
        # audio_tower sits at the layout under test.
        mock_full = MagicMock()
        if layout == "nested":
            mock_full.model.audio_tower = tower
        else:
            mock_full.audio_tower = tower
            # A bare MagicMock would auto-create `.model.audio_tower`, so the
            # nested branch would always win and the flat path would never be
            # exercised. Delete `.model` to make attribute access raise.
            del mock_full.model

        with monkeypatch.context() as m:
            mock_loader = MagicMock(return_value=mock_full)
            m.setattr("transformers.AutoModelForSeq2SeqLM.from_pretrained", mock_loader)

            cfg = MagicMock()
            cfg.audio_model_id = "zai-org/GLM-ASR-something"
            cfg.attn_implementation = "eager"
            cfg.freeze_audio_encoder = True

            encoder = ASRModel._load_audio_encoder(cfg, torch.float32)

            # Should have called the GLM loader, not WhisperModel
            mock_loader.assert_called_once()
            assert encoder is tower
            tower.to.assert_called_once_with(dtype=torch.float32)
            tower.requires_grad_.assert_called_with(False)
            # Frozen encoder gets switched to inference mode via `.train(False)`.
            tower.train.assert_called_once_with(False)

    def test_glm_branch_raises_when_audio_tower_missing(self, monkeypatch):
        """A future layout change should fail loudly, not hand back a stub."""
        from unittest.mock import MagicMock

        from tiny_audio.asr_modeling import ASRModel

        mock_full = MagicMock()
        del mock_full.model
        del mock_full.audio_tower

        with monkeypatch.context() as m:
            m.setattr(
                "transformers.AutoModelForSeq2SeqLM.from_pretrained",
                MagicMock(return_value=mock_full),
            )

            cfg = MagicMock()
            cfg.audio_model_id = "zai-org/GLM-ASR-something"
            cfg.attn_implementation = "eager"
            cfg.freeze_audio_encoder = True

            with pytest.raises(AttributeError, match="audio_tower"):
                ASRModel._load_audio_encoder(cfg, torch.float32)


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


class TestGradientCheckpointing:
    """ASRModel overrides `_set_gradient_checkpointing`, so its signature has to
    stay compatible with whatever `PreTrainedModel.gradient_checkpointing_enable`
    passes. transformers 5.16 added a keyword-only `every_n_layers` and the old
    two-argument override raised TypeError only after the model and the dataset
    had finished loading -- an expensive way to learn about a signature change.
    These tests drive the real upstream entry point rather than calling the
    override directly, so a future kwarg fails here instead of on a GPU."""

    def test_enable_via_upstream_entry_point(self, base_asr_model):
        base_asr_model.gradient_checkpointing_enable()
        assert base_asr_model.language_model.is_gradient_checkpointing
        base_asr_model.gradient_checkpointing_disable()
        assert not base_asr_model.language_model.is_gradient_checkpointing

    def test_enable_accepts_upstream_kwargs(self, base_asr_model):
        # Mirrors the exact call transformers.Trainer makes.
        base_asr_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
            every_n_layers=1,
        )
        assert base_asr_model.language_model.is_gradient_checkpointing
        base_asr_model.gradient_checkpointing_disable()

    def test_signature_matches_upstream(self, base_asr_model):
        import inspect

        from transformers.modeling_utils import PreTrainedModel

        upstream = inspect.signature(PreTrainedModel._set_gradient_checkpointing).parameters
        ours = inspect.signature(base_asr_model._set_gradient_checkpointing).parameters
        accepts_var_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in ours.values())
        missing = [n for n in upstream if n != "self" and n not in ours]
        assert accepts_var_kwargs or not missing, f"override missing kwargs: {missing}"

    def test_does_not_expose_legacy_value_param(self, base_asr_model):
        # Upstream treats a `value` parameter as the pre-4.35 format and routes
        # to a different code path entirely, silently skipping our override.
        import inspect

        ours = inspect.signature(base_asr_model._set_gradient_checkpointing).parameters
        assert "value" not in ours

    def test_frozen_encoder_is_not_checkpointed(self, base_asr_model):
        # Nothing backprops through a frozen encoder, so checkpointing it is
        # pure recompute cost for zero memory saved.
        targets = base_asr_model._gradient_checkpointing_targets()
        assert base_asr_model.language_model in targets
        assert base_asr_model.audio_tower not in targets


class TestFlashAttentionHeadDimGuard:
    """FlashAttention has kernels only for head_dim <= 256 and raises at the
    first forward, not at load -- so without a config-time guard the failure
    lands minutes into a run. Gemma 4 E2B is the motivating case: 7 of its 35
    layers use head_dim=512."""

    def test_reads_uniform_head_dim(self):
        from tiny_audio.asr_modeling import _max_attention_head_dim

        class Cfg:
            head_dim = 128

        assert _max_attention_head_dim(Cfg()) == 128

    def test_takes_max_across_heterogeneous_layers(self):
        from tiny_audio.asr_modeling import _max_attention_head_dim

        class Layer:
            def __init__(self, d):
                self.head_dim = d

        class Cfg:
            per_layer_config = [Layer(256), Layer(512), Layer(256)]

        assert _max_attention_head_dim(Cfg()) == 512

    def test_per_layer_wins_over_ambiguous_global(self):
        # Heterogeneous configs raise rather than return a number when the
        # global attribute is read, so per_layer_config must be consulted first.
        from tiny_audio.asr_modeling import _max_attention_head_dim

        class Layer:
            def __init__(self, d):
                self.head_dim = d

        class Cfg:
            per_layer_config = [Layer(256), Layer(512)]

            @property
            def head_dim(self):
                raise RuntimeError("ambiguous per-layer attribute")

        assert _max_attention_head_dim(Cfg()) == 512

    def test_unknown_head_dim_is_none(self):
        # None means "cannot determine" and must not be treated as 0, which
        # would wrongly leave FA2 enabled or wrongly disable it.
        from tiny_audio.asr_modeling import _max_attention_head_dim

        class Cfg:
            pass

        assert _max_attention_head_dim(Cfg()) is None

    def test_guard_threshold_matches_flash_attention(self):
        from tiny_audio.asr_modeling import FLASH_ATTENTION_MAX_HEAD_DIM

        assert FLASH_ATTENTION_MAX_HEAD_DIM == 256


class TestGemmaDecodeLoopPatch:
    """Gemma 4's causal LM needs two things patched for audio generation to
    work at all, and both failed in ways that only showed up at inference:
    `per_layer_inputs` must be dropped after the first decode step, and the
    wrapper must not hide `inputs_embeds` from signature introspection."""

    @staticmethod
    def _stub():
        class Stub:
            def prepare_inputs_for_generation(
                self,
                input_ids,
                inputs_embeds=None,
                per_layer_inputs=None,
                is_first_iteration=False,
                **kwargs,
            ):
                return {
                    "input_ids": input_ids,
                    "inputs_embeds": inputs_embeds,
                    "per_layer_inputs": per_layer_inputs,
                }

        return Stub()

    def test_keeps_per_layer_inputs_on_first_step(self):
        from tiny_audio.asr_modeling import _patch_gemma_decode_loop

        stub = self._stub()
        _patch_gemma_decode_loop(stub)
        out = stub.prepare_inputs_for_generation(
            None, inputs_embeds="E", per_layer_inputs="PLE", is_first_iteration=True
        )
        assert out["per_layer_inputs"] == "PLE"

    def test_drops_per_layer_inputs_on_later_steps(self):
        # Step 2+ passes input_ids, and Gemma's forward raises if PLE comes
        # along with it.
        from tiny_audio.asr_modeling import _patch_gemma_decode_loop

        stub = self._stub()
        _patch_gemma_decode_loop(stub)
        out = stub.prepare_inputs_for_generation(
            [[1]], per_layer_inputs="PLE", is_first_iteration=False
        )
        assert "per_layer_inputs" not in out

    def test_wrapper_preserves_inputs_embeds_in_signature(self):
        # generation._prepare_model_inputs gates inputs_embeds support on this
        # exact introspection; a bare *args wrapper makes generate() reject
        # inputs_embeds, which is the only way audio reaches the decoder.
        import inspect

        from tiny_audio.asr_modeling import _patch_gemma_decode_loop

        stub = self._stub()
        _patch_gemma_decode_loop(stub)
        params = set(inspect.signature(stub.prepare_inputs_for_generation).parameters)
        assert "inputs_embeds" in params


class TestEosTokenResolution:
    """generation_config.eos_token_id must hold only real vocab entries."""

    def test_excludes_tokens_absent_from_vocab(self, base_asr_model):
        # SmolLM2 has no "<end_of_turn>"; convert_tokens_to_ids falls back to
        # unk rather than returning None, which is how Gemma ended up with
        # eos_token_id=[unk, unk] and never stopped generating.
        tok = base_asr_model.tokenizer
        for token_id in base_asr_model.generation_config.eos_token_id:
            assert tok.convert_ids_to_tokens(token_id) != "<end_of_turn>"

    def test_keeps_endoftext_even_though_it_is_also_unk(self, base_asr_model):
        # SmolLM2's unk_token IS "<|endoftext|>", so filtering on
        # `id == unk_token_id` would drop a legitimate stop token.
        tok = base_asr_model.tokenizer
        endoftext = tok.convert_tokens_to_ids("<|endoftext|>")
        assert endoftext in base_asr_model.generation_config.eos_token_id

    def test_includes_tokenizer_eos(self, base_asr_model):
        assert (
            base_asr_model.tokenizer.eos_token_id in base_asr_model.generation_config.eos_token_id
        )

    def test_is_non_empty(self, base_asr_model):
        assert base_asr_model.generation_config.eos_token_id


class TestStateDictTrainableModules:
    """state_dict must serialize every module the freeze flags leave trainable."""

    def test_frozen_encoder_is_not_saved(self, base_asr_model):
        keys = base_asr_model.state_dict()
        assert not any(k.startswith("audio_tower.") for k in keys)
        assert any(k.startswith("projector.") for k in keys)

    def test_unfrozen_encoder_is_saved(self):
        from tiny_audio.asr_config import ASRConfig
        from tiny_audio.asr_modeling import ASRModel

        config = ASRConfig(
            audio_model_id="openai/whisper-tiny",
            text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            projector_type="mlp",
            model_dtype="float32",
            attn_implementation="eager",
            freeze_audio_encoder=False,
        )
        model = ASRModel(config)

        keys = model.state_dict()
        assert any(k.startswith("audio_tower.") for k in keys)

        # The saved keys must be the ones load_state_dict expects, or the
        # round-trip silently drops them under strict=False.
        result = model.load_state_dict(keys, strict=False)
        assert result.unexpected_keys == []

    def test_encoder_keys_cover_trainable_encoder_params(self):
        from tiny_audio.asr_config import ASRConfig
        from tiny_audio.asr_modeling import ASRModel

        config = ASRConfig(
            audio_model_id="openai/whisper-tiny",
            text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            projector_type="mlp",
            model_dtype="float32",
            attn_implementation="eager",
            freeze_audio_encoder=False,
        )
        model = ASRModel(config)

        saved = {
            k[len("audio_tower.") :] for k in model.state_dict() if k.startswith("audio_tower.")
        }
        trainable = {n for n, p in model.audio_tower.named_parameters() if p.requires_grad}
        assert trainable, "encoder should be trainable when freeze_audio_encoder=False"
        assert trainable <= saved

    def test_encoder_updates_survive_save_reload(self, tmp_path):
        """End-to-end: the checkpoint a training run writes must carry the encoder."""
        import torch

        from tiny_audio.asr_config import ASRConfig
        from tiny_audio.asr_modeling import ASRModel

        config = ASRConfig(
            audio_model_id="openai/whisper-tiny",
            text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            projector_type="mlp",
            model_dtype="float32",
            attn_implementation="eager",
            freeze_audio_encoder=False,
        )
        model = ASRModel(config)

        # Stand in for what training would do to the encoder.
        probe_name, probe = next(iter(model.audio_tower.named_parameters()))
        with torch.no_grad():
            probe.add_(1.0)
        expected = probe.detach().clone()

        model.save_pretrained(tmp_path)
        reloaded = ASRModel.from_pretrained(str(tmp_path))

        actual = dict(reloaded.audio_tower.named_parameters())[probe_name]
        assert torch.allclose(actual, expected), "encoder updates lost on reload"


class TestFreezePerLayerEmbeddings:
    """freeze_text_per_layer_embeddings targets Gemma 4's second vocab table.

    `embed_tokens_per_layer` is a per-token lookup separate from
    `embed_tokens`, so `freeze_text_embed_tokens` does not reach it. A tiny
    real Gemma4TextModel stands in for E2B here -- loading the 9.5 GiB
    checkpoint just to assert requires_grad would be absurd, but the layout
    the locator walks is the class's, not the checkpoint's.
    """

    @staticmethod
    def _tiny_gemma_text_model():
        from transformers.models.gemma4 import modeling_gemma4 as gemma4

        config = gemma4.Gemma4TextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            hidden_size_per_layer_input=8,
            attn_implementation="eager",
        )
        return gemma4.Gemma4TextModel(config)

    @staticmethod
    def _locator_for(language_model):
        """Bind the real locator methods to a stand-in holding `language_model`.

        Avoids mutating the session-scoped ASRModel fixture (and nn.Module
        rejects non-module children anyway); the methods under test read
        nothing but `self.language_model`.
        """
        from tiny_audio.asr_modeling import ASRModel

        class _Stub:
            _ple_text_model = ASRModel._ple_text_model
            _per_layer_embedding_table = ASRModel._per_layer_embedding_table

            def __init__(self, lm):
                self.language_model = lm

        return _Stub(language_model)

    def test_locator_finds_table_on_text_only_layout(self):
        """A text-only Gemma sits at language_model.model (see _ple_text_model)."""
        import types

        text_model = self._tiny_gemma_text_model()
        locator = self._locator_for(types.SimpleNamespace(model=text_model))
        assert locator._per_layer_embedding_table() is text_model.embed_tokens_per_layer

    def test_locator_finds_table_on_multimodal_layout(self):
        """The wrapper checkpoint nests it one level deeper."""
        import types

        text_model = self._tiny_gemma_text_model()
        wrapper = types.SimpleNamespace(model=types.SimpleNamespace(language_model=text_model))
        locator = self._locator_for(wrapper)
        assert locator._per_layer_embedding_table() is text_model.embed_tokens_per_layer

    def test_locator_returns_none_without_per_layer_table(self, base_asr_model):
        """SmolLM2 has no per-layer embeddings, so the flag has nothing to freeze."""
        assert base_asr_model._per_layer_embedding_table() is None

    def test_freeze_marks_only_the_per_layer_table(self):
        """The table stops requiring grad; the rest of the decoder keeps training."""
        import types

        text_model = self._tiny_gemma_text_model()
        locator = self._locator_for(types.SimpleNamespace(model=text_model))

        table = locator._per_layer_embedding_table()
        assert table.weight.requires_grad, "precondition: starts trainable"
        table.weight.requires_grad_(False)

        assert not text_model.embed_tokens_per_layer.weight.requires_grad
        # The small per-layer projections are ordinary machinery, not a lookup
        # table, and must stay trainable.
        assert text_model.per_layer_model_projection.weight.requires_grad
        assert text_model.embed_tokens.weight.requires_grad

    def test_flag_is_a_no_op_and_warns_on_non_gemma(self, caplog):
        """Setting it on Qwen3/Llama-style decoders must not fail silently."""
        import logging as _logging

        from tiny_audio.asr_config import ASRConfig
        from tiny_audio.asr_modeling import ASRModel

        config = ASRConfig(
            audio_model_id="openai/whisper-tiny",
            text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            projector_type="mlp",
            model_dtype="float32",
            attn_implementation="eager",
            freeze_language_model=False,
            freeze_text_per_layer_embeddings=True,
        )
        with caplog.at_level(_logging.WARNING):
            model = ASRModel(config)

        assert "no per-layer embedding table" in caplog.text
        # Decoder is still trainable -- the flag did not freeze anything else.
        assert any(p.requires_grad for p in model.language_model.parameters())

    def test_config_roundtrips_the_flag(self):
        from tiny_audio.asr_config import ASRConfig

        config = ASRConfig(freeze_text_per_layer_embeddings=True)
        assert config.freeze_text_per_layer_embeddings is True
        assert ASRConfig().freeze_text_per_layer_embeddings is False
        assert ASRConfig.from_dict(config.to_dict()).freeze_text_per_layer_embeddings is True
