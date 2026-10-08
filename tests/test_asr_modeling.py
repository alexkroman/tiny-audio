"""Tests for ASRModel — projector dispatch, tokenizer init, embeddings, audio token counting."""

import inspect
import json
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar, cast
from unittest.mock import MagicMock

import pytest
import torch
from conftest import eos_ids, gemma_decode_loop_stub, generation_settings, input_embedding_rows
from peft.tuners.lora import LoraLayer
from torch.nn.modules.module import _IncompatibleKeys
from transformers.modeling_utils import PreTrainedModel
from transformers.models.whisper.modeling_whisper import WhisperEncoder

import tiny_audio.asr_layers as layers_mod
from tiny_audio.asr_attention import (
    FLASH_ATTENTION_MAX_HEAD_DIM,
    _max_attention_head_dim,
    resolve_attn_implementation,
)
from tiny_audio.asr_config import ASRConfig
from tiny_audio.asr_layers import (
    MPS_MAX_TENSOR_ELEMENTS,
    chunk_oversized_embeddings,
    mps_unsafe_parameters,
)
from tiny_audio.asr_modeling import (
    VOCAB_PAD_MULTIPLE,
    ASRModel,
    _assert_projector_loaded,
    _patch_gemma_decode_loop,
)
from tiny_audio.asr_processing import ASRProcessor
from tiny_audio.asr_types import LoadStateDictResult, apply_chat_template
from tiny_audio.projectors import MLPAudioProjector

if TYPE_CHECKING:
    from tiny_audio.asr_types import GenerativeDecoder


class TestProjectorDispatch:
    """_create_projector should dispatch on projector_type."""

    def test_default_mlp_projector(self, base_asr_model: ASRModel) -> None:
        assert isinstance(base_asr_model.projector, MLPAudioProjector)

    def test_unknown_projector_type_raises(self) -> None:
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

    def test_audio_token_is_in_tokenizer(self, base_asr_model: ASRModel) -> None:
        audio_id = base_asr_model.tokenizer.convert_tokens_to_ids("<audio>")
        assert audio_id is not None
        assert audio_id != base_asr_model.tokenizer.unk_token_id

    def test_audio_token_id_attribute(self, base_asr_model: ASRModel) -> None:
        assert base_asr_model.audio_token_id == base_asr_model.tokenizer.convert_tokens_to_ids(
            "<audio>"
        )

    def test_pad_token_set(self, base_asr_model: ASRModel) -> None:
        assert base_asr_model.tokenizer.pad_token is not None
        assert base_asr_model.tokenizer.pad_token_id is not None

    def test_padding_side_is_right(self, base_asr_model: ASRModel) -> None:
        assert base_asr_model.tokenizer.padding_side == "right"

    def test_embedding_resized_to_tokenizer_length(self, base_asr_model: ASRModel) -> None:
        embed = cast(torch.nn.Embedding, base_asr_model.language_model.get_input_embeddings())
        assert embed.num_embeddings >= len(base_asr_model.tokenizer)

    def test_generation_config_eos_synced(self, base_asr_model: ASRModel) -> None:
        eos_ids = generation_settings(base_asr_model).eos_token_id
        assert eos_ids is None or all(e is not None for e in eos_ids)

    def test_generation_config_carries_no_repeat_ngram_size(self, base_asr_model: ASRModel) -> None:
        # `generate()` consults the GenerationConfig only; a value that lives
        # on ASRConfig alone is inert, which is how the loop guard shipped
        # disabled to the Hub.
        no_repeat_ngram_size = generation_settings(base_asr_model).no_repeat_ngram_size
        assert no_repeat_ngram_size == base_asr_model.config.no_repeat_ngram_size
        assert no_repeat_ngram_size == 12


class TestEmbeddings:
    """get_input_embeddings / set_input_embeddings / get_output_embeddings."""

    def test_get_input_embeddings_returns_module(self, base_asr_model: ASRModel) -> None:
        embed = base_asr_model.get_input_embeddings()
        assert isinstance(embed, torch.nn.Module)

    def test_get_output_embeddings_returns_module(self, base_asr_model: ASRModel) -> None:
        out = base_asr_model.get_output_embeddings()
        assert isinstance(out, torch.nn.Module)


class TestEncoderOutputLengths:
    """_compute_encoder_output_lengths and _get_num_audio_tokens use conv formula."""

    def test_compute_encoder_output_lengths_shape(self, base_asr_model: ASRModel) -> None:
        # whisper-tiny defaults: mel_len 3000 -> 1500 after conv
        attention_mask = torch.ones(2, 3000)
        lengths = base_asr_model._compute_encoder_output_lengths(attention_mask)
        assert lengths.shape == (2,)
        assert lengths[0].item() == 1500

    def test_get_num_audio_tokens_matches_projector(self, base_asr_model: ASRModel) -> None:
        attention_mask = torch.ones(1, 3000)
        n = base_asr_model._get_num_audio_tokens(attention_mask)
        assert n > 0
        assert isinstance(n, int)


class TestFeatureExtractor:
    """_create_feature_extractor returns a usable extractor."""

    def test_feature_extractor_attached(self, base_asr_model: ASRModel) -> None:
        assert base_asr_model.feature_extractor is not None
        assert base_asr_model.feature_extractor.sampling_rate == 16000


class TestStateDict:
    """state_dict only contains projector when LM is frozen."""

    def test_state_dict_only_has_projector_keys(self, base_asr_model: ASRModel) -> None:
        sd = base_asr_model.state_dict()
        assert all(k.startswith("projector.") for k in sd)

    def test_state_dict_includes_lm_when_unfrozen(self) -> None:
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

    def test_whisper_branch_loads_encoder_only(self, base_asr_model: ASRModel) -> None:
        # base_asr_model uses whisper-tiny → audio_tower should be Whisper's encoder
        # (not the full WhisperModel)

        assert isinstance(base_asr_model.audio_tower, WhisperEncoder)

    def test_audio_encoder_is_frozen(self, base_asr_model: ASRModel) -> None:
        for p in base_asr_model.audio_tower.parameters():
            assert p.requires_grad is False

    def test_audio_encoder_in_eval_mode(self, base_asr_model: ASRModel) -> None:
        assert base_asr_model.audio_tower.training is False

    @pytest.mark.parametrize("layout", ["nested", "flat"])
    def test_glm_branch_uses_audio_tower(
        self, monkeypatch: pytest.MonkeyPatch, layout: str
    ) -> None:
        """Verify GLM dispatch path without downloading the real GLM model.

        Both submodule layouts must work. transformers 5.x makes
        GlmAsrForConditionalGeneration a wrapper holding a GlmAsrModel at
        `.model`, and that inner model owns audio_tower; older versions hung
        audio_tower off the top-level model.
        """
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

    def test_glm_branch_raises_when_audio_tower_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A future layout change should fail loudly, not hand back a stub."""
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

    def test_lm_frozen_by_default(self, base_asr_model: ASRModel) -> None:
        for p in base_asr_model.language_model.parameters():
            assert p.requires_grad is False

    def test_use_cache_synced(self, base_asr_model: ASRModel) -> None:
        assert base_asr_model.language_model.config.use_cache is True


class TestLoRASetup:
    """_setup_lora wraps language_model with PEFT when use_lora=True."""

    def test_lora_model_has_peft_config(self, lora_asr_model: ASRModel) -> None:
        assert hasattr(lora_asr_model.language_model, "peft_config")

    def test_lora_target_modules_applied(self, lora_asr_model: ASRModel) -> None:
        # PEFT replaces target Linear layers with LoraLayer wrappers

        has_lora = any(isinstance(m, LoraLayer) for m in lora_asr_model.language_model.modules())
        assert has_lora

    def test_non_lora_model_has_no_peft(self, base_asr_model: ASRModel) -> None:
        assert not hasattr(base_asr_model.language_model, "peft_config")


class TestFreezeProjector:
    """freeze_projector=True freezes projector params."""

    def test_freeze_projector_disables_grad(self) -> None:
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

    def test_forward_text_only(self, base_asr_model: ASRModel) -> None:
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

    def test_forward_with_labels_returns_loss(self, base_asr_model: ASRModel) -> None:
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

    def test_generate_requires_input_features(self, base_asr_model: ASRModel) -> None:
        with pytest.raises(ValueError, match="input_features"):
            base_asr_model.generate(input_ids=torch.tensor([[1, 2, 3]]))

    def test_generate_requires_audio_attention_mask(self, base_asr_model: ASRModel) -> None:
        with pytest.raises(ValueError, match="audio_attention_mask"):
            base_asr_model.generate(
                input_features=torch.randn(1, 80, 3000),
                input_ids=torch.tensor([[1, 2, 3]]),
            )

    def test_generate_returns_tokens(self, base_asr_model: ASRModel) -> None:
        """Generate end-to-end with synthetic mel features (whisper-tiny)."""
        # Whisper-tiny expects 80 mel bins, 3000 frames (30s of audio at 16kHz)
        input_features = torch.zeros(1, 80, 3000)
        audio_attention_mask = torch.ones(1, 3000, dtype=torch.long)

        out = cast(
            torch.Tensor,
            base_asr_model.generate(
                input_features=input_features,
                audio_attention_mask=audio_attention_mask,
                max_new_tokens=4,
            ),
        )
        # Returns only generated tokens (input is stripped)
        assert out.dim() == 2
        assert out.shape[0] == 1
        assert out.shape[1] <= 4


class TestGenerateStreaming:
    """generate_streaming yields partial transcript pieces."""

    def test_streaming_yields_strings(self, base_asr_model: ASRModel) -> None:
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

    def test_save_creates_expected_files(self, base_asr_model: ASRModel, tmp_path: Path) -> None:
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

    def test_save_then_load_round_trip(self, base_asr_model: ASRModel, tmp_path: Path) -> None:
        save_dir = tmp_path / "model"
        base_asr_model.save_pretrained(save_dir)

        loaded = ASRModel.from_pretrained(str(save_dir))

        # Check projector weights match
        original_proj = dict(base_asr_model.projector.state_dict())
        loaded_proj = dict(loaded.projector.state_dict())
        assert original_proj.keys() == loaded_proj.keys()
        for k, v in original_proj.items():
            assert torch.allclose(v, loaded_proj[k])

    def test_save_lora_writes_adapter_config(
        self, lora_asr_model: ASRModel, tmp_path: Path
    ) -> None:
        save_dir = tmp_path / "lora_model"
        lora_asr_model.save_pretrained(save_dir)

        # PEFT writes these
        assert (save_dir / "adapter_config.json").exists()
        assert (save_dir / "adapter_model.safetensors").exists()

    def test_save_lora_clears_base_model_path_when_no_repo_id(
        self, lora_asr_model: ASRModel, tmp_path: Path
    ) -> None:
        save_dir = tmp_path / "lora_model"
        lora_asr_model.save_pretrained(save_dir)

        with (save_dir / "adapter_config.json").open() as f:
            adapter_cfg = json.load(f)

        # Should be empty string (not None / "None") when no repo_id is given
        assert adapter_cfg["base_model_name_or_path"] == ""

    def test_save_lora_uses_repo_id_when_provided(
        self, lora_asr_model: ASRModel, tmp_path: Path
    ) -> None:
        save_dir = tmp_path / "lora_model_with_repo"
        lora_asr_model.save_pretrained(save_dir, repo_id="alex/test-model")

        with (save_dir / "adapter_config.json").open() as f:
            adapter_cfg = json.load(f)

        assert adapter_cfg["base_model_name_or_path"] == "alex/test-model"


class TestProcessor:
    """get_processor wires together feature extractor, tokenizer, projector."""

    def test_get_processor_returns_asrprocessor(self, base_asr_model: ASRModel) -> None:
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

    def test_enable_via_upstream_entry_point(self, base_asr_model: ASRModel) -> None:
        base_asr_model.gradient_checkpointing_enable()
        assert base_asr_model.language_model.is_gradient_checkpointing
        base_asr_model.gradient_checkpointing_disable()  # type: ignore[no-untyped-call]  # untyped upstream
        assert not base_asr_model.language_model.is_gradient_checkpointing

    def test_enable_accepts_upstream_kwargs(self, base_asr_model: ASRModel) -> None:
        # Mirrors the exact call transformers.Trainer makes.
        base_asr_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
            every_n_layers=1,
        )
        assert base_asr_model.language_model.is_gradient_checkpointing
        base_asr_model.gradient_checkpointing_disable()  # type: ignore[no-untyped-call]  # untyped upstream

    def test_signature_matches_upstream(self, base_asr_model: ASRModel) -> None:
        upstream = inspect.signature(PreTrainedModel._set_gradient_checkpointing).parameters
        ours = inspect.signature(base_asr_model._set_gradient_checkpointing).parameters
        accepts_var_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in ours.values())
        missing = [n for n in upstream if n != "self" and n not in ours]
        assert accepts_var_kwargs or not missing, f"override missing kwargs: {missing}"

    def test_does_not_expose_legacy_value_param(self, base_asr_model: ASRModel) -> None:
        # Upstream treats a `value` parameter as the pre-4.35 format and routes
        # to a different code path entirely, silently skipping our override.

        ours = inspect.signature(base_asr_model._set_gradient_checkpointing).parameters
        assert "value" not in ours

    def test_frozen_encoder_is_not_checkpointed(self, base_asr_model: ASRModel) -> None:
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

    def test_reads_uniform_head_dim(self) -> None:
        class Cfg:
            head_dim = 128

        assert _max_attention_head_dim(Cfg()) == 128

    def test_takes_max_across_heterogeneous_layers(self) -> None:
        class Layer:
            def __init__(self, d: int) -> None:
                self.head_dim = d

        class Cfg:
            per_layer_config: ClassVar[list[Layer]] = [Layer(256), Layer(512), Layer(256)]

        assert _max_attention_head_dim(Cfg()) == 512

    def test_per_layer_wins_over_ambiguous_global(self) -> None:
        # Heterogeneous configs raise rather than return a number when the
        # global attribute is read, so per_layer_config must be consulted first.

        class Layer:
            def __init__(self, d: int) -> None:
                self.head_dim = d

        class Cfg:
            per_layer_config: ClassVar[list[Layer]] = [Layer(256), Layer(512)]

            @property
            def head_dim(self) -> int:
                msg = "ambiguous per-layer attribute"
                raise RuntimeError(msg)

        assert _max_attention_head_dim(Cfg()) == 512

    def test_unknown_head_dim_is_none(self) -> None:
        # None means "cannot determine" and must not be treated as 0, which
        # would wrongly leave FA2 enabled or wrongly disable it.

        class Cfg:
            pass

        assert _max_attention_head_dim(Cfg()) is None

    def test_guard_threshold_matches_flash_attention(self) -> None:
        assert FLASH_ATTENTION_MAX_HEAD_DIM == 256


class TestGemmaDecodeLoopPatch:
    """Gemma 4's causal LM needs two things patched for audio generation to
    work at all, and both failed in ways that only showed up at inference:
    `per_layer_inputs` must be dropped after the first decode step, and the
    wrapper must not hide `inputs_embeds` from signature introspection."""

    def test_keeps_per_layer_inputs_on_first_step(self) -> None:
        stub = gemma_decode_loop_stub()
        _patch_gemma_decode_loop(stub)
        out = stub.prepare_inputs_for_generation(
            None, inputs_embeds="E", per_layer_inputs="PLE", is_first_iteration=True
        )
        assert out["per_layer_inputs"] == "PLE"

    def test_drops_per_layer_inputs_on_later_steps(self) -> None:
        # Step 2+ passes input_ids, and Gemma's forward raises if PLE comes
        # along with it.

        stub = gemma_decode_loop_stub()
        _patch_gemma_decode_loop(stub)
        out = stub.prepare_inputs_for_generation(
            [[1]], per_layer_inputs="PLE", is_first_iteration=False
        )
        assert "per_layer_inputs" not in out

    def test_wrapper_preserves_inputs_embeds_in_signature(self) -> None:
        # generation._prepare_model_inputs gates inputs_embeds support on this
        # exact introspection; a bare *args wrapper makes generate() reject
        # inputs_embeds, which is the only way audio reaches the decoder.

        stub = gemma_decode_loop_stub()
        _patch_gemma_decode_loop(stub)
        params = set(inspect.signature(stub.prepare_inputs_for_generation).parameters)
        assert "inputs_embeds" in params


class TestEosTokenResolution:
    """generation_config.eos_token_id must hold only real vocab entries."""

    def test_excludes_tokens_absent_from_vocab(self, base_asr_model: ASRModel) -> None:
        # SmolLM2 has no "<end_of_turn>"; convert_tokens_to_ids falls back to
        # unk rather than returning None, which is how Gemma ended up with
        # eos_token_id=[unk, unk] and never stopped generating.
        tok = base_asr_model.tokenizer
        for token_id in eos_ids(base_asr_model):
            assert tok.convert_ids_to_tokens(token_id) != "<end_of_turn>"

    def test_keeps_endoftext_even_though_it_is_also_unk(self, base_asr_model: ASRModel) -> None:
        # SmolLM2's unk_token IS "<|endoftext|>", so filtering on
        # `id == unk_token_id` would drop a legitimate stop token.
        tok = base_asr_model.tokenizer
        endoftext = tok.convert_tokens_to_ids("<|endoftext|>")
        assert endoftext in eos_ids(base_asr_model)

    def test_includes_tokenizer_eos(self, base_asr_model: ASRModel) -> None:
        assert base_asr_model.tokenizer.eos_token_id in eos_ids(base_asr_model)

    def test_is_non_empty(self, base_asr_model: ASRModel) -> None:
        assert eos_ids(base_asr_model)

    def test_includes_the_templates_own_turn_terminator(self, base_asr_model: ASRModel) -> None:
        """The token the chat template appends after assistant content must stop generation.

        This is the invariant the hardcoded name list silently violated on
        Gemma 4, whose template closes turns with "<turn|>" rather than
        "<end_of_turn>". Training supervises that token as the end of the
        transcript, so omitting it from eos_token_id let every sample run to
        max_new_tokens and buried the transcript under repeats of it.
        """
        tok = base_asr_model.tokenizer
        sentinel = "⁣turnendprobe⁣"
        rendered = cast(
            str,
            apply_chat_template(
                tok,
                [
                    {"role": "user", "content": "x"},
                    {"role": "assistant", "content": sentinel},
                ],
                tokenize=False,
                add_generation_prompt=False,
            ),
        )
        tail_ids = tok(rendered.split(sentinel)[-1], add_special_tokens=False)["input_ids"]
        assert tail_ids, "template appends nothing after assistant content"
        assert tail_ids[0] in eos_ids(base_asr_model)

    def test_derives_a_terminator_the_name_probes_do_not_know(
        self, base_asr_model: ASRModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Derivation must find a turn terminator absent from the hardcoded names.

        Stands in for Gemma 4's "<turn|>": a real vocab entry that none of
        "<|im_end|>" / "<|endoftext|>" / "<end_of_turn>" would have matched.
        """
        tok = base_asr_model.tokenizer
        marker = "<|im_start|>"
        monkeypatch.setattr(
            tok,
            "chat_template",
            "{% for m in messages %}{{ m['content'] }}" + marker + "{% endfor %}",
        )
        assert base_asr_model._derive_turn_end_token_id() == tok.convert_tokens_to_ids(marker)

    def test_derivation_ignores_a_plain_text_terminator(
        self, base_asr_model: ASRModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A template ending on ordinary text yields no stop token.

        Adopting a text token as EOS would truncate real transcripts, so the
        derivation must decline and leave the fallbacks in charge.
        """
        monkeypatch.setattr(
            base_asr_model.tokenizer,
            "chat_template",
            "{% for m in messages %}{{ m['content'] }} END OF REPLY{% endfor %}",
        )
        assert base_asr_model._derive_turn_end_token_id() is None


class TestStateDictTrainableModules:
    """state_dict must serialize every module the freeze flags leave trainable."""

    def test_frozen_encoder_is_not_saved(self, base_asr_model: ASRModel) -> None:
        keys = base_asr_model.state_dict()
        assert not any(k.startswith("audio_tower.") for k in keys)
        assert any(k.startswith("projector.") for k in keys)

    def test_unfrozen_encoder_is_saved(self) -> None:
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
        result = cast(LoadStateDictResult, model.load_state_dict(keys, strict=False))
        assert result.unexpected_keys == []

    def test_encoder_keys_cover_trainable_encoder_params(self) -> None:
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

    def test_encoder_updates_survive_save_reload(self, tmp_path: Path) -> None:
        """End-to-end: the checkpoint a training run writes must carry the encoder."""
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


class TestMpsUnsafeParameters:
    """MPS indexes tensor storage with 32-bit offsets.

    A gather into a parameter holding more than INT32_MAX elements wraps around
    and returns the wrong rows with no error. Gemma 4 E2B's 2.35B-element
    per-layer embedding table hits this, and the only symptom is fluent, wrong
    output -- so the oversize has to be detected, never inferred from results.
    """

    def test_flags_a_parameter_past_int32(self) -> None:
        # `meta` gives a parameter with real shape metadata and no allocation,
        # so the test states the 2GB+ case without needing 2GB+.
        module = torch.nn.Module()
        module.big = torch.nn.Parameter(
            torch.empty(MPS_MAX_TENSOR_ELEMENTS + 1, device="meta"), requires_grad=False
        )
        flagged = mps_unsafe_parameters(module)
        assert [name for name, _ in flagged] == ["big"]
        assert flagged[0][1] > MPS_MAX_TENSOR_ELEMENTS

    def test_passes_a_parameter_at_the_limit(self) -> None:
        module = torch.nn.Module()
        module.ok = torch.nn.Parameter(
            torch.empty(MPS_MAX_TENSOR_ELEMENTS, device="meta"), requires_grad=False
        )
        assert mps_unsafe_parameters(module) == []

    def test_ordinary_model_is_safe(self, base_asr_model: ASRModel) -> None:
        assert mps_unsafe_parameters(base_asr_model) == []


class TestChunkedEmbedding:
    """Splitting an oversized table must not change what it returns.

    The MPS fix is a workaround, so the bar is exactness: chunked lookup has to
    equal the single-tensor lookup element for element on every backend, or it
    trades a loud bug for a quiet one.
    """

    def _forced(self, monkeypatch: pytest.MonkeyPatch, limit: int) -> ModuleType:
        """Lower the element cap so a test-sized table counts as oversized."""
        monkeypatch.setattr(layers_mod, "MPS_MAX_TENSOR_ELEMENTS", limit)
        return layers_mod

    def test_matches_the_unchunked_lookup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = self._forced(monkeypatch, 64)
        torch.manual_seed(0)
        emb = torch.nn.Embedding(50, 8)
        ids = torch.tensor([[0, 7, 49, 23]])
        expected = emb(ids)

        chunked = mod.ChunkedEmbedding(emb)
        assert len(chunked.chunks) > 1, "table should have been split"
        assert torch.equal(chunked(ids), expected)

    def test_preserves_the_gemma_embed_scale(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = self._forced(monkeypatch, 64)
        torch.manual_seed(0)
        emb = torch.nn.Embedding(50, 8)
        # Gemma4TextScaledWordEmbedding multiplies the lookup by embed_scale;
        # dropping it would shrink every per-layer embedding by sqrt(dim).
        emb.embed_scale = torch.tensor(16.0)
        ids = torch.tensor([[1, 2, 3]])
        expected = torch.nn.functional.embedding(ids, emb.weight) * 16.0
        assert torch.equal(mod.ChunkedEmbedding(emb)(ids), expected)

    def test_chunk_oversized_embeddings_reports_and_is_idempotent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = self._forced(monkeypatch, 64)
        root = torch.nn.Module()
        root.inner = torch.nn.Module()
        root.inner.table = torch.nn.Embedding(50, 8)

        assert mod.chunk_oversized_embeddings(root) == ["inner.table"]
        assert isinstance(root.inner.table, mod.ChunkedEmbedding)
        # Second pass finds nothing: ChunkedEmbedding is not an nn.Embedding.
        assert mod.chunk_oversized_embeddings(root) == []

    def test_leaves_small_tables_alone(self) -> None:
        root = torch.nn.Module()
        root.table = torch.nn.Embedding(10, 4)
        assert chunk_oversized_embeddings(root) == []
        assert isinstance(root.table, torch.nn.Embedding)


class TestAttnImplementationOnMps:
    """MPS sdpa miscomputes cached single-token decode against a sliding-window mask.

    Gemma 4 is exactly that layout, and a full-sequence forward hides it -- only
    generation is wrong. Eager is the correct kernel there.
    """

    def test_prefers_eager_when_mps_is_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        for requested in (None, "sdpa", "flash_attention_2"):
            assert resolve_attn_implementation(requested) == "eager"

    def test_respects_an_explicit_eager_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert resolve_attn_implementation("eager") == "eager"

    def test_leaves_non_mps_resolution_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert resolve_attn_implementation("sdpa") == "sdpa"
        assert resolve_attn_implementation("flash_attention_2") == "sdpa"


class TestForwardPassesReturnDict:
    """forward() must hand the decoder an explicit `return_dict`.

    liger's patched forwards do `kwargs.pop("return_dict", None)` and fall back
    to `self.config.use_return_dict` when it is absent, which transformers 5.x
    has deprecated and warns about on every patched run. Supplying the value
    keeps that fallback from running, and every consumer here already reads the
    output by attribute, so True is the only correct value.
    """

    def test_return_dict_true_reaches_the_language_model(self, base_asr_model: ASRModel) -> None:
        seen: dict[str, object] = {}
        real = base_asr_model.language_model

        class Spy(torch.nn.Module):
            def forward(self, *args: Any, **kwargs: Any) -> Any:
                seen.update(kwargs)
                return real(*args, **kwargs)

            def __getattr__(self, name: str) -> Any:
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(real, name)

        spy = Spy()
        ids = base_asr_model.tokenizer("hi <audio> there", return_tensors="pt")["input_ids"]
        base_asr_model.language_model = cast("GenerativeDecoder", spy)
        try:
            base_asr_model(
                input_ids=ids,
                input_features=torch.randn(1, 80, 3000),
                audio_attention_mask=torch.ones(1, 3000),
            )
        finally:
            base_asr_model.language_model = real

        assert seen.get("return_dict") is True

    def test_caller_can_override_return_dict(self, base_asr_model: ASRModel) -> None:
        """setdefault, not a hard set -- an explicit caller value must win."""
        src = inspect.getsource(type(base_asr_model).forward)
        assert 'kwargs.setdefault("return_dict", True)' in src


class TestVocabPadding:
    """The embedding table must stay tensor-core aligned after the audio token.

    Qwen ships tables already padded to a multiple of 128 and leaves the spare
    rows unaddressable (248,320 allocated vs 248,077 addressable on Qwen3.5-2B).
    Resizing to len(tokenizer) alone shrinks past that padding onto a dimension
    that is not even a multiple of 8, which lands the model's largest GEMM --
    lm_head at V~248k -- on an unaligned shape.
    """

    def test_embedding_rows_are_aligned(self, base_asr_model: ASRModel) -> None:
        rows = input_embedding_rows(base_asr_model)
        assert rows % VOCAB_PAD_MULTIPLE == 0

    def test_table_still_covers_every_token(self, base_asr_model: ASRModel) -> None:
        """Padding must grow the table, never truncate below the tokenizer."""
        rows = input_embedding_rows(base_asr_model)
        assert rows >= len(base_asr_model.tokenizer)
        assert base_asr_model.audio_token_id < rows

    def test_output_head_and_config_agree(self, base_asr_model: ASRModel) -> None:
        """lm_head and config.vocab_size must track the padded size.

        They are read independently at generate time and by save_pretrained, so
        a mismatch surfaces as a shape error at checkpoint save rather than here.
        """
        rows = input_embedding_rows(base_asr_model)
        output_embeddings = cast(
            torch.nn.Linear,
            base_asr_model.language_model.get_output_embeddings(),  # type: ignore[no-untyped-call]  # untyped upstream
        )
        assert output_embeddings.weight.shape[0] == rows
        assert base_asr_model.language_model.config.vocab_size == rows


class TestAssertProjectorLoaded:
    """Tests for the projector key guard on from_pretrained."""

    @staticmethod
    def _keys(missing: Sequence[str] = (), unexpected: Sequence[str] = ()) -> _IncompatibleKeys:
        return _IncompatibleKeys(list(missing), list(unexpected))

    def test_accepts_frozen_module_keys(self) -> None:
        """Encoder/decoder keys are never saved, so their absence is expected."""
        _assert_projector_loaded(
            self._keys(
                missing=["audio_tower.encoder.layers.0.weight", "language_model.norm.weight"]
            ),
            "mlp",
        )

    def test_raises_on_missing_projector_key(self) -> None:
        """A projector left at random init must not load silently."""
        with pytest.raises(RuntimeError, match=r"projector\.linear_1\.weight"):
            _assert_projector_loaded(self._keys(missing=["projector.linear_1.weight"]), "mlp")

    def test_raises_on_unexpected_projector_key(self) -> None:
        with pytest.raises(RuntimeError, match=r"projector\.stale\.weight"):
            _assert_projector_loaded(self._keys(unexpected=["projector.stale.weight"]), "mlp")

    def test_legacy_layout_gets_an_actionable_hint(self) -> None:
        """norm_2 in the checkpoint means it predates the layout change."""
        with pytest.raises(RuntimeError, match="predates the MLP projector layout change"):
            _assert_projector_loaded(
                self._keys(
                    missing=["projector.input_norm.weight"],
                    unexpected=["projector.norm.weight", "projector.norm_2.weight"],
                ),
                "mlp",
            )


class TestFusedCrossEntropyRouting:
    """`skip_logits` must reach the decoder on TRAIN steps, not just eval.

    liger's own default is `skip_logits = self.training and labels is not
    None`, read off the language model. `ASRModel.train` forces the decoder
    into eval mode whenever it is frozen, so under `freeze_language_model` (or
    LoRA) that default is False mid-training and the (batch, seq, vocab)
    logits tensor comes back — 12.37 GiB at the granite_qwen_lora geometry,
    which is an OOM in backward rather than a slowdown.
    """

    @staticmethod
    def _record_skip_logits(model: ASRModel, monkeypatch: pytest.MonkeyPatch) -> list[object]:
        """Report `skip_logits` as the decoder would see it, per call."""
        seen: list[object] = []
        inner = model.language_model.forward

        def recording_forward(*args: Any, **kwargs: Any) -> Any:
            seen.append(kwargs.pop("skip_logits", None))
            return inner(*args, **kwargs)

        monkeypatch.setattr(model.language_model, "forward", recording_forward)
        # The stub decoder stands in for liger's patched `lce_forward`, which
        # is the only forward that declares the parameter.
        monkeypatch.setattr(model, "_lm_accepts_skip_logits", True)
        return seen

    @pytest.mark.parametrize("training", [True, False])
    def test_skip_logits_requested_whenever_labels_are_present(
        self, base_asr_model: ASRModel, monkeypatch: pytest.MonkeyPatch, training: bool
    ) -> None:
        seen = self._record_skip_logits(base_asr_model, monkeypatch)
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])

        was_training = base_asr_model.training
        try:
            base_asr_model.train(training)
            base_asr_model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                labels=input_ids.clone(),
            )
        finally:
            base_asr_model.train(was_training)

        assert seen == [True]

    def test_unlabelled_forward_keeps_its_logits(
        self, base_asr_model: ASRModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """generate() and friends run without labels and do need the logits."""
        seen = self._record_skip_logits(base_asr_model, monkeypatch)
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])

        with torch.no_grad():
            out = base_asr_model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

        assert seen == [None]
        assert out.logits is not None

    def test_explicit_skip_logits_is_not_overridden(
        self, base_asr_model: ASRModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A caller that asks for logits with labels still gets them."""
        seen = self._record_skip_logits(base_asr_model, monkeypatch)
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])

        with torch.no_grad():
            base_asr_model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                labels=input_ids.clone(),
                skip_logits=False,
            )

        assert seen == [False]
