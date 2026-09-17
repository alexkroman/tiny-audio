"""Weight-decay / LR routing in ASRTrainer.create_optimizer.

The three-way component split (encoder / decoder / projector) crossed with the
decay / no-decay split had no coverage, so a regression in either axis was
silent. The embedding case in particular is invisible at runtime: decaying a
tied embedding table does not error, it just slowly degrades rare-token
prediction over tens of thousands of steps.
"""

import pytest


@pytest.fixture(scope="module")
def joint_asr_model():
    """ASRModel with the decoder unfrozen.

    The session-scoped `base_asr_model` fixture leaves `freeze_language_model`
    at its True default, which keeps every `language_model.*` parameter out of
    the optimizer entirely — there would be nothing to assert about decoder
    weight-decay routing.
    """
    from tiny_audio.asr_config import ASRConfig
    from tiny_audio.asr_modeling import ASRModel

    config = ASRConfig(
        audio_model_id="openai/whisper-tiny",
        text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        projector_type="mlp",
        model_dtype="float32",
        attn_implementation="eager",
        freeze_language_model=False,
    )
    return ASRModel(config)


@pytest.fixture(scope="module")
def optimizer(joint_asr_model, tmp_path_factory):
    from transformers import TrainingArguments

    from scripts.train import ASRTrainer

    args = TrainingArguments(
        output_dir=str(tmp_path_factory.mktemp("optimizer_groups")),
        learning_rate=1e-3,
        weight_decay=0.01,
        report_to=[],
    )
    trainer = ASRTrainer(
        model=joint_asr_model,
        args=args,
        decoder_learning_rate=2e-5,
        projector_weight_decay=0.0,
    )
    return trainer.create_optimizer()


def _group_of(optimizer, tensor):
    """The param group holding `tensor`, matched by identity.

    Identity matching matters: under tie_word_embeddings the embedding table
    and lm_head are the same object, so an equality- or name-based lookup
    would be ambiguous about which one it found.
    """
    for group in optimizer.param_groups:
        if any(p is tensor for p in group["params"]):
            return group
    raise AssertionError("parameter is not in any optimizer group")


def _named_param(model, predicate):
    for name, param in model.named_parameters():
        if param.requires_grad and predicate(name):
            return name, param
    raise AssertionError("no trainable parameter matched the predicate")


class TestEmbeddingWeightDecay:
    """Embedding tables must land in the no-decay group.

    Under narrow ASR fine-tuning most of the vocab never appears in a batch,
    so those rows get no task gradient and weight decay becomes the only force
    on them — they shrink monotonically toward zero. With tie_word_embeddings
    that tensor also backs lm_head, so the damage reaches the output
    projection.
    """

    def test_embedding_is_tied_in_this_fixture(self, joint_asr_model):
        """Guard: if tying ever breaks, the test below stops being meaningful."""
        lm = joint_asr_model.language_model
        assert lm.get_input_embeddings().weight is lm.get_output_embeddings().weight

    def test_embed_tokens_is_not_decayed(self, joint_asr_model, optimizer):
        embed = joint_asr_model.language_model.get_input_embeddings().weight
        assert _group_of(optimizer, embed)["weight_decay"] == 0.0

    def test_embed_tokens_still_gets_the_decoder_lr(self, joint_asr_model, optimizer):
        """No-decay must not cost the embedding its component LR routing."""
        embed = joint_asr_model.language_model.get_input_embeddings().weight
        assert _group_of(optimizer, embed)["lr"] == pytest.approx(2e-5)

    def test_exclusion_survives_output_head_traversal_order(self, joint_asr_model, optimizer):
        """The tied tensor is excluded whichever name named_parameters() yields.

        get_parameter_names walks the module tree and emits both
        `...embed_tokens.weight` and `...lm_head.weight` for one tensor, while
        named_parameters() deduplicates to whichever the traversal reaches
        first. A name-based exclusion would work on decoders that register the
        input table first and silently fail on any that register the head
        first; identity matching is order-invariant.
        """
        lm = joint_asr_model.language_model
        for tensor in (lm.get_input_embeddings().weight, lm.get_output_embeddings().weight):
            assert _group_of(optimizer, tensor)["weight_decay"] == 0.0


class TestDecayGroupStillPopulated:
    """The exclusions must stay narrow — ordinary matmul weights still decay."""

    def test_decoder_linear_weights_are_decayed(self, joint_asr_model, optimizer):
        _, param = _named_param(
            joint_asr_model,
            lambda n: n.startswith("language_model.") and n.endswith("q_proj.weight"),
        )
        group = _group_of(optimizer, param)
        assert group["weight_decay"] == pytest.approx(0.01)
        assert group["lr"] == pytest.approx(2e-5)

    def test_norm_gains_are_not_decayed(self, joint_asr_model, optimizer):
        _, param = _named_param(
            joint_asr_model,
            lambda n: n.startswith("language_model.") and "layernorm" in n.lower(),
        )
        assert _group_of(optimizer, param)["weight_decay"] == 0.0


class TestProjectorRouting:
    """Projector keeps its own weight decay and the base LR."""

    def test_projector_uses_override_weight_decay_and_base_lr(self, joint_asr_model, optimizer):
        _, param = _named_param(
            joint_asr_model,
            lambda n: n.startswith("projector.") and n.endswith("weight"),
        )
        group = _group_of(optimizer, param)
        assert group["weight_decay"] == 0.0
        assert group["lr"] == pytest.approx(1e-3)

    def test_frozen_encoder_contributes_no_groups(self, joint_asr_model, optimizer):
        """freeze_audio_encoder defaults True, so audio_tower never enters."""
        in_optimizer = {id(p) for g in optimizer.param_groups for p in g["params"]}
        encoder_params = [
            p for n, p in joint_asr_model.named_parameters() if n.startswith("audio_tower.")
        ]
        assert encoder_params, "fixture should have an audio tower"
        assert not any(id(p) in in_optimizer for p in encoder_params)
