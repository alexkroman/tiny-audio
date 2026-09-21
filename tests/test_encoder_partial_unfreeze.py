"""Tests for partial encoder unfreezing (`encoder_trainable_top_layers`).

The whole point of a partial unfreeze is that it touches EXACTLY the intended
blocks. Getting it wrong is silent and expensive in both directions: unfreeze
too little and the run is the frozen baseline wearing a different config;
unfreeze too much and you damage an encoder IBM trained on ~60k hours using
7.6k hours of our data.

So these tests assert on the identity of the unfrozen parameters, not on a
count, and the final test proves the freeze holds under autograd rather than
just checking a flag.
"""

import pytest
import torch
from torch import nn

from tiny_audio.asr_modeling import find_encoder_layer_stack, unfreeze_encoder_top_layers


class FakeBlock(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):
        return self.lin(x)


class FakeGraniteEncoder(nn.Module):
    """Mirrors GraniteSpeech5Encoder's children: input_linear, layers, out, out_mid."""

    def __init__(self, depth=16, dim=8):
        super().__init__()
        self.input_linear = nn.Linear(dim, dim)
        self.layers = nn.ModuleList(FakeBlock(dim) for _ in range(depth))
        self.out = nn.Linear(dim, dim)
        self.out_mid = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.input_linear(x)
        for layer in self.layers:
            x = layer(x)
        return self.out(self.out_mid(x))


@pytest.fixture
def frozen_encoder():
    enc = FakeGraniteEncoder()
    enc.requires_grad_(False)
    return enc


class TestFindLayerStack:
    def test_finds_granite_style_layers(self, frozen_encoder):
        path, stack = find_encoder_layer_stack(frozen_encoder)
        assert path == "layers"
        assert len(stack) == 16

    def test_finds_nested_whisper_style(self):
        class Nested(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Module()
                self.encoder.layers = nn.ModuleList(FakeBlock() for _ in range(4))

        path, stack = find_encoder_layer_stack(Nested())
        assert path == "encoder.layers"
        assert len(stack) == 4

    def test_returns_none_when_absent(self):
        assert find_encoder_layer_stack(nn.Linear(4, 4)) is None


class TestUnfreezeSelectsTheRightBlocks:
    @pytest.mark.parametrize("top_n", [1, 2, 4, 6, 16])
    def test_exactly_the_top_n_blocks_are_trainable(self, frozen_encoder, top_n):
        unfreeze_encoder_top_layers(frozen_encoder, top_n)
        depth = len(frozen_encoder.layers)
        for i, block in enumerate(frozen_encoder.layers):
            should_train = i >= depth - top_n
            for name, p in block.named_parameters():
                assert p.requires_grad is should_train, (
                    f"layers.{i}.{name}: expected requires_grad={should_train} "
                    f"for top_n={top_n} of depth={depth}"
                )

    def test_it_is_the_top_blocks_not_the_bottom(self, frozen_encoder):
        """Guards the most plausible off-by-one: slicing [:n] instead of [-n:]."""
        unfreeze_encoder_top_layers(frozen_encoder, 4)
        trainable = [i for i, b in enumerate(frozen_encoder.layers) if b.lin.weight.requires_grad]
        assert trainable == [12, 13, 14, 15]

    def test_output_projections_are_unfrozen(self, frozen_encoder):
        """`out`/`out_mid` sit between the top block and the projector; leaving
        them frozen would force the new blocks to adapt through a fixed map."""
        unfreeze_encoder_top_layers(frozen_encoder, 2)
        assert frozen_encoder.out.weight.requires_grad is True
        assert frozen_encoder.out_mid.weight.requires_grad is True

    def test_input_projection_stays_frozen(self, frozen_encoder):
        """The feature front-end is the part most expensive to damage."""
        unfreeze_encoder_top_layers(frozen_encoder, 2)
        assert frozen_encoder.input_linear.weight.requires_grad is False

    def test_returns_the_names_it_changed(self, frozen_encoder):
        names = unfreeze_encoder_top_layers(frozen_encoder, 2)
        assert "layers.14.lin.weight" in names
        assert "layers.15.lin.weight" in names
        assert "out.weight" in names
        assert not any(n.startswith("layers.13") for n in names)
        assert not any(n.startswith("input_linear") for n in names)

    def test_zero_is_a_noop(self, frozen_encoder):
        assert unfreeze_encoder_top_layers(frozen_encoder, 0) == []
        assert not any(p.requires_grad for p in frozen_encoder.parameters())


class TestFailsLoudly:
    def test_too_many_layers_raises(self, frozen_encoder):
        with pytest.raises(ValueError, match="exceeds the encoder's 16 blocks"):
            unfreeze_encoder_top_layers(frozen_encoder, 17)

    def test_unrecognised_encoder_raises(self):
        """Silently unfreezing nothing would make a frozen baseline masquerade
        as a partial-unfreeze experiment."""
        with pytest.raises(ValueError, match="Could not locate"):
            unfreeze_encoder_top_layers(nn.Linear(4, 4), 2)


class TestGradientsActuallyFlowWhereIntended:
    """The decisive test: requires_grad is a flag, autograd is the truth."""

    def test_grads_land_only_on_unfrozen_blocks(self, frozen_encoder):
        unfreeze_encoder_top_layers(frozen_encoder, 3)
        out = frozen_encoder(torch.randn(2, 8))
        out.sum().backward()

        depth = len(frozen_encoder.layers)
        for i, block in enumerate(frozen_encoder.layers):
            has_grad = block.lin.weight.grad is not None
            expected = i >= depth - 3
            assert has_grad is expected, f"layers.{i}: grad present={has_grad}, expected={expected}"

        assert frozen_encoder.out.weight.grad is not None
        assert frozen_encoder.input_linear.weight.grad is None

    def test_frozen_blocks_do_not_change_under_an_optimizer_step(self, frozen_encoder):
        unfreeze_encoder_top_layers(frozen_encoder, 2)
        before_frozen = frozen_encoder.layers[0].lin.weight.detach().clone()
        before_live = frozen_encoder.layers[15].lin.weight.detach().clone()

        opt = torch.optim.SGD([p for p in frozen_encoder.parameters() if p.requires_grad], lr=0.1)
        frozen_encoder(torch.randn(4, 8)).sum().backward()
        opt.step()

        assert torch.equal(frozen_encoder.layers[0].lin.weight, before_frozen)
        assert not torch.equal(frozen_encoder.layers[15].lin.weight, before_live)


class TestGradientsSurviveASRModelForward:
    """The bug the toy-encoder tests above cannot see.

    Everything else in this file exercises `unfreeze_encoder_top_layers`
    against a synthetic module. That verifies the helper sets `requires_grad`
    correctly, and it always did -- but `ASRModel.forward` separately decided
    whether to run the encoder under `torch.no_grad()`, and it decided by
    reading `config.freeze_audio_encoder`. The documented partial-unfreeze
    recipe leaves that flag True (see ASRConfig: "Only meaningful with
    `freeze_audio_encoder: true` -- this then selectively re-enables the top
    N"), so the top blocks were unfrozen, entered the optimizer, and received
    exactly zero gradient. Silent: no error, no warning, just a no-op
    experiment paying full AdamW state.
    """

    def test_top_blocks_get_gradient_through_the_real_forward(self, base_asr_config):
        import copy

        from tiny_audio.asr_modeling import ASRModel

        config = copy.deepcopy(base_asr_config)
        config.freeze_audio_encoder = True
        config.encoder_trainable_top_layers = 1
        model = ASRModel(config)
        model.train()

        trainable = [p for p in model.audio_tower.parameters() if p.requires_grad]
        assert trainable, "fixture encoder exposed no trainable params to test"

        mel_bins = model.audio_tower.config.num_mel_bins
        input_ids = torch.tensor([[model.audio_token_id, 1, 2]])
        model(
            input_features=torch.randn(1, mel_bins, 3000),
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=input_ids.clone(),
        ).loss.backward()

        assert all(p.grad is not None for p in trainable), (
            "unfrozen encoder params got no gradient -- ASRModel.forward is "
            "running the encoder under no_grad despite requires_grad=True"
        )
        assert any(p.grad.abs().sum() > 0 for p in trainable)

    def test_fully_frozen_encoder_still_gets_no_gradient(self, base_asr_config):
        import copy

        from tiny_audio.asr_modeling import ASRModel

        config = copy.deepcopy(base_asr_config)
        config.freeze_audio_encoder = True
        config.encoder_trainable_top_layers = 0
        model = ASRModel(config)
        model.train()

        input_ids = torch.tensor([[model.audio_token_id, 1, 2]])
        model(
            input_features=torch.randn(1, model.audio_tower.config.num_mel_bins, 3000),
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=input_ids.clone(),
        ).loss.backward()

        assert all(p.grad is None for p in model.audio_tower.parameters())


class TestPartialUnfreezeIsUsableEndToEnd:
    """The two things a top-N unfreeze needs beyond `requires_grad`.

    Both were gated on `config.freeze_audio_encoder`, which the documented
    recipe leaves True, so both silently did the wrong thing: the encoder
    trained in a dtype too coarse to represent its own update, and whatever it
    did learn was dropped at checkpoint time.
    """

    def _config(self, base_asr_config, **overrides):
        import copy

        config = copy.deepcopy(base_asr_config)
        config.freeze_audio_encoder = True
        config.encoder_trainable_top_layers = 1
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    def test_encoder_dtype_overrides_model_dtype(self, base_asr_config):
        from tiny_audio.asr_modeling import ASRModel

        model = ASRModel(
            self._config(base_asr_config, model_dtype="bfloat16", encoder_dtype="float32")
        )
        assert {p.dtype for p in model.audio_tower.parameters()} == {torch.float32}
        assert {p.dtype for p in model.language_model.parameters()} == {torch.bfloat16}

    def test_encoder_dtype_defaults_to_model_dtype(self, base_asr_config):
        from tiny_audio.asr_config import ASRConfig

        assert ASRConfig(model_dtype="bfloat16").encoder_dtype == "bfloat16"
        assert ASRConfig(model_dtype="float32").encoder_dtype == "float32"

    def test_state_dict_persists_exactly_the_trainable_encoder_tensors(self, base_asr_config):
        from tiny_audio.asr_modeling import ASRModel

        model = ASRModel(self._config(base_asr_config))
        trainable = sorted(n for n, p in model.audio_tower.named_parameters() if p.requires_grad)
        assert trainable, "fixture encoder exposed no trainable params"

        saved = sorted(
            k.removeprefix("audio_tower.")
            for k in model.state_dict()
            if k.startswith("audio_tower.")
        )
        assert saved == trainable

    def test_fully_frozen_encoder_is_still_absent_from_state_dict(self, base_asr_config):
        from tiny_audio.asr_modeling import ASRModel

        model = ASRModel(self._config(base_asr_config, encoder_trainable_top_layers=0))
        assert not [k for k in model.state_dict() if k.startswith("audio_tower.")]

    def test_gradient_checkpointing_includes_a_partially_unfrozen_encoder(self, base_asr_config):
        from tiny_audio.asr_modeling import ASRModel

        model = ASRModel(self._config(base_asr_config))
        assert model.audio_tower in model._gradient_checkpointing_targets()

        frozen = ASRModel(self._config(base_asr_config, encoder_trainable_top_layers=0))
        assert frozen.audio_tower not in frozen._gradient_checkpointing_targets()


class TestPostProjectionOptOut:
    """`out`/`out_mid` are 23% of a top-4 budget for 0.27% of its gradient."""

    def test_post_projections_excluded_when_opted_out(self, frozen_encoder):
        unfreeze_encoder_top_layers(frozen_encoder, 2, include_post_projections=False)
        assert frozen_encoder.out.weight.requires_grad is False
        assert frozen_encoder.out_mid.weight.requires_grad is False
        assert frozen_encoder.layers[15].lin.weight.requires_grad is True
        assert frozen_encoder.input_linear.weight.requires_grad is False

    def test_opting_out_only_drops_the_post_projections(self, frozen_encoder):
        with_post = set(unfreeze_encoder_top_layers(frozen_encoder, 2))
        for p in frozen_encoder.parameters():
            p.requires_grad_(False)
        without = set(
            unfreeze_encoder_top_layers(frozen_encoder, 2, include_post_projections=False)
        )
        assert without < with_post
        assert all(n.startswith(("out.", "out_mid.")) for n in with_post - without)

    def test_default_still_includes_them(self, frozen_encoder):
        unfreeze_encoder_top_layers(frozen_encoder, 2)
        assert frozen_encoder.out.weight.requires_grad is True

    def test_config_flag_reaches_the_model(self, base_asr_config):
        import copy

        from tiny_audio.asr_modeling import ASRModel

        config = copy.deepcopy(base_asr_config)
        config.freeze_audio_encoder = True
        config.encoder_trainable_top_layers = 1
        config.encoder_trainable_post_projections = False
        model = ASRModel(config)
        without = {n for n, p in model.audio_tower.named_parameters() if p.requires_grad}
        assert without, "expected the top block to be trainable"
        assert all(n.startswith("layers.") for n in without), sorted(without)[:5]

        config.encoder_trainable_post_projections = True
        with_post = {
            n for n, p in ASRModel(config).audio_tower.named_parameters() if p.requires_grad
        }
        assert without < with_post


class TestEncoderIsNeverLeftInTrainMode:
    """Construction must not decide train/eval mode.

    `_load_audio_encoder` used to call `encoder.train(True)` after a partial
    unfreeze. Any consumer that then called neither `train()` nor `eval()`
    inherited train mode -- and Granite's encoder carries 16 BatchNorm1d
    modules, which in train mode normalise with per-utterance batch statistics
    instead of the pretrained running statistics. Measured cost on a real
    checkpoint: Earnings22 WER 12.27% -> 37.44%.
    """

    def _model(self, base_asr_config, top_n):
        import copy

        from tiny_audio.asr_modeling import ASRModel

        config = copy.deepcopy(base_asr_config)
        config.freeze_audio_encoder = True
        config.encoder_trainable_top_layers = top_n
        return ASRModel(config)

    @pytest.mark.parametrize("top_n", [0, 1])
    def test_encoder_is_in_eval_mode_straight_after_construction(self, base_asr_config, top_n):
        model = self._model(base_asr_config, top_n)
        assert model.audio_tower.training is False
        assert not any(m.training for m in model.audio_tower.modules())

    def test_partial_unfreeze_still_leaves_params_trainable(self, base_asr_config):
        model = self._model(base_asr_config, 1)
        assert any(p.requires_grad for p in model.audio_tower.parameters())

    def test_model_train_does_not_wake_the_encoder(self, base_asr_config):
        """Frozen BatchNorm statistics are the point -- see _load_audio_encoder."""
        model = self._model(base_asr_config, 1)
        model.train()
        assert model.audio_tower.training is False
        model.eval()
        assert model.audio_tower.training is False


class TestFullyUnfrozenEncoderStillPinsBatchNorm:
    """A fully trainable encoder must still normalise on pretrained BN stats.

    `freeze_audio_encoder: false` (granite_qwen_full) is the one path that
    leaves the encoder in train mode, and Granite Speech 5.0 carries 16
    BatchNorm1d modules. That matters more than the usual small-batch worry,
    because the failure is bias rather than variance: for non-Whisper feature
    extractors the collator pads to the batch longest, Granite's convolution
    module zeroes the pad positions and THEN runs BatchNorm1d over the full
    padded (B, C, T), and BN reduces over B*T -- so the pad frames are counted
    as data at any batch size. At BN's default momentum=0.1 the running
    statistics converge to those polluted values within ~30-50 steps, and
    every later eval and checkpoint uses them.

    The channel is also invisible to `encoder_learning_rate`: running stats
    are buffers, so they carry no gradient and sit in no optimizer group.
    Measured cost of wrong BN statistics on this encoder: Earnings22 WER
    12.27% -> 37.44%.
    """

    def _model(self, base_asr_config):
        import copy

        from tiny_audio.asr_modeling import ASRModel

        config = copy.deepcopy(base_asr_config)
        config.freeze_audio_encoder = False
        model = ASRModel(config)
        # whisper-tiny has no BatchNorm of its own; add one so the branch is
        # actually exercised on a real ASRModel rather than a stand-in.
        model.audio_tower.add_module("probe_bn", nn.BatchNorm1d(4))
        return model

    def test_batchnorm_stays_in_eval_mode_under_train(self, base_asr_config):
        model = self._model(base_asr_config)
        model.train()
        assert model.audio_tower.probe_bn.training is False

    def test_the_rest_of_the_encoder_does_enter_train_mode(self, base_asr_config):
        """Only the BN statistics are pinned -- this is not a backdoor freeze."""
        model = self._model(base_asr_config)
        model.train()
        assert model.audio_tower.training is True
        assert model.audio_tower.conv1.training is True

    def test_batchnorm_affine_params_remain_trainable(self, base_asr_config):
        model = self._model(base_asr_config)
        model.train()
        bn = model.audio_tower.probe_bn
        assert bn.weight.requires_grad
        assert bn.bias.requires_grad

    def test_pinned_batchnorm_does_not_update_running_stats(self, base_asr_config):
        model = self._model(base_asr_config)
        model.train()
        bn = model.audio_tower.probe_bn
        before = bn.running_mean.clone()
        # Heavily off-centre input: a train-mode BN would move running_mean.
        bn(torch.randn(8, 4) + 10.0)
        assert torch.equal(bn.running_mean, before)

    def test_pinned_batchnorm_still_passes_gradient(self, base_asr_config):
        model = self._model(base_asr_config)
        model.train()
        bn = model.audio_tower.probe_bn
        x = torch.randn(8, 4, requires_grad=True)
        bn(x).sum().backward()
        assert x.grad is not None
        assert bn.weight.grad is not None

    def test_frozen_encoder_path_is_unchanged(self, base_asr_config):
        """The `freeze_audio_encoder: true` branch must be untouched."""
        import copy

        from tiny_audio.asr_modeling import ASRModel

        config = copy.deepcopy(base_asr_config)
        config.freeze_audio_encoder = True
        model = ASRModel(config)
        model.train()
        assert model.audio_tower.training is False
