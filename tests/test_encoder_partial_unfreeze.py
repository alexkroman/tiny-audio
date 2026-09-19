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
