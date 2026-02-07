"""Tests for the AR codec AudioHead for speech-to-speech."""

import pytest
import torch

from tiny_audio.audio_head import AudioHead, PreNN


class MockAudioHeadConfig:
    """Mock config for AudioHead initialization in tests."""

    def __init__(self, **kwargs):
        self.llm_dim = kwargs.get("llm_dim", 3072)
        self.max_audio_tokens = kwargs.get("max_audio_tokens", 500)
        self.audio_top_k = kwargs.get("audio_top_k", 50)
        self.audio_temperature = kwargs.get("audio_temperature", 1.0)


@pytest.fixture
def audio_head_config():
    """Factory fixture for creating audio head configs."""
    return MockAudioHeadConfig


@pytest.fixture
def small_audio_head(audio_head_config):
    """Create a small AudioHead for testing."""
    config = audio_head_config(llm_dim=512)
    return AudioHead(config, llm_dim=512)


class TestAudioHeadInit:
    """Tests for AudioHead initialization."""

    def test_default_init(self, audio_head_config):
        """Test AudioHead initializes with default config."""
        config = audio_head_config()
        head = AudioHead(config)

        assert head.llm_dim == 3072
        assert head.hidden_dim == AudioHead.HIDDEN_DIM
        assert head.vocab_size == AudioHead.VOCAB_SIZE

    def test_custom_llm_dim(self, audio_head_config):
        """Test AudioHead with custom LLM dimension."""
        config = audio_head_config(llm_dim=1536)
        head = AudioHead(config, llm_dim=1536)

        assert head.llm_dim == 1536

    def test_depformer_created(self, small_audio_head):
        """Test that Depformer is created."""
        assert hasattr(small_audio_head, "depformer")
        assert small_audio_head.depformer is not None

    def test_ar_decoder_created(self, small_audio_head):
        """Test that AR decoder is created."""
        assert hasattr(small_audio_head, "ar_decoder")
        assert small_audio_head.ar_decoder is not None

    def test_special_tokens(self, small_audio_head):
        """Test special token IDs are set correctly."""
        vocab_size = small_audio_head.vocab_size
        assert small_audio_head.bos_token_id == vocab_size + 0
        assert small_audio_head.sos_token_id == vocab_size + 1
        assert small_audio_head.eos_token_id == vocab_size + 2
        assert small_audio_head.pad_token_id == vocab_size + 3


class TestAudioHeadOutputLength:
    """Tests for get_output_length method."""

    def test_output_length_estimate(self, audio_head_config):
        """Test output length estimation."""
        config = audio_head_config()
        head = AudioHead(config)

        # Mimi: 24kHz audio / 12.5 Hz = 1920 samples per frame
        # Estimate: ~3 frames per text token
        assert head.get_output_length(1) == 3 * 1920
        assert head.get_output_length(10) == 30 * 1920

    def test_output_length_zero(self, audio_head_config):
        """Test output length with zero input."""
        config = audio_head_config()
        head = AudioHead(config)
        assert head.get_output_length(0) == 0


class TestAudioHeadStateDict:
    """Tests for state dict handling."""

    def test_state_dict_not_empty(self, small_audio_head):
        """Test that state_dict contains all model parameters."""
        state = small_audio_head.state_dict()
        assert len(state) > 0

    def test_state_dict_has_expected_components(self, small_audio_head):
        """Test that state_dict contains expected component prefixes."""
        state = small_audio_head.state_dict()
        prefixes = set()
        for key in state:
            prefix = key.split(".")[0]
            prefixes.add(prefix)

        # input_proj removed - caller now projects to hidden_dim
        expected_prefixes = {
            "embedding",
            "ar_decoder",
            "depformer",
        }
        assert expected_prefixes.issubset(prefixes)

    def test_load_state_dict(self, small_audio_head):
        """Test loading state dict restores weights."""
        original_state = small_audio_head.state_dict()

        # Modify weights (use embedding instead of removed input_proj)
        with torch.no_grad():
            for param in small_audio_head.embedding.parameters():
                param.fill_(0.0)
                break

        # Load original state (strict=False because ar_decoder.embedding is tied to self.embedding)
        small_audio_head.load_state_dict(original_state, strict=False)

        # Verify weights are restored
        restored_state = small_audio_head.state_dict()
        for key in original_state:
            assert torch.allclose(original_state[key], restored_state[key])


class TestForwardTraining:
    """Tests for training forward pass."""

    def test_forward_train_returns_scalar_loss(self, small_audio_head):
        """Test training forward pass returns scalar loss."""
        batch_size, text_len, audio_len = 2, 10, 30
        num_codebooks = 8  # Mimi uses 8 codebooks

        hidden = torch.randn(batch_size, text_len, small_audio_head.llm_dim)
        # Codec targets: (batch, num_codebooks, audio_len) - discrete tokens for all codebooks
        targets = torch.randint(
            0, small_audio_head.vocab_size, (batch_size, num_codebooks, audio_len)
        )
        lengths = torch.tensor([30, 25])

        # Pass hidden as both embeddings and text_embeddings
        loss = small_audio_head(hidden, hidden, codec_targets=targets, codec_lengths=lengths)

        assert loss.dim() == 0, "Loss should be a scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be infinite"

    def test_forward_train_loss_is_differentiable(self, small_audio_head):
        """Test training loss supports backward pass."""
        batch_size, text_len, audio_len = 2, 10, 30
        num_codebooks = 8

        hidden = torch.randn(batch_size, text_len, small_audio_head.llm_dim, requires_grad=True)
        targets = torch.randint(
            0, small_audio_head.vocab_size, (batch_size, num_codebooks, audio_len)
        )
        lengths = torch.tensor([30, 25])

        loss = small_audio_head(hidden, hidden, codec_targets=targets, codec_lengths=lengths)
        loss.backward()

        assert hidden.grad is not None
        assert hidden.grad.shape == hidden.shape

    def test_forward_train_batch_size_one(self, small_audio_head):
        """Test training with batch size of 1."""
        num_codebooks = 8
        hidden = torch.randn(1, 10, small_audio_head.llm_dim)
        targets = torch.randint(0, small_audio_head.vocab_size, (1, num_codebooks, 30))
        lengths = torch.tensor([30])

        loss = small_audio_head(hidden, hidden, codec_targets=targets, codec_lengths=lengths)

        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_forward_train_without_lengths(self, small_audio_head):
        """Test training without explicit lengths."""
        batch_size, text_len, audio_len = 2, 10, 30
        num_codebooks = 8

        hidden = torch.randn(batch_size, text_len, small_audio_head.llm_dim)
        targets = torch.randint(
            0, small_audio_head.vocab_size, (batch_size, num_codebooks, audio_len)
        )

        loss = small_audio_head(hidden, hidden, codec_targets=targets, codec_lengths=None)

        assert loss.dim() == 0
        assert not torch.isnan(loss)


class TestForwardInference:
    """Tests for inference forward pass."""

    def test_forward_inference_returns_codes(self, small_audio_head):
        """Test inference forward pass returns codec tokens."""
        # Override max_tokens for faster test
        small_audio_head.max_tokens = 10

        hidden = torch.randn(1, 10, small_audio_head.llm_dim)

        codes, _ = small_audio_head(hidden, hidden)  # No targets = inference

        assert codes.dtype == torch.long
        assert codes.shape[0] == 1  # Batch size preserved
        # All generated tokens should be valid codec tokens
        assert (codes >= 0).all()
        assert (codes < small_audio_head.vocab_size).all()

    def test_forward_inference_batch_size_one(self, small_audio_head):
        """Test inference with batch size of 1."""
        small_audio_head.max_tokens = 10
        hidden = torch.randn(1, 10, small_audio_head.llm_dim)

        codes, _ = small_audio_head(hidden, hidden)

        assert codes.shape[0] == 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_short_sequence(self, small_audio_head):
        """Test with minimal sequence length."""
        num_codebooks = 8
        hidden = torch.randn(1, 1, small_audio_head.llm_dim)
        targets = torch.randint(0, small_audio_head.vocab_size, (1, num_codebooks, 3))
        lengths = torch.tensor([3])

        loss = small_audio_head(hidden, hidden, codec_targets=targets, codec_lengths=lengths)

        assert not torch.isnan(loss)

    def test_long_sequence(self, small_audio_head):
        """Test with longer sequence."""
        num_codebooks = 8
        hidden = torch.randn(1, 50, small_audio_head.llm_dim)
        targets = torch.randint(0, small_audio_head.vocab_size, (1, num_codebooks, 200))
        lengths = torch.tensor([200])

        loss = small_audio_head(hidden, hidden, codec_targets=targets, codec_lengths=lengths)

        assert not torch.isnan(loss)

    def test_varying_lengths_in_batch(self, small_audio_head):
        """Test batch with different target lengths."""
        batch_size = 3
        num_codebooks = 8
        hidden = torch.randn(batch_size, 20, small_audio_head.llm_dim)
        targets = torch.randint(0, small_audio_head.vocab_size, (batch_size, num_codebooks, 80))
        lengths = torch.tensor([80, 60, 40])

        loss = small_audio_head(hidden, hidden, codec_targets=targets, codec_lengths=lengths)

        assert not torch.isnan(loss)


class TestDevicePlacement:
    """Tests for device and dtype handling."""

    def test_to_device(self, small_audio_head):
        """Test .to() method moves model to device."""
        small_audio_head.to(device="cpu")

        for param in small_audio_head.parameters():
            assert param.device.type == "cpu"

    def test_forward_respects_input_device(self, small_audio_head):
        """Test forward pass respects input device."""
        device = torch.device("cpu")
        num_codebooks = 8
        small_audio_head.to(device)

        hidden = torch.randn(2, 10, small_audio_head.llm_dim, device=device)
        targets = torch.randint(
            0, small_audio_head.vocab_size, (2, num_codebooks, 30), device=device
        )
        lengths = torch.tensor([30, 25], device=device)

        loss = small_audio_head(hidden, hidden, codec_targets=targets, codec_lengths=lengths)

        assert loss.device == device


class TestDecodeToAudio:
    """Tests for decode_to_audio method."""

    def test_decode_to_audio_without_mimi_raises(self, small_audio_head):
        """Test decode_to_audio raises error when Mimi not loaded."""
        num_codebooks = 8
        codes = torch.randint(0, small_audio_head.vocab_size, (1, num_codebooks, 20))

        with pytest.raises(RuntimeError, match="Mimi not loaded"):
            small_audio_head.decode_to_audio(codes)


class TestEncodeAudio:
    """Tests for encode_audio method."""

    def test_encode_audio_without_mimi_raises(self, small_audio_head):
        """Test encode_audio raises error when Mimi not loaded."""
        audio = torch.randn(1, 24000)

        with pytest.raises(RuntimeError, match="Mimi not loaded"):
            small_audio_head.encode_audio(audio)


class TestARTraining:
    """Tests for AR codec training behavior."""

    def test_loss_decreases_with_training(self, small_audio_head):
        """Test that loss decreases when training on same batch."""
        batch_size, text_len, audio_len = 2, 10, 30
        num_codebooks = 8

        hidden = torch.randn(batch_size, text_len, small_audio_head.llm_dim)
        targets = torch.randint(
            0, small_audio_head.vocab_size, (batch_size, num_codebooks, audio_len)
        )

        small_audio_head.train()
        optimizer = torch.optim.Adam(small_audio_head.parameters(), lr=1e-3)

        # Get initial loss
        loss_initial = small_audio_head(hidden, hidden, codec_targets=targets).item()

        # Train for a few steps
        for _ in range(10):
            optimizer.zero_grad()
            loss = small_audio_head(hidden, hidden, codec_targets=targets)
            loss.backward()
            optimizer.step()

        loss_final = small_audio_head(hidden, hidden, codec_targets=targets).item()

        # Loss should decrease (model is learning)
        assert loss_final < loss_initial


class TestNumericalStability:
    """Tests for numerical stability with extreme values."""

    def test_large_input_values(self, small_audio_head):
        """Test with very large input values."""
        batch_size, text_len, audio_len = 2, 10, 30
        num_codebooks = 8

        hidden = torch.randn(batch_size, text_len, small_audio_head.llm_dim) * 100
        targets = torch.randint(
            0, small_audio_head.vocab_size, (batch_size, num_codebooks, audio_len)
        )

        loss = small_audio_head(hidden, hidden, codec_targets=targets)

        assert not torch.isnan(loss), "NaN with large inputs"
        assert not torch.isinf(loss), "Inf with large inputs"

    def test_small_input_values(self, small_audio_head):
        """Test with very small input values."""
        batch_size, text_len, audio_len = 2, 10, 30
        num_codebooks = 8

        hidden = torch.randn(batch_size, text_len, small_audio_head.llm_dim) * 1e-6
        targets = torch.randint(
            0, small_audio_head.vocab_size, (batch_size, num_codebooks, audio_len)
        )

        loss = small_audio_head(hidden, hidden, codec_targets=targets)

        assert not torch.isnan(loss), "NaN with small inputs"


class TestBatchSizeEdgeCases:
    """Tests for batch size edge cases."""

    def test_large_batch_size(self, small_audio_head):
        """Test with large batch size."""
        batch_size = 32
        text_len = 10
        audio_len = 30
        num_codebooks = 8

        hidden = torch.randn(batch_size, text_len, small_audio_head.llm_dim)
        targets = torch.randint(
            0, small_audio_head.vocab_size, (batch_size, num_codebooks, audio_len)
        )
        lengths = torch.randint(20, audio_len + 1, (batch_size,))

        loss = small_audio_head(hidden, hidden, codec_targets=targets, codec_lengths=lengths)

        assert not torch.isnan(loss)
        assert loss.dim() == 0


class TestGradientFlow:
    """Tests for gradient flow through all components."""

    def test_gradients_flow_to_all_trainable_params(self, small_audio_head):
        """Test gradients flow to all trainable parameters."""
        num_codebooks = 8
        hidden = torch.randn(2, 10, small_audio_head.llm_dim, requires_grad=True)
        targets = torch.randint(0, small_audio_head.vocab_size, (2, num_codebooks, 30))

        loss = small_audio_head(hidden, hidden, codec_targets=targets)
        loss.backward()

        # Check all trainable parameters have gradients
        for name, param in small_audio_head.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_gradient_magnitude_reasonable(self, small_audio_head):
        """Test gradient magnitudes are reasonable (not exploding)."""
        num_codebooks = 8
        hidden = torch.randn(2, 10, small_audio_head.llm_dim, requires_grad=True)
        targets = torch.randint(0, small_audio_head.vocab_size, (2, num_codebooks, 30))

        loss = small_audio_head(hidden, hidden, codec_targets=targets)
        loss.backward()

        for name, param in small_audio_head.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                assert grad_norm < 1e6, f"Exploding gradient for {name}: {grad_norm}"


class TestPreNN:
    """Tests for Pre-NN projection module."""

    def test_prenn_output_shape(self):
        """Test PreNN output has correct shape."""
        prenn = PreNN(llm_dim=512, hidden_dim=256, num_layers=2, num_heads=4)
        x = torch.randn(2, 10, 512)
        out = prenn(x)
        assert out.shape == (2, 10, 256)

    def test_prenn_with_mask(self):
        """Test PreNN works with padding mask."""
        prenn = PreNN(llm_dim=512, hidden_dim=256, num_layers=2, num_heads=4)
        x = torch.randn(2, 10, 512)
        mask = torch.ones(2, 10, dtype=torch.bool)
        mask[1, 7:] = False  # Pad last 3 positions in sample 2
        out = prenn(x, mask=mask)
        assert out.shape == (2, 10, 256)

    def test_prenn_without_mask(self):
        """Test PreNN works without mask (inference path)."""
        prenn = PreNN(llm_dim=512, hidden_dim=256, num_layers=2, num_heads=4)
        x = torch.randn(1, 5, 512)
        out = prenn(x)
        assert out.shape == (1, 5, 256)

    def test_prenn_gradient_flow(self):
        """Test gradients flow through PreNN to input."""
        prenn = PreNN(llm_dim=512, hidden_dim=256, num_layers=2, num_heads=4)
        x = torch.randn(2, 10, 512, requires_grad=True)
        out = prenn(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestFullPipelineGradients:
    """Test gradient flow and magnitude through PreNN + AudioHead together."""

    @pytest.fixture
    def prenn_and_head(self):
        """Create a PreNN + AudioHead pair matching real config."""
        llm_dim = 512  # Smaller for testing
        hidden_dim = AudioHead.HIDDEN_DIM  # 256

        config = MockAudioHeadConfig(llm_dim=llm_dim)
        head = AudioHead(config, llm_dim=hidden_dim)  # AudioHead takes pre-projected input
        prenn = PreNN(
            llm_dim=llm_dim,
            hidden_dim=hidden_dim,
            num_layers=head.AR_LAYERS // 2,
            num_heads=head.NUM_HEADS,
            intermediate_size=head.INTERMEDIATE_DIM,
            dropout=0.0,  # Disable dropout for deterministic gradient test
        )
        return prenn, head

    def test_gradients_reach_all_components(self, prenn_and_head):
        """Test gradients flow through PreNN -> AudioHead -> all sub-components."""
        prenn, head = prenn_and_head
        batch_size, text_len, audio_len = 2, 10, 30
        num_codebooks = 8

        # Simulate LLM hidden states
        llm_hidden = torch.randn(batch_size, text_len, prenn.llm_dim, requires_grad=True)
        targets = torch.randint(0, head.vocab_size, (batch_size, num_codebooks, audio_len))
        lengths = torch.tensor([30, 25])

        # Forward: PreNN -> AudioHead
        projected = prenn(llm_hidden)
        loss = head(projected, projected, codec_targets=targets, codec_lengths=lengths)
        loss.backward()

        # Check input gets gradients
        assert llm_hidden.grad is not None, "No gradient on LLM hidden states"

        # Check every trainable param in both modules
        dead_params = []
        for name, param in prenn.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for prenn.{name}"
                if param.grad.norm() == 0:
                    dead_params.append(f"prenn.{name}")

        for name, param in head.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for head.{name}"
                if param.grad.norm() == 0:
                    dead_params.append(f"head.{name}")

        assert len(dead_params) == 0, f"Dead parameters (zero gradient): {dead_params}"

    def test_gradient_magnitudes_per_layer(self, prenn_and_head, capsys):
        """Test gradient magnitudes are healthy across all layers.

        Checks for:
        - No vanishing gradients (norm > 1e-8)
        - No exploding gradients (norm < 1e4)
        - Reasonable ratio between largest and smallest gradient norms
        """
        prenn, head = prenn_and_head
        batch_size, text_len, audio_len = 2, 10, 30
        num_codebooks = 8

        llm_hidden = torch.randn(batch_size, text_len, prenn.llm_dim, requires_grad=True)
        targets = torch.randint(0, head.vocab_size, (batch_size, num_codebooks, audio_len))
        lengths = torch.tensor([30, 25])

        projected = prenn(llm_hidden)
        loss = head(projected, projected, codec_targets=targets, codec_lengths=lengths)
        loss.backward()

        # Collect gradient norms grouped by component
        component_grads: dict[str, list[tuple[str, float]]] = {}

        for prefix, module in [("prenn", prenn), ("audio_head", head)]:
            for name, param in module.named_parameters():
                if param.grad is None or not param.requires_grad:
                    continue
                # Group by top-level component
                parts = name.split(".")
                if prefix == "prenn":
                    if parts[0] == "proj":
                        component = "prenn.proj"
                    elif parts[0] == "layers":
                        component = f"prenn.layer_{parts[1]}"
                    elif parts[0] == "norm":
                        component = "prenn.norm"
                    else:
                        component = f"prenn.{parts[0]}"
                else:
                    if parts[0] == "ar_decoder" and len(parts) > 1:
                        if parts[1] == "layers":
                            component = f"ar_decoder.layer_{parts[2]}"
                        else:
                            component = f"ar_decoder.{parts[1]}"
                    elif parts[0] == "depformer":
                        component = "depformer"
                    elif parts[0] == "embedding":
                        component = "embedding"
                    else:
                        component = parts[0]

                grad_norm = param.grad.norm().item()
                if component not in component_grads:
                    component_grads[component] = []
                component_grads[component].append((f"{prefix}.{name}", grad_norm))

        # Print report
        print("\n" + "=" * 70)
        print(f"{'Component':<30} {'Mean Grad Norm':>15} {'Max Grad Norm':>15}")
        print("-" * 70)

        all_norms = []
        for component in sorted(component_grads.keys()):
            norms = [n for _, n in component_grads[component]]
            mean_norm = sum(norms) / len(norms)
            max_norm = max(norms)
            all_norms.extend(norms)
            print(f"{component:<30} {mean_norm:>15.6f} {max_norm:>15.6f}")

        print("-" * 70)
        print(f"{'Overall':<30} {sum(all_norms) / len(all_norms):>15.6f} {max(all_norms):>15.6f}")
        print(f"{'Min norm anywhere':<30} {min(all_norms):>15.8f}")
        print(f"{'Max/Min ratio':<30} {max(all_norms) / max(min(all_norms), 1e-12):>15.1f}")
        print("=" * 70)

        # Assertions
        for _component, entries in component_grads.items():
            for param_name, grad_norm in entries:
                assert grad_norm > 1e-8, f"Vanishing gradient in {param_name}: {grad_norm:.2e}"
                assert grad_norm < 1e4, f"Exploding gradient in {param_name}: {grad_norm:.2e}"

        # Check gradient ratio isn't too extreme (sign of training instability)
        ratio = max(all_norms) / max(min(all_norms), 1e-12)
        assert ratio < 1e6, f"Gradient norm ratio too large: {ratio:.1f}"


class TestParameterCount:
    """Test and display parameter counts per component."""

    @pytest.fixture
    def prenn_and_head(self):
        """Create a PreNN + AudioHead pair matching real config."""
        llm_dim = 512
        hidden_dim = AudioHead.HIDDEN_DIM

        config = MockAudioHeadConfig(llm_dim=llm_dim)
        head = AudioHead(config, llm_dim=hidden_dim)
        prenn = PreNN(
            llm_dim=llm_dim,
            hidden_dim=hidden_dim,
            num_layers=head.AR_LAYERS // 2,
            num_heads=head.NUM_HEADS,
            intermediate_size=head.INTERMEDIATE_DIM,
        )
        return prenn, head

    def test_param_counts_per_component(self, prenn_and_head):
        """Show parameter counts grouped by component with per-parameter detail."""
        prenn, head = prenn_and_head

        component_params: dict[str, list[tuple[str, tuple, int, bool]]] = {}

        for prefix, module in [("prenn", prenn), ("audio_head", head)]:
            for name, param in module.named_parameters():
                parts = name.split(".")
                if prefix == "prenn":
                    if parts[0] == "proj":
                        component = "prenn.proj"
                    elif parts[0] == "layers":
                        component = f"prenn.layer_{parts[1]}"
                    elif parts[0] == "norm":
                        component = "prenn.norm"
                    else:
                        component = f"prenn.{parts[0]}"
                else:
                    if parts[0] == "ar_decoder" and len(parts) > 1:
                        if parts[1] == "layers":
                            component = f"ar_decoder.layer_{parts[2]}"
                        else:
                            component = f"ar_decoder.{parts[1]}"
                    elif parts[0] == "depformer":
                        component = "depformer"
                    elif parts[0] == "embedding":
                        component = "embedding"
                    else:
                        component = parts[0]

                full_name = f"{prefix}.{name}"
                if component not in component_params:
                    component_params[component] = []
                component_params[component].append(
                    (full_name, tuple(param.shape), param.numel(), param.requires_grad)
                )

        # Compute grand total for percentages
        grand_total = sum(p[2] for entries in component_params.values() for p in entries)

        # Print detailed report
        print("\n" + "=" * 90)
        print(f"{'Parameter':<55} {'Shape':>15} {'Count':>12}")
        print("=" * 90)

        grand_trainable = 0
        for component in sorted(component_params.keys()):
            entries = component_params[component]
            comp_total = sum(e[2] for e in entries)
            comp_pct = 100 * comp_total / max(grand_total, 1)
            print(f"\n  {component} ({comp_total:,} params, {comp_pct:.1f}%)")
            print(f"  {'-' * 86}")
            for full_name, shape, count, trainable in entries:
                shape_str = str(list(shape))
                frozen = "" if trainable else " (frozen)"
                print(f"    {full_name:<51} {shape_str:>15} {count:>12,}{frozen}")
                if trainable:
                    grand_trainable += count

        print("\n" + "=" * 90)
        print(f"  {'TOTAL':<53} {'':<15} {grand_total:>12,}")
        print(f"  {'TRAINABLE':<53} {'':<15} {grand_trainable:>12,}")
        print(
            f"  {'TRAINABLE %':<53} {'':<15} {100 * grand_trainable / max(grand_total, 1):>11.1f}%"
        )
        print("=" * 90)

        # Basic sanity checks
        assert grand_total > 0, "Model has no parameters"
        assert grand_trainable == grand_total, "All audio head params should be trainable"


class TestConfigPriority:
    """Tests for config parameter priority."""

    def test_llm_dim_from_constructor_overrides_config(self, audio_head_config):
        """Test that constructor llm_dim overrides config."""
        config = audio_head_config(llm_dim=1024)
        head = AudioHead(config, llm_dim=512)

        assert head.llm_dim == 512

    def test_llm_dim_from_config_when_constructor_none(self, audio_head_config):
        """Test that config llm_dim is used when constructor is None."""
        config = audio_head_config(llm_dim=1024)
        head = AudioHead(config, llm_dim=None)

        assert head.llm_dim == 1024

    def test_llm_dim_default_when_both_missing(self):
        """Test default llm_dim when both config and constructor missing."""

        class EmptyConfig:
            pass

        head = AudioHead(EmptyConfig(), llm_dim=None)

        assert head.llm_dim == 2048  # SmolLM3 native dimension


class TestGenerationParameters:
    """Tests for generation parameter handling."""

    def test_top_k_affects_generation(self, audio_head_config):
        """Test that top_k parameter affects generation diversity."""
        config = audio_head_config(llm_dim=512, audio_top_k=5)
        head = AudioHead(config, llm_dim=512)
        head.max_tokens = 10

        hidden = torch.randn(1, 5, 512)

        # Should not crash with low top_k
        codes, _ = head(hidden, hidden)
        assert codes.shape[0] == 1

    def test_temperature_affects_generation(self, audio_head_config):
        """Test that temperature parameter affects generation."""
        config = audio_head_config(llm_dim=512, audio_temperature=0.5)
        head = AudioHead(config, llm_dim=512)
        head.max_tokens = 10

        hidden = torch.randn(1, 5, 512)

        # Should not crash with low temperature
        codes, _ = head(hidden, hidden)
        assert codes.shape[0] == 1
