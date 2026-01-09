"""Tests for ASRModel."""

import pytest
import torch

from tiny_audio.asr_config import ASRConfig
from tiny_audio.asr_modeling import ASRModel

# Mark all tests in this module as slow (load ML models)
pytestmark = pytest.mark.slow


# Use session-scoped fixtures from conftest.py for most tests
@pytest.fixture
def config(base_asr_config):
    """Alias for session-scoped base_asr_config."""
    return base_asr_config


@pytest.fixture
def model(base_asr_model):
    """Alias for session-scoped base_asr_model."""
    return base_asr_model


class TestASRModelInitialization:
    """Tests for ASRModel initialization."""

    def test_model_creation(self, model):
        """Test that model can be created."""
        assert model is not None
        assert hasattr(model, "audio_tower")
        assert hasattr(model, "language_model")
        assert hasattr(model, "projector")
        assert hasattr(model, "tokenizer")
        assert hasattr(model, "feature_extractor")

    def test_audio_tower_frozen(self, model):
        """Test that audio encoder is frozen."""
        for param in model.audio_tower.parameters():
            assert not param.requires_grad

    def test_language_model_frozen(self, model):
        """Test that language model is frozen."""
        for param in model.language_model.parameters():
            assert not param.requires_grad

    def test_projector_trainable(self, model):
        """Test that projector is trainable."""
        trainable_params = sum(p.numel() for p in model.projector.parameters() if p.requires_grad)
        assert trainable_params > 0

    def test_audio_token_added(self, model):
        """Test that <audio> token is added to tokenizer."""
        assert "<audio>" in model.tokenizer.get_vocab()
        assert model.audio_token_id is not None
        assert model.audio_token_id == model.tokenizer.convert_tokens_to_ids("<audio>")

    def test_generation_config(self, model):
        """Test that generation config is set up correctly."""
        assert model.generation_config.do_sample is False
        assert model.generation_config.max_new_tokens == model.config.max_new_tokens


class TestASRModelStateDict:
    """Tests for state dict behavior."""

    def test_state_dict_only_projector(self, model):
        """Test that state_dict only contains projector weights."""
        state = model.state_dict()
        for key in state:
            assert key.startswith("projector."), f"Unexpected key in state dict: {key}"

    def test_state_dict_not_empty(self, model):
        """Test that state_dict is not empty."""
        state = model.state_dict()
        assert len(state) > 0


class TestAudioEncoding:
    """Tests for audio encoding functionality."""

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio features."""
        # Whisper expects (batch, n_mels=80, mel_len=3000) for 30 seconds
        # We use 3000 frames which is what Whisper's feature extractor outputs
        return torch.randn(1, 80, 3000)

    @pytest.fixture
    def audio_mask(self):
        """Create audio attention mask."""
        # Match the mel length
        return torch.ones(1, 3000)

    def test_encode_audio_shape(self, model, sample_audio, audio_mask):
        """Test that audio encoding produces correct shape."""
        audio_embeds = model._encode_audio(sample_audio, audio_mask)

        # Should be flattened: (batch * seq, hidden_dim)
        assert len(audio_embeds.shape) == 2
        assert audio_embeds.shape[1] == model.config.llm_dim

    def test_encode_audio_no_gradients_to_encoder(self, model, sample_audio, audio_mask):
        """Test that encoder does not receive gradients."""
        sample_audio.requires_grad = True
        audio_embeds = model._encode_audio(sample_audio, audio_mask)

        # Audio embeds should have gradients (from projector)
        loss = audio_embeds.sum()
        loss.backward()

        # But audio tower params should not have gradients computed
        for param in model.audio_tower.parameters():
            assert param.grad is None

    def test_get_num_audio_tokens(self, model, audio_mask):
        """Test audio token count calculation."""
        num_tokens = model._get_num_audio_tokens(audio_mask)
        # With 3000 mel frames -> 1500 encoder frames (stride-2) -> projector output
        expected = model.projector.get_output_length(1500)
        assert num_tokens == expected


class TestASRModelForward:
    """Tests for forward pass."""

    @pytest.fixture
    def batch(self, model):
        """Create a sample batch for testing."""
        # Create audio features (Whisper expects 3000 mel frames)
        audio_features = torch.randn(1, 80, 3000)
        audio_mask = torch.ones(1, 3000)

        # Get expected number of audio tokens
        num_audio_tokens = int(model._get_num_audio_tokens(audio_mask))

        # Build input with audio tokens
        audio_placeholder = "<audio>" * num_audio_tokens
        messages = [{"role": "user", "content": f"Transcribe: {audio_placeholder}"}]

        chat_result = model.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # Handle both BatchEncoding and plain tensor returns
        input_ids = chat_result.input_ids if hasattr(chat_result, "input_ids") else chat_result

        attention_mask = torch.ones_like(input_ids)

        # Create labels (mask audio tokens)
        labels = input_ids.clone()
        labels[input_ids == model.audio_token_id] = -100

        return {
            "input_ids": input_ids,
            "input_features": audio_features,
            "audio_attention_mask": audio_mask,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def test_forward_returns_loss(self, model, batch):
        """Test that forward pass returns a loss when labels provided."""
        outputs = model(**batch)
        assert hasattr(outputs, "loss")
        assert outputs.loss is not None
        assert outputs.loss.numel() == 1

    def test_forward_returns_logits(self, model, batch):
        """Test that forward pass returns logits."""
        outputs = model(**batch)
        assert hasattr(outputs, "logits")
        assert outputs.logits is not None
        # Logits shape: (batch, seq_len, vocab_size)
        assert len(outputs.logits.shape) == 3

    def test_forward_without_labels(self, model, batch):
        """Test forward pass without labels."""
        del batch["labels"]
        outputs = model(**batch)
        assert outputs.loss is None
        assert outputs.logits is not None

    def test_forward_gradients_to_projector(self, model, batch):
        """Test that gradients flow to projector."""
        model.train()
        outputs = model(**batch)
        outputs.loss.backward()

        # Projector should have gradients
        has_grad = False
        for param in model.projector.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "Projector should receive gradients"


class TestASRModelGeneration:
    """Tests for generation functionality."""

    @pytest.fixture
    def audio_input(self, model):
        """Create audio input for generation."""
        # Whisper expects 3000 mel frames
        audio_features = torch.randn(1, 80, 3000)
        audio_mask = torch.ones(1, 3000)
        return audio_features, audio_mask

    def test_generate_returns_tokens(self, model, audio_input):
        """Test that generate returns token ids."""
        audio_features, audio_mask = audio_input
        output = model.generate(
            input_features=audio_features,
            audio_attention_mask=audio_mask,
            max_new_tokens=10,
        )
        assert output is not None
        assert isinstance(output, torch.Tensor)
        assert len(output.shape) in [1, 2]  # (seq,) or (batch, seq)

    def test_generate_without_input_ids(self, model, audio_input):
        """Test generation builds prompt internally when no input_ids."""
        audio_features, audio_mask = audio_input
        output = model.generate(
            input_features=audio_features,
            audio_attention_mask=audio_mask,
            max_new_tokens=5,
        )
        assert output is not None

    def test_generate_requires_audio(self, model):
        """Test that generate raises error without audio."""
        with pytest.raises(ValueError, match="input_features required"):
            model.generate()

    def test_generate_requires_mask(self, model):
        """Test that generate raises error without attention mask."""
        audio_features = torch.randn(1, 80, 100)
        with pytest.raises(ValueError, match="audio_attention_mask required"):
            model.generate(input_features=audio_features)


class TestProjectorTypes:
    """Tests for different projector types."""

    @pytest.mark.parametrize("projector_type", ["mlp", "mosa"])
    def test_projector_type_initialization(self, projector_type):
        """Test that model initializes with different projector types."""
        config = ASRConfig(
            audio_model_id="openai/whisper-tiny",
            text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            projector_type=projector_type,
            model_dtype="float32",
            attn_implementation="eager",
        )
        model = ASRModel(config)
        assert model.projector is not None

    def test_invalid_projector_type(self):
        """Test that invalid projector type raises error."""
        config = ASRConfig(
            audio_model_id="openai/whisper-tiny",
            text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            projector_type="invalid_type",
            model_dtype="float32",
            attn_implementation="eager",
        )
        with pytest.raises(ValueError, match="Unknown projector_type"):
            ASRModel(config)


class TestEmbeddingOperations:
    """Tests for embedding-related operations."""

    def test_get_input_embeddings(self, model):
        """Test get_input_embeddings returns LLM embeddings."""
        embeddings = model.get_input_embeddings()
        assert embeddings is not None
        assert embeddings is model.language_model.get_input_embeddings()

    def test_get_output_embeddings(self, model):
        """Test get_output_embeddings returns LLM output layer."""
        embeddings = model.get_output_embeddings()
        assert embeddings is not None
        assert embeddings is model.language_model.get_output_embeddings()

    def test_audio_embeddings_replace_tokens(self, model):
        """Test that audio embeddings replace <audio> tokens correctly."""
        # Create input with audio tokens
        num_audio_tokens = 10
        text = "Hello " + "<audio>" * num_audio_tokens + " world"
        input_ids = model.tokenizer.encode(text, return_tensors="pt")

        # Get original embeddings
        original_embeds = model.language_model.get_input_embeddings()(input_ids)

        # Find audio token positions
        audio_mask = input_ids == model.audio_token_id
        num_found = audio_mask.sum().item()
        assert num_found == num_audio_tokens

        # The embedding at audio positions should be the same initially
        audio_positions = audio_mask[0].nonzero(as_tuple=True)[0]
        first_audio_embed = original_embeds[0, audio_positions[0]]
        second_audio_embed = original_embeds[0, audio_positions[1]]
        assert torch.allclose(first_audio_embed, second_audio_embed)


class TestProcessorIntegration:
    """Tests for processor integration."""

    def test_get_processor(self, model):
        """Test that get_processor returns a valid processor."""
        processor = model.get_processor()
        assert processor is not None
        assert hasattr(processor, "feature_extractor")
        assert hasattr(processor, "tokenizer")

    def test_get_processor_has_encoder_conv_layers(self, model):
        """Test that processor has encoder_conv_layers attribute."""
        processor = model.get_processor()
        assert hasattr(processor, "encoder_conv_layers")
        assert processor.encoder_conv_layers == model.config.encoder_conv_layers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
