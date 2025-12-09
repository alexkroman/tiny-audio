import torch
import sys
from pathlib import Path
import pytest

# Add src directory to path to allow for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shared_moe_projector import SharedMoEAudioProjector

# Mock config class to initialize the model
class MockConfig:
    def __init__(self):
        self.encoder_dim = 128
        self.llm_dim = 256
        self.projector_hidden_dim = 512
        self.projector_pool_stride = 4
        self.num_experts = 4
        self.num_experts_per_tok = 2
        self.router_aux_loss_coef = 0.01
        self.router_z_loss_coef = 0.001
        self.model_dtype = "float32"

@pytest.fixture
def model_and_config():
    """Pytest fixture to provide a model instance and its config."""
    config = MockConfig()
    model = SharedMoEAudioProjector(config)
    model.train()
    return model, config

def test_instantiation(model_and_config):
    """Tests if the model can be instantiated correctly."""
    model, config = model_and_config
    assert model is not None, "Model should be instantiated."

def test_forward_pass_and_shape(model_and_config):
    """Tests the forward pass and checks the output shape."""
    model, config = model_and_config
    batch_size = 2
    seq_len = 100
    input_features = torch.randn(batch_size, seq_len, config.encoder_dim)

    output = model(input_features)

    expected_seq_len = seq_len // config.projector_pool_stride
    expected_shape = (batch_size, expected_seq_len, config.llm_dim)

    assert output.shape == expected_shape, f"Output shape is {output.shape}, expected {expected_shape}"

def test_aux_loss_calculation(model_and_config):
    """Tests that the auxiliary loss can be calculated."""
    model, config = model_and_config
    batch_size = 2
    seq_len = 100
    input_features = torch.randn(batch_size, seq_len, config.encoder_dim)

    # Forward pass is necessary to populate router logits
    model(input_features)

    aux_loss = model.get_aux_loss()
    assert isinstance(aux_loss, torch.Tensor), "Aux loss should be a tensor."
    assert aux_loss.ndim == 0, "Aux loss should be a scalar."

def test_backward_pass_and_gradients(model_and_config):
    """Tests the backward pass and checks for gradient existence."""
    model, config = model_and_config
    batch_size = 2
    seq_len = 100
    input_features = torch.randn(batch_size, seq_len, config.encoder_dim)

    output = model(input_features)
    aux_loss = model.get_aux_loss()

    loss = output.sum() + aux_loss
    loss.backward()

    grad_found = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if torch.any(param.grad != 0):
                grad_found = True
                break

    assert grad_found, "No non-zero gradients found on projector parameters after backward pass."
