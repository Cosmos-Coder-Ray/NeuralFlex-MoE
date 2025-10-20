"""Unit tests for model components"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import pytest

from neuraflex_moe.models import NeuralFlexMoE
from neuraflex_moe.config import MODEL_CONFIG


@pytest.fixture
def model():
    """Create model instance for testing"""
    config = MODEL_CONFIG.copy()
    config["num_hidden_layers"] = 2  # Smaller for testing
    return NeuralFlexMoE(config)


@pytest.fixture
def sample_input():
    """Create sample input"""
    batch_size = 2
    seq_length = 16
    return torch.randint(0, 1000, (batch_size, seq_length))


def test_model_initialization(model):
    """Test model initializes correctly"""
    assert model is not None
    assert hasattr(model, 'embed_tokens')
    assert hasattr(model, 'layers')
    assert hasattr(model, 'norm')
    assert hasattr(model, 'lm_head')


def test_model_forward(model, sample_input):
    """Test forward pass"""
    outputs = model(sample_input)
    
    assert 'logits' in outputs
    assert outputs['logits'].shape[0] == sample_input.shape[0]
    assert outputs['logits'].shape[1] == sample_input.shape[1]
    assert outputs['logits'].shape[2] == MODEL_CONFIG['vocab_size']


def test_model_generation(model, sample_input):
    """Test generation capability"""
    model.eval()
    with torch.no_grad():
        outputs = model(sample_input)
        logits = outputs['logits']
        next_token = logits[:, -1, :].argmax(dim=-1)
    
    assert next_token.shape[0] == sample_input.shape[0]


def test_gradient_checkpointing(model):
    """Test gradient checkpointing"""
    model.gradient_checkpointing_enable()
    assert model.gradient_checkpointing == True
    
    model.gradient_checkpointing_disable()
    assert model.gradient_checkpointing == False


def test_model_parameters(model):
    """Test model has trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    assert total_params > 0
    assert trainable_params > 0
    assert trainable_params == total_params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
