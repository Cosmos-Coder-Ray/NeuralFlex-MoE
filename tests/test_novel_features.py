"""Unit tests for novel features"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import pytest

from neuraflex_moe.core import (
    SelfOrganizingPathways,
    TemporalContextCompressor,
    UncertaintyAwareGeneration,
    ContinuousLearningModule,
    AdaptiveReasoningChain
)
from neuraflex_moe.config import NOVEL_FEATURES_CONFIG, MODEL_CONFIG


@pytest.fixture
def sample_tensor():
    """Create sample tensor"""
    return torch.randn(2, 16, 2048)


def test_self_organizing_pathways(sample_tensor):
    """Test self-organizing pathways"""
    config = NOVEL_FEATURES_CONFIG["self_organizing_pathways"]
    sonp = SelfOrganizingPathways(config)
    
    output = sonp.adaptive_routing(sample_tensor)
    assert output.shape == sample_tensor.shape
    
    stats = sonp.get_pathway_stats()
    assert "usage_rate" in stats


def test_temporal_context_compression(sample_tensor):
    """Test temporal context compression"""
    config = NOVEL_FEATURES_CONFIG["temporal_context_compression"]
    config["base_hidden_size"] = 2048
    tcc = TemporalContextCompressor(config)
    
    compressed, retrieved = tcc(sample_tensor, compress=True, retrieve=False)
    assert compressed is not None
    assert compressed.shape[0] == sample_tensor.shape[0]


def test_uncertainty_aware_generation():
    """Test uncertainty-aware generation"""
    config = NOVEL_FEATURES_CONFIG["uncertainty_aware_generation"]
    config["base_hidden_size"] = 2048
    config["vocab_size"] = 65536
    uag = UncertaintyAwareGeneration(config)
    
    logits = torch.randn(2, 16, 65536)
    hidden_states = torch.randn(2, 16, 2048)
    
    confidence = uag.compute_confidence(logits, hidden_states)
    assert confidence.shape == (2, 16)
    assert (confidence >= 0).all() and (confidence <= 1).all()


def test_continuous_learning_module():
    """Test continuous learning module"""
    config = NOVEL_FEATURES_CONFIG["continuous_learning"]
    clm = ContinuousLearningModule(config)
    
    input_ids = torch.randint(0, 1000, (2, 16))
    output_ids = torch.randint(0, 1000, (2, 16))
    
    clm.add_interaction(input_ids, output_ids, feedback_score=0.9)
    
    stats = clm.get_learning_stats()
    assert stats["experience_buffer_size"] == 1


def test_adaptive_reasoning_chain(sample_tensor):
    """Test adaptive reasoning chain"""
    config = {
        "base_hidden_size": 2048,
        "num_thought_tokens": 8,
        "max_reasoning_steps": 3,
        "confidence_threshold": 0.85
    }
    arc = AdaptiveReasoningChain(config)
    
    result = arc(sample_tensor, max_steps=3)
    
    assert "reasoning_states" in result
    assert "final_state" in result
    assert "confidences" in result
    assert "num_steps" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
