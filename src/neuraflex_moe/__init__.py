"""
NeuralFlex-MoE: Mixture of Experts with Adaptive Reasoning
A revolutionary lightweight LLM architecture combining MoE with novel adaptive reasoning chains.
"""

__version__ = "0.1.0"
__author__ = "NeuralFlex Team"

from .config import MODEL_CONFIG, TRAINING_CONFIG
from .models.neuraflex_moe import NeuralFlexMoE

__all__ = [
    "MODEL_CONFIG",
    "TRAINING_CONFIG",
    "NeuralFlexMoE",
]
