"""Core novel features for NeuralFlex-MoE"""

from .self_organizing_pathways import SelfOrganizingPathways
from .temporal_context_compression import TemporalContextCompressor
from .uncertainty_aware_generation import UncertaintyAwareGeneration
from .continuous_learning import ContinuousLearningModule
from .adaptive_reasoning import AdaptiveReasoningChain
from .multi_turn_reasoner import MultiTurnReasoner

__all__ = [
    "SelfOrganizingPathways",
    "TemporalContextCompressor",
    "UncertaintyAwareGeneration",
    "ContinuousLearningModule",
    "AdaptiveReasoningChain",
    "MultiTurnReasoner",
]
