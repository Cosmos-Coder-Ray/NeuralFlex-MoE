"""Inference modules"""

from .generator import OptimizedInference
from .quantization import ModelQuantizer

__all__ = ["OptimizedInference", "ModelQuantizer"]
