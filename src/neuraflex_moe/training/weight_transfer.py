"""
Weight transfer system for leveraging existing model weights.
Enables bootstrapping from Llama-2, Mistral, Phi-2, Qwen models.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Dict, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class WeightTransferSystem:
    """
    Intelligent weight transfer from existing models.
    Supports Llama-2, Mistral, Phi-2, Qwen architectures.
    """
    
    def __init__(self):
        self.compatible_models = {
            "llama": "meta-llama/Llama-2-7b-hf",
            "mistral": "mistralai/Mistral-7B-v0.1",
            "phi": "microsoft/phi-2",
            "qwen": "Qwen/Qwen1.5-7B"
        }
        
    def transfer_weights(
        self, 
        source_model_id: str, 
        target_model: nn.Module,
        strategy: str = "intelligent"
    ) -> nn.Module:
        """
        Transfer weights from source to target model.
        
        Args:
            source_model_id: HuggingFace model ID or local path
            target_model: Our NeuralFlex-MoE model
            strategy: 'intelligent', 'direct', or 'partial'
        """
        logger.info(f"Loading source model: {source_model_id}")
        
        # Load source model
        source_model = AutoModelForCausalLM.from_pretrained(
            source_model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu"  # Load to CPU first
        )
        
        source_state = source_model.state_dict()
        target_state = target_model.state_dict()
        
        # Create weight mapping
        weight_mapping = self._create_weight_mapping(
            source_state, 
            target_state,
            strategy
        )
        
        # Transfer weights
        transferred_count = 0
        for source_key, target_key in weight_mapping.items():
            if self._are_compatible(source_state[source_key], target_state[target_key]):
                target_state[target_key] = self._adapt_weights(
                    source_state[source_key],
                    target_state[target_key]
                )
                transferred_count += 1
        
        # Load transferred weights
        target_model.load_state_dict(target_state, strict=False)
        
        logger.info(f"✓ Transferred {transferred_count} weight tensors")
        logger.info(f"  Coverage: {transferred_count / len(target_state) * 100:.1f}%")
        
        return target_model
    
    def _create_weight_mapping(
        self, 
        source_state: Dict, 
        target_state: Dict,
        strategy: str
    ) -> Dict[str, str]:
        """Create mapping between source and target weights"""
        
        mapping = {}
        
        # Common layer mappings
        layer_mappings = {
            # Embeddings
            "model.embed_tokens": "embed_tokens.token_embedding",
            "embed_tokens": "embed_tokens.token_embedding",
            
            # Attention layers
            "self_attn.q_proj": "self_attn.q_proj",
            "self_attn.k_proj": "self_attn.k_proj",
            "self_attn.v_proj": "self_attn.v_proj",
            "self_attn.o_proj": "self_attn.o_proj",
            
            # MLP/FFN layers - map to first expert
            "mlp.gate_proj": "moe.experts.0.gate_proj",
            "mlp.up_proj": "moe.experts.0.up_proj",
            "mlp.down_proj": "moe.experts.0.down_proj",
            
            # Normalization
            "input_layernorm": "input_layernorm",
            "post_attention_layernorm": "post_attention_layernorm",
            "norm": "norm",
            
            # LM head
            "lm_head": "lm_head"
        }
        
        for source_key in source_state.keys():
            # Try direct mapping first
            if source_key in target_state:
                mapping[source_key] = source_key
                continue
            
            # Try pattern matching
            for source_pattern, target_pattern in layer_mappings.items():
                if source_pattern in source_key:
                    target_key = source_key.replace(source_pattern, target_pattern)
                    if target_key in target_state:
                        mapping[source_key] = target_key
                        break
        
        return mapping
    
    def _are_compatible(self, source_tensor: torch.Tensor, target_tensor: torch.Tensor) -> bool:
        """Check if tensors are compatible for transfer"""
        # Same shape is ideal
        if source_tensor.shape == target_tensor.shape:
            return True
        
        # Allow some flexibility in dimensions
        if len(source_tensor.shape) == len(target_tensor.shape):
            # Check if we can adapt (e.g., truncate or pad)
            return True
        
        return False
    
    def _adapt_weights(
        self, 
        source_tensor: torch.Tensor, 
        target_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Adapt source weights to fit target shape"""
        
        if source_tensor.shape == target_tensor.shape:
            return source_tensor.clone()
        
        # Handle dimension mismatches
        adapted = target_tensor.clone()
        
        # Truncate or pad each dimension
        for dim in range(len(source_tensor.shape)):
            source_size = source_tensor.shape[dim]
            target_size = target_tensor.shape[dim]
            
            if source_size > target_size:
                # Truncate
                indices = [slice(None)] * len(source_tensor.shape)
                indices[dim] = slice(0, target_size)
                source_tensor = source_tensor[tuple(indices)]
            elif source_size < target_size:
                # Pad with zeros
                pad_size = target_size - source_size
                padding = [0] * (2 * len(source_tensor.shape))
                padding[2 * dim + 1] = pad_size
                source_tensor = torch.nn.functional.pad(source_tensor, padding)
        
        return source_tensor
    
    def replicate_to_experts(
        self, 
        model: nn.Module, 
        num_experts: int = 16
    ) -> nn.Module:
        """
        Replicate the first expert's weights to all other experts.
        Useful after transferring FFN weights to expert 0.
        """
        logger.info(f"Replicating expert 0 weights to {num_experts} experts...")
        
        for layer_idx, layer in enumerate(model.layers):
            if hasattr(layer, 'moe') and hasattr(layer.moe, 'experts'):
                # Get first expert's weights
                expert_0_state = layer.moe.experts[0].state_dict()
                
                # Copy to all other experts with small noise
                for expert_idx in range(1, num_experts):
                    expert_state = expert_0_state.copy()
                    
                    # Add small random noise to break symmetry
                    for key in expert_state:
                        noise = torch.randn_like(expert_state[key]) * 0.01
                        expert_state[key] = expert_state[key] + noise
                    
                    layer.moe.experts[expert_idx].load_state_dict(expert_state)
        
        logger.info("✓ Expert weights replicated")
        return model


def quick_transfer(
    source_model: str = "mistralai/Mistral-7B-v0.1",
    target_model = None,
    output_dir: str = "./models/transferred"
) -> nn.Module:
    """
    Quick weight transfer for common use case.
    
    Usage:
        from neuraflex_moe.models import NeuralFlexMoE
        from neuraflex_moe.config import MODEL_CONFIG
        
        model = NeuralFlexMoE(MODEL_CONFIG)
        model = quick_transfer("mistralai/Mistral-7B-v0.1", model)
    """
    if target_model is None:
        from neuraflex_moe.models import NeuralFlexMoE
        from neuraflex_moe.config import MODEL_CONFIG
        target_model = NeuralFlexMoE(MODEL_CONFIG)
    
    # Transfer weights
    transfer_system = WeightTransferSystem()
    model = transfer_system.transfer_weights(source_model, target_model)
    
    # Replicate to all experts
    model = transfer_system.replicate_to_experts(model)
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path / "model.pt")
    
    logger.info(f"✓ Model saved to {output_dir}")
    
    return model
