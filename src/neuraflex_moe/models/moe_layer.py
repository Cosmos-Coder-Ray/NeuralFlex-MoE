"""Mixture of Experts layer implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Expert(nn.Module):
    """Single expert network"""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TopKRouter(nn.Module):
    """Top-K routing with load balancing"""
    
    def __init__(self, hidden_size: int, num_experts: int, num_experts_per_tok: int):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Load balancing loss
        router_probs = F.softmax(router_logits, dim=-1)
        expert_usage = router_probs.mean(dim=0)
        aux_loss = self.num_experts * (expert_usage * expert_usage).sum()
        
        return routing_weights, selected_experts, aux_loss


class MoELayer(nn.Module):
    """Mixture of Experts layer with sparse routing"""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["base_hidden_size"]
        self.num_experts = config["num_experts"]
        self.num_experts_per_tok = config["num_experts_per_tok"]
        self.intermediate_size = config["intermediate_size"]
        
        self.router = TopKRouter(
            self.hidden_size,
            self.num_experts,
            self.num_experts_per_tok
        )
        
        self.experts = nn.ModuleList([
            Expert(self.hidden_size, self.intermediate_size)
            for _ in range(self.num_experts)
        ])
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        routing_weights, selected_experts, aux_loss = self.router(hidden_states)
        
        final_output = torch.zeros_like(hidden_states_flat)
        
        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]
            
            expert_mask = (selected_experts == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
            
            expert_input = hidden_states_flat[expert_mask]
            expert_output = expert(expert_input)
            
            expert_weights = routing_weights[expert_mask]
            expert_weights = expert_weights[
                (selected_experts[expert_mask] == expert_idx).nonzero(as_tuple=True)
            ]
            
            final_output[expert_mask] += expert_output * expert_weights.unsqueeze(-1)
        
        final_output = final_output.view(batch_size, seq_len, hidden_dim)
        
        return final_output, aux_loss
