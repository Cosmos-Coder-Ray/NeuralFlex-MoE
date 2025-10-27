"""Attention mechanisms with Flash Attention support"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False


class FlashAttentionMoE(nn.Module):
    """Multi-head attention with Flash Attention 2 and GQA support"""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["base_hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_key_value_heads = config.get("num_key_value_heads", self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.sliding_window = config.get("sliding_window", None)
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.attention_dropout = config.get("attention_dropout_prob", 0.1)
        self.use_flash_attention = FLASH_ATTENTION_AVAILABLE
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        
        batch_size, seq_length, _ = hidden_states.shape
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        
        if position_embeddings is not None:
            cos, sin = position_embeddings
            from .embeddings import apply_rotary_pos_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # Repeat k/v heads for GQA
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=2)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=2)
        
        if self.use_flash_attention and query_states.dtype in [torch.float16, torch.bfloat16]:
            attn_output = flash_attn_func(
                query_states, key_states, value_states,
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=True,
                window_size=(self.sliding_window, self.sliding_window) if self.sliding_window else (-1, -1)
            )
        else:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2)
        
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, past_key_value
