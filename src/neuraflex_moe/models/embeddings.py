"""Embedding layers with RoPE support"""

import torch
import torch.nn as nn
import math


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    
    def __init__(self, dim, max_position_embeddings=32768, base=500000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        return emb.cos()[None, :, :], emb.sin()[None, :, :]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class NeuralFlexEmbedding(nn.Module):
    """Token and position embeddings for NeuralFlex"""
    
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config["vocab_size"], config["base_hidden_size"])
        self.rotary_emb = RotaryEmbedding(
            config["base_hidden_size"] // config["num_attention_heads"],
            max_position_embeddings=config["max_position_embeddings"],
            base=config["rope_theta"]
        )
        self.dropout = nn.Dropout(config.get("hidden_dropout_prob", 0.1))
        
    def forward(self, input_ids):
        embeddings = self.token_embedding(input_ids)
        embeddings = self.dropout(embeddings)
        return embeddings
