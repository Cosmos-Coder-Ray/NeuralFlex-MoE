"""Main NeuralFlex-MoE model implementation"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from .attention import FlashAttentionMoE
from .moe_layer import MoELayer
from .embeddings import NeuralFlexEmbedding, RotaryEmbedding


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


class NeuralFlexDecoderLayer(nn.Module):
    """Single transformer decoder layer with MoE"""
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config["base_hidden_size"]
        
        self.self_attn = FlashAttentionMoE(config)
        self.moe = MoELayer(config)
        
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.get("layer_norm_eps", 1e-6))
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.get("layer_norm_eps", 1e-6))
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], torch.Tensor]:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        hidden_states, aux_loss = self.moe(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value, aux_loss


class NeuralFlexMoE(nn.Module):
    """NeuralFlex-MoE: Mixture of Experts with Adaptive Reasoning"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["base_hidden_size"]
        
        self.embed_tokens = NeuralFlexEmbedding(config)
        self.rotary_emb = RotaryEmbedding(
            self.hidden_size // config["num_attention_heads"],
            max_position_embeddings=config["max_position_embeddings"],
            base=config["rope_theta"]
        )
        
        self.layers = nn.ModuleList([
            NeuralFlexDecoderLayer(config, layer_idx)
            for layer_idx in range(config["num_hidden_layers"])
        ])
        
        self.norm = RMSNorm(self.hidden_size, eps=config.get("layer_norm_eps", 1e-6))
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        self.gradient_checkpointing = False
        self.post_init()
        
    def post_init(self):
        """Initialize weights"""
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        std = self.config.get("initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            
    def get_input_embeddings(self):
        return self.embed_tokens.token_embedding
        
    def set_input_embeddings(self, value):
        self.embed_tokens.token_embedding = value
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple:
        
        use_cache = use_cache if use_cache is not None else self.config.get("use_cache", True)
        
        batch_size, seq_length = input_ids.shape
        
        hidden_states = self.embed_tokens(input_ids)
        
        position_embeddings = self.rotary_emb(hidden_states, seq_len=seq_length)
        
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        
        past_key_values = past_key_values if past_key_values is not None else [None] * len(self.layers)
        present_key_values = [] if use_cache else None
        
        total_aux_loss = 0.0
        all_hidden_states = [] if output_hidden_states else None
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            past_key_value = past_key_values[idx]
            
            if self.gradient_checkpointing and self.training:
                hidden_states, present_key_value, aux_loss = self._gradient_checkpointing_func(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_embeddings,
                    past_key_value,
                    use_cache,
                )
            else:
                hidden_states, present_key_value, aux_loss = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )
            
            total_aux_loss += aux_loss
            
            if use_cache:
                present_key_values.append(present_key_value)
        
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        logits = self.lm_head(hidden_states)
        
        return {
            "logits": logits,
            "past_key_values": present_key_values,
            "hidden_states": all_hidden_states,
            "aux_loss": total_aux_loss / len(self.layers),
        }
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing"""
        self.gradient_checkpointing = True
        
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self.gradient_checkpointing = False
    
    @staticmethod
    def _gradient_checkpointing_func(module, *args, **kwargs):
        """Wrapper for gradient checkpointing"""
        from torch.utils.checkpoint import checkpoint
        return checkpoint(module, *args, use_reentrant=False, **kwargs)
