import torch
import torch.nn as nn
from typing import Optional, Tuple, List

# NeuralFlex Components
from .attention import FlashAttentionMoE
from .moe_layer import MoELayer
from .embeddings import NeuralFlexEmbedding, RotaryEmbedding
from .image_encoder import SimpleCNNEncoder
from .audio_encoder import AudioEncoder
from .cmrb import CMRB
from ..core.adaptive_reasoning import TRM_ARC

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
    """Single transformer decoder layer with MoE, ARC, and TCC"""
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config["base_hidden_size"]
        
        self.self_attn = FlashAttentionMoE(config)
        self.moe = MoELayer(config)
        self.arc = ARC(self.hidden_size)
        self.tcc = TCC(self.hidden_size)
        
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.get("layer_norm_eps", 1e-6))
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.get("layer_norm_eps", 1e-6))
        self.post_moe_layernorm = RMSNorm(self.hidden_size, eps=config.get("layer_norm_eps", 1e-6))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], torch.Tensor]:
        
        # 1. Self-Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        # 2. Mixture of Experts
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, aux_loss = self.moe(hidden_states)
        hidden_states = residual + hidden_states

        # 3. Adaptive Reasoning Chain (ARC) & Temporal Context Compression (TCC)
        residual = hidden_states
        hidden_states = self.post_moe_layernorm(hidden_states)
        
        # Apply ARC for iterative refinement
        hidden_states = self.arc(hidden_states)
        

        
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value, aux_loss

class NeuralFlexMoE(nn.Module):
    """NeuralFlex-MoE: Any-to-Text model with MoE and Adaptive Reasoning"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["base_hidden_size"]
        
        # Text Embedding
        self.embed_tokens = NeuralFlexEmbedding(config)
        
        # Multi-modal Encoders
        self.image_encoder = SimpleCNNEncoder(self.hidden_size)
        self.audio_encoder = AudioEncoder(self.hidden_size)
        
        # Multi-modal Fusion Bridge
        self.cmrb = CMRB(
            embed_dim=self.hidden_size,
            num_heads=config["num_attention_heads"] // 2 # Use fewer heads for cross-attention
        )
        
        # Core Transformer Blocks
        self.layers = nn.ModuleList([
            NeuralFlexDecoderLayer(config, layer_idx)
            for layer_idx in range(config["num_hidden_layers"])
        ])
        
        # Final normalization and output head
        self.norm = RMSNorm(self.hidden_size, eps=config.get("layer_norm_eps", 1e-6))
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        self.gradient_checkpointing = False
        self.post_init()
        
    def post_init(self):
        """Initialize weights"""
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        std = self.config.get("initializer_range", 0.02)
        if isinstance(module, (nn.Linear, nn.Conv2d)):
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
        pixel_values: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict:
        
        use_cache = use_cache if use_cache is not None else self.config.get("use_cache", True)
        
        # 1. Get text embeddings
        text_embeddings = self.embed_tokens(input_ids)

        # 2. Get multi-modal embeddings (if any)
        modal_context_embeddings = []
        if pixel_values is not None:
            image_embeddings = self.image_encoder(pixel_values)
            # Add a CLS-like token for the image
            modal_context_embeddings.append(image_embeddings.unsqueeze(1))

        if audio_values is not None:
            audio_embeddings = self.audio_encoder(audio_values)
            modal_context_embeddings.append(audio_embeddings)

        # 3. Fuse embeddings with CMRB
        if modal_context_embeddings:
            # Concatenate all modal embeddings to form the context
            context = torch.cat(modal_context_embeddings, dim=1)
            hidden_states = self.cmrb(text_embeddings, context)
        else:
            hidden_states = text_embeddings

        # 4. Pass through transformer decoder layers
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
                # Custom gradient checkpointing function to handle multiple outputs
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                hidden_states, present_key_value, aux_loss = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states, attention_mask, None, past_key_value, use_cache
                )
            else:
                hidden_states, present_key_value, aux_loss = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_embeddings=None, # RoPE is handled inside FlashAttentionMoE
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
            "aux_loss": total_aux_loss / len(self.layers) if self.layers else 0,
        }elf.layers) if self.layers else 0,
        }