"""
Cross-Modal Reasoning Bridge (CMRB) for NeuralFlex-MoE.

This module uses cross-attention to fuse multi-modal embeddings (image, audio)
with text embeddings, creating a rich, unified representation.
"""

import torch
import torch.nn as nn
from typing import Optional

class CMRB(nn.Module):
    """
    A Cross-Modal Reasoning Bridge using cross-attention.

    This bridge takes text embeddings as the query and a sequence of multi-modal
    (e.g., concatenated image and audio) embeddings as the key and value.
    It allows the model to attend to the most relevant parts of the non-text
    modalities when processing the text, enriching the text representation.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # The core cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Expects (batch, seq, feature)
        )

        # Feed-forward network to process the output of the attention layer
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_embeddings: torch.Tensor,
        context_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuses text embeddings with multi-modal context embeddings.

        Args:
            text_embeddings: The embeddings for the input text (query).
                             Shape: (batch_size, text_seq_len, embed_dim)
            context_embeddings: The embeddings from other modalities (e.g., image, audio).
                                This serves as the key and value.
                                Shape: (batch_size, context_seq_len, embed_dim)

        Returns:
            The fused embeddings, with the same shape as text_embeddings.
        """
        # If there's no context, just return the text embeddings
        if context_embeddings is None:
            return text_embeddings

        # --- Cross-Attention Step ---
        # The text embeddings attend to the context embeddings.
        attn_output, _ = self.cross_attention(
            query=text_embeddings,
            key=context_embeddings,
            value=context_embeddings
        )

        # Add & Norm (first residual connection)
        fused_embeddings = self.norm1(text_embeddings + self.dropout(attn_output))

        # --- Feed-Forward Step ---
        ffn_output = self.ffn(fused_embeddings)

        # Add & Norm (second residual connection)
        fused_embeddings = self.norm2(fused_embeddings + self.dropout(ffn_output))

        return fused_embeddings