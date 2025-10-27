"""Temporal Context Compression (TCC) for NeuralFlex-MoE"""

import torch
import torch.nn as nn

class TCC(nn.Module):
    """A simple Temporal Context Compression module"""

    def __init__(self, hidden_size: int, compression_ratio: int = 4):
        super().__init__()
        self.compression_conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=compression_ratio,
            stride=compression_ratio,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compresses the sequence of hidden states"""
        # The input hidden_states are of shape (batch_size, seq_len, hidden_size)
        # We need to transpose it to (batch_size, hidden_size, seq_len) for the 1D convolution
        hidden_states = hidden_states.transpose(1, 2)
        compressed_hidden_states = self.compression_conv(hidden_states)
        # Transpose it back to (batch_size, compressed_seq_len, hidden_size)
        return compressed_hidden_states.transpose(1, 2)
