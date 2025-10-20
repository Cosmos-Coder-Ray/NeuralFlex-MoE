"""Temporal Context Compression (TCC) implementation"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from collections import deque


class CompressedMemoryBank(nn.Module):
    """Memory bank for storing compressed context"""
    
    def __init__(self, hidden_size: int, max_size: int = 10000):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_size = max_size
        self.memory = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        
    def store(self, compressed_context: torch.Tensor, timestamp: Optional[int] = None):
        """Store compressed context"""
        self.memory.append(compressed_context.detach().cpu())
        self.timestamps.append(timestamp if timestamp is not None else len(self.memory))
        
    def retrieve(self, n: int = 1) -> List[torch.Tensor]:
        """Retrieve n most recent compressed contexts"""
        return list(self.memory)[-n:] if n <= len(self.memory) else list(self.memory)
    
    def clear(self):
        """Clear memory bank"""
        self.memory.clear()
        self.timestamps.clear()


class TemporalContextCompressor(nn.Module):
    """
    Compresses historical context into learned representations,
    enabling 10x longer context windows without memory increase
    """
    
    def __init__(self, config):
        super().__init__()
        self.compression_ratio = config.get("compression_ratio", 10)
        self.hidden_size = config.get("base_hidden_size", 2048)
        self.memory_bank_size = config.get("memory_bank_size", 10000)
        self.enabled = config.get("enabled", True)
        
        # Compression layers
        self.compressor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size // self.compression_ratio),
        )
        
        # Decompression layers
        self.decompressor = nn.Sequential(
            nn.Linear(self.hidden_size // self.compression_ratio, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size),
        )
        
        self.memory_bank = CompressedMemoryBank(
            self.hidden_size // self.compression_ratio,
            self.memory_bank_size
        )
        
        # Attention for selective retrieval
        self.retrieval_attention = nn.MultiheadAttention(
            self.hidden_size,
            num_heads=8,
            batch_first=True
        )
        
    def hierarchical_compress(self, context: torch.Tensor) -> torch.Tensor:
        """Compress context hierarchically"""
        if not self.enabled:
            return context
        
        batch_size, seq_len, hidden_size = context.shape
        
        # Compress in chunks
        chunk_size = seq_len // self.compression_ratio
        if chunk_size == 0:
            chunk_size = 1
        
        compressed_chunks = []
        for i in range(0, seq_len, chunk_size):
            chunk = context[:, i:i+chunk_size, :]
            # Average pooling over time
            chunk_compressed = chunk.mean(dim=1, keepdim=True)
            # Further compress with learned transformation
            chunk_compressed = self.compressor(chunk_compressed)
            compressed_chunks.append(chunk_compressed)
        
        compressed = torch.cat(compressed_chunks, dim=1)
        return compressed
    
    def compress_context(self, context: torch.Tensor, timestamp: Optional[int] = None) -> torch.Tensor:
        """Compress and store context"""
        compressed = self.hierarchical_compress(context)
        
        # Store in memory bank
        for i in range(compressed.shape[0]):
            self.memory_bank.store(compressed[i], timestamp)
        
        return compressed
    
    def decompress_context(self, compressed: torch.Tensor) -> torch.Tensor:
        """Decompress context back to original size"""
        if not self.enabled:
            return compressed
        
        decompressed = self.decompressor(compressed)
        
        # Expand temporal dimension
        batch_size, compressed_len, _ = compressed.shape
        expanded_len = compressed_len * self.compression_ratio
        
        # Repeat and interpolate
        decompressed = decompressed.repeat_interleave(self.compression_ratio, dim=1)
        
        return decompressed
    
    def retrieve_relevant_context(
        self,
        query: torch.Tensor,
        n_contexts: int = 5
    ) -> torch.Tensor:
        """Retrieve most relevant compressed contexts"""
        if not self.enabled or len(self.memory_bank.memory) == 0:
            return torch.zeros(query.shape[0], 1, self.hidden_size, device=query.device)
        
        # Retrieve recent contexts
        recent_contexts = self.memory_bank.retrieve(n_contexts)
        
        if not recent_contexts:
            return torch.zeros(query.shape[0], 1, self.hidden_size, device=query.device)
        
        # Stack and move to device
        stacked_contexts = torch.stack([c.to(query.device) for c in recent_contexts], dim=1)
        
        # Decompress
        decompressed = self.decompress_context(stacked_contexts)
        
        # Use attention to select relevant parts
        attended_context, _ = self.retrieval_attention(
            query, decompressed, decompressed
        )
        
        return attended_context
    
    def forward(
        self,
        current_context: torch.Tensor,
        compress: bool = True,
        retrieve: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for context compression
        
        Args:
            current_context: Current context tensor
            compress: Whether to compress and store
            retrieve: Whether to retrieve past contexts
            
        Returns:
            Tuple of (compressed_context, retrieved_context)
        """
        compressed = None
        retrieved = None
        
        if compress:
            compressed = self.compress_context(current_context)
        
        if retrieve:
            retrieved = self.retrieve_relevant_context(current_context)
        
        return compressed, retrieved
    
    def get_compression_stats(self) -> dict:
        """Get compression statistics"""
        return {
            "compression_ratio": self.compression_ratio,
            "memory_bank_size": len(self.memory_bank.memory),
            "max_memory_size": self.memory_bank_size,
            "compressed_dim": self.hidden_size // self.compression_ratio,
        }
