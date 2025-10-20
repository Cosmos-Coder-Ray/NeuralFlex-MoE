"""Memory optimization utilities"""

import torch
import gc


def optimize_memory(model=None):
    """Optimize memory usage"""
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Run garbage collection
    gc.collect()
    
    # Enable memory efficient attention if model provided
    if model is not None:
        try:
            model.enable_xformers_memory_efficient_attention()
        except:
            pass
        
        # Enable gradient checkpointing
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    
    return True


def get_memory_stats():
    """Get current memory statistics"""
    stats = {}
    
    if torch.cuda.is_available():
        stats['cuda_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
        stats['cuda_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB
        stats['cuda_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    return stats


def print_memory_stats():
    """Print memory statistics"""
    stats = get_memory_stats()
    print("\n=== Memory Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value:.2f} GB")
    print("========================\n")
