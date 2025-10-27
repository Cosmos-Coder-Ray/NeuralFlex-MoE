"""Optimized inference with Flash Attention and KV caching"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
from transformers import AutoTokenizer


class DynamicKVCache:
    """Dynamic key-value cache for efficient inference"""
    
    def __init__(self, max_size: int = 8192):
        self.max_size = max_size
        self.cache = {}
        
    def update(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        """Update cache for a layer"""
        if layer_idx not in self.cache:
            self.cache[layer_idx] = {"key": key, "value": value}
        else:
            self.cache[layer_idx]["key"] = torch.cat([self.cache[layer_idx]["key"], key], dim=1)
            self.cache[layer_idx]["value"] = torch.cat([self.cache[layer_idx]["value"], value], dim=1)
            
            # Trim if exceeds max size
            if self.cache[layer_idx]["key"].shape[1] > self.max_size:
                self.cache[layer_idx]["key"] = self.cache[layer_idx]["key"][:, -self.max_size:, :]
                self.cache[layer_idx]["value"] = self.cache[layer_idx]["value"][:, -self.max_size:, :]
    
    def get(self, layer_idx: int):
        """Get cache for a layer"""
        return self.cache.get(layer_idx, None)
    
    def clear(self):
        """Clear all cache"""
        self.cache = {}


class OptimizedInference:
    """Optimized inference with various speedup techniques"""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
        use_flash_attention: bool = True
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.use_flash_attention = use_flash_attention
        self.kv_cache = DynamicKVCache(max_size=8192)
        
        # Set model to eval mode
        self.model.eval()
        
    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        use_cache: bool = True
    ) -> Dict[str, any]:
        """
        Generate text with optimizations
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            use_cache: Whether to use KV cache
            
        Returns:
            Dictionary with generated text and metadata
        """
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Clear cache
        if use_cache:
            self.kv_cache.clear()
        
        generated_tokens = []
        past_key_values = None
        
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            for _ in range(max_tokens):
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=use_cache
                )
                
                logits = outputs["logits"][:, -1, :]
                past_key_values = outputs.get("past_key_values", None)
                
                # Apply temperature
                logits = logits / temperature
                
                # Apply repetition penalty
                if generated_tokens:
                    for token_id in set(generated_tokens):
                        logits[0, token_id] /= repetition_penalty
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return {
            "text": generated_text,
            "tokens": generated_tokens,
            "num_tokens": len(generated_tokens),
            "prompt": prompt
        }
    
    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        **kwargs
    ) -> List[Dict[str, any]]:
        """Generate for multiple prompts"""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, max_tokens=max_tokens, **kwargs)
            results.append(result)
        return results
