"""
Configuration for NeuralFlex-MoE model.
"""

MODEL_CONFIG = {
    "model_name": "NeuralFlex-MoE",
    "variants": ["3B", "7B", "13B"],
    "architecture": "Hybrid-MoE-Transformer",
    "base_hidden_size": 2048,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,  # GQA optimization
    "intermediate_size": 5632,
    "num_hidden_layers": 24,
    "vocab_size": 65536,
    "max_position_embeddings": 32768,
    "rope_theta": 500000.0,
    "sliding_window": 4096,
    "num_experts": 16,
    "num_experts_per_tok": 2,
    "expert_capacity_factor": 1.25,
    "layer_norm_eps": 1e-6,
    "initializer_range": 0.02,
    "use_cache": True,
    
    # Uncertainty settings
    "uncertainty_threshold": 0.7,
    "alternative_beams": 3,
    
    # Continuous Learning
    "experience_replay_size": 10000,
    "ewc_lambda": 0.4,
    "batch_size": 8,
}

# Smaller config for testing/debugging on Colab free tier
DEBUG_CONFIG = MODEL_CONFIG.copy()
DEBUG_CONFIG.update({
    "base_hidden_size": 512,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "intermediate_size": 1024,
    "num_hidden_layers": 4,
    "num_experts": 4,
    "num_experts_per_tok": 2,
})
