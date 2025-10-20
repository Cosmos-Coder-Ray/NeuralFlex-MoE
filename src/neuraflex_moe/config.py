"""Configuration settings for NeuralFlex-MoE"""

MODEL_CONFIG = {
    "model_name": "NeuralFlex-MoE",
    "variants": ["3B", "7B", "13B"],
    "architecture": "Hybrid-MoE-Transformer",
    "base_hidden_size": 2048,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 5632,
    "num_hidden_layers": 24,
    "vocab_size": 65536,
    "max_position_embeddings": 32768,
    "rope_theta": 500000.0,
    "sliding_window": 4096,
    "num_experts": 16,
    "num_experts_per_tok": 2,
    "expert_capacity_factor": 1.25,
    "hidden_dropout_prob": 0.1,
    "attention_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-6,
    "use_cache": True,
}

TRAINING_CONFIG = {
    "strategy": "FSDP",
    "gradient_checkpointing": True,
    "mixed_precision": "bf16",
    "gradient_accumulation_steps": 8,
    "micro_batch_size": 2,
    "optimizer": "AdamW-8bit",
    "learning_rate": 3e-4,
    "warmup_steps": 2000,
    "total_steps": 100000,
    "eval_steps": 500,
    "save_steps": 1000,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "lr_scheduler_type": "cosine",
}

MOE_CONFIG = {
    "sparse_experts": 16,
    "active_experts": 2,
    "expert_compression": "int8",
    "routing_algorithm": "top-k-gating-with-noise",
    "load_balancing": "auxiliary-loss",
    "router_z_loss_coef": 0.001,
    "router_aux_loss_coef": 0.01,
}

NOVEL_FEATURES_CONFIG = {
    "self_organizing_pathways": {
        "enabled": True,
        "pathway_threshold": 0.01,
        "pruning_rate": 0.1,
        "growth_rate": 0.05,
    },
    "temporal_context_compression": {
        "enabled": True,
        "compression_ratio": 10,
        "memory_bank_size": 10000,
    },
    "uncertainty_aware_generation": {
        "enabled": True,
        "uncertainty_threshold": 0.7,
        "alternative_beams": 3,
    },
    "continuous_learning": {
        "enabled": True,
        "experience_replay_size": 10000,
        "ewc_lambda": 0.4,
    },
}
