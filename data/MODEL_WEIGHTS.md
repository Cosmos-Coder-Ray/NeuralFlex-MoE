# Pre-trained Model Weights for NeuralFlex-MoE

This document lists the recommended open-source base models for initializing the weights of NeuralFlex-MoE using the `WeightTransferSystem`. Using a pre-trained model as a starting point can significantly reduce training time and improve final performance.

---

## 1. Recommended Base Models

These models are compatible with the NeuralFlex-MoE architecture and offer strong foundational capabilities in language and reasoning. The `WeightTransferSystem` in `src/neuraflex_moe/training/weight_transfer.py` contains the logic to map their weights to our model's architecture.

| Model | Hugging Face ID | Key Strengths | Notes |
|---|---|---|---|
| **Mistral-7B-v0.1** | `mistralai/Mistral-7B-v0.1` | Excellent performance for its size, strong reasoning and code capabilities. Uses Grouped-Query Attention (GQA). | **Primary Recommendation.** A great starting point. |
| **Microsoft Phi-2** | `microsoft/phi-2` | A 2.7B parameter model with performance that rivals models 5x larger. Trained on high-quality "textbook" data. | Excellent choice for the 3B NeuralFlex variant. |
| **Qwen1.5-7B** | `Qwen/Qwen1.5-7B` | A powerful 7B model from Alibaba Cloud, part of a strong open-source series. | Good alternative to Mistral-7B. |
| **Llama-2-7b** | `meta-llama/Llama-2-7b-hf` | A solid and widely-used base model from Meta. | A reliable and well-understood option. |

---

## 2. Weight Transfer Process

The process of transferring weights is handled by the `WeightTransferSystem` and can be initiated via a script.

**Key Steps:**

1.  **Download Source Model:** The script will first download the desired base model from Hugging Face.
2.  **Intelligent Mapping:** The system maps the layers from the source model (e.g., `mistral.self_attn.q_proj`) to the corresponding layers in NeuralFlex-MoE (`neuraflex.self_attn.q_proj`).
3.  **FFN to MoE:** The Feed-Forward Network (FFN) weights from the source model are transferred to the **first expert** in each MoE layer.
4.  **Expert Replication:** The weights from the first expert are then copied to all other experts in the layer, with a small amount of random noise added to break symmetry and encourage specialization during training.

**Example Usage (to be implemented in `scripts/transfer_weights.py`):**

```bash
# This script will use the WeightTransferSystem to perform the transfer
python scripts/transfer_weights.py \
    --source_model_id "mistralai/Mistral-7B-v0.1" \
    --target_config_path "configs/neuraflex_7b.yaml" \
    --output_dir "./models/transferred/neuraflex-7b-from-mistral"
```

This will create a new set of initial weights for NeuralFlex-MoE, ready for fine-tuning or further pre-training.
