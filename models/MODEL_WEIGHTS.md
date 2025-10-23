
# Model Weights for NeuralFlex-MoE

This file outlines the recommended open-source model weights for initializing NeuralFlex-MoE. The selection is based on the model's architecture and the goal of leveraging existing knowledge to accelerate training.

## Base Model Weights

These models are recommended as a starting point for the weight transfer process. The `WeightTransferSystem` in the codebase is designed to intelligently map and adapt weights from these models to the NeuralFlex-MoE architecture.

*   **Mistral-7B-v0.1:** A high-performing 7B parameter model from Mistral AI.
    *   **Hugging Face:** `mistralai/Mistral-7B-v0.1`
    *   **Compatibility:** Good architectural overlap, especially in the attention mechanism.
*   **Llama-2-7b:** The 7B parameter version of the Llama 2 model from Meta.
    *   **Hugging Face:** `meta-llama/Llama-2-7b`
    *   **Compatibility:** A strong baseline with a well-understood architecture.
*   **Qwen1.5-7B:** The 7B parameter version of the Qwen 1.5 model from Alibaba Cloud.
    *   **Hugging Face:** `Qwen/Qwen1.5-7B`
    *   **Compatibility:** A powerful model with a similar architecture.
*   **microsoft/phi-2:** A 2.7B parameter model from Microsoft.
    *   **Hugging Face:** `microsoft/phi-2`
    *   **Compatibility:** A good choice for initializing smaller variants of NeuralFlex-MoE.

## Weight Transfer Process

The `WeightTransferSystem` (`src/neuraflex_moe/training/weight_transfer.py`) is responsible for the following:

1.  **Loading Pre-trained Weights:** It downloads the weights of the selected source model from Hugging Face.
2.  **Weight Mapping:** It intelligently maps the layers from the source model to the target NeuralFlex-MoE model. This process accounts for differences in layer names and configurations.
3.  **Weight Adaptation:** It adapts the weights to the NeuralFlex-MoE architecture. This may involve resizing, reshaping, or fine-tuning the weights to ensure compatibility.

