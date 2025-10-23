"""Image encoders for NeuralFlex-MoE"""

import torch
import torch.nn as nn

class SimpleCNNEncoder(nn.Module):
    """A simple CNN-based image encoder"""

    def __init__(self, output_dim: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, output_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.cnn(images)


class VisionTransformerEncoder(nn.Module):
    """A placeholder for a Vision Transformer (ViT) encoder"""

    def __init__(self, output_dim: int):
        super().__init__()
        # This is a placeholder for a pre-trained ViT model.
        # In a real implementation, you would load a pre-trained ViT model
        # from a library like `timm` or `transformers`.
        self.vit = nn.Linear(2048, output_dim) # Placeholder

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # The input to a real ViT would be a tensor of shape (batch_size, 3, height, width)
        # The output would be a sequence of embeddings.
        # For this placeholder, we assume the input is already processed.
        return self.vit(images)
