"""
Audio encoder for NeuralFlex-MoE, enabling audio understanding.
"""

import torch
import torch.nn as nn
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model

class AudioEncoder(nn.Module):
    """
    An audio encoder based on a pre-trained Wav2Vec2 model.

    This module freezes a pre-trained Wav2Vec2 model and uses it as a feature
    extractor. The output embeddings are then projected to match the hidden
    dimension of the main NeuralFlex-MoE model.
    """

    def __init__(self, output_dim: int, pretrained_model_name: str = "facebook/wav2vec2-base-960h"):
        super().__init__()
        self.output_dim = output_dim

        # Load pre-trained Wav2Vec2 model
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        
        # Freeze the pre-trained model's parameters
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

        # Get the hidden size of the pre-trained model
        pretrained_hidden_size = self.wav2vec2.config.hidden_size

        # Projection layer to map Wav2Vec2 output to the desired dimension
        self.projection = nn.Linear(pretrained_hidden_size, output_dim)

    def forward(self, audio_values: torch.Tensor) -> torch.Tensor:
        """
        Encodes audio waveforms into a sequence of embeddings.

        Args:
            audio_values: A batch of audio waveforms/features from the audio processor.
                          Shape: (batch_size, sequence_length)

        Returns:
            A tensor of audio embeddings.
            Shape: (batch_size, num_features, output_dim)
        """
        self.wav2vec2.eval() # Ensure the frozen model is in eval mode
        
        # Pass audio through the Wav2Vec2 model
        # The output is a tuple, we want the last hidden state
        outputs = self.wav2vec2(audio_values)
        audio_embeddings = outputs.last_hidden_state

        # Project the embeddings to the model's hidden dimension
        projected_embeddings = self.projection(audio_embeddings)

        return projected_embeddings
