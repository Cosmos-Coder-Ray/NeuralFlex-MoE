"""
Adaptive Reasoning Chain (ARC) implementation, inspired by the principles of the
Tiny Recursive Model (TRM) from Samsung AI.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

class TRM_ARC(nn.Module):
    """
    A Tiny Recursive Model-inspired Adaptive Reasoning Chain.

    This module applies a core reasoning network iteratively to refine a hidden state.
    It maintains two internal states:
    - z: A latent reasoning state (the "scratchpad").
    - y: The current answer state.

    In each step, the model updates both states based on the original input and the
    previous states, allowing for progressive refinement of the answer.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.get("base_hidden_size", 2048)
        self.max_reasoning_steps = config.get("arc_steps", 5) # Number of recursive steps
        self.dropout_rate = config.get("arc_dropout", 0.1)

        # The core recursive network. It takes the concatenation of the original input (x),
        # the latent reasoning state (z), and the current answer state (y).
        self.core_network = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            # The output is split into the new z and y states
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2) 
        )

        self.layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        max_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Perform TRM-style iterative reasoning.

        Args:
            hidden_states: The input from the previous layer in the transformer.
                           Shape: (batch, seq_len, hidden_size)
            max_steps: Optional override for the number of reasoning steps.

        Returns:
            The refined hidden states.
            Shape: (batch, seq_len, hidden_size)
        """
        num_steps = max_steps or self.max_reasoning_steps
        
        # Initialize latent reasoning state (z) and answer state (y)
        z = torch.zeros_like(hidden_states)
        y = torch.zeros_like(hidden_states)

        # The original input is preserved throughout the reasoning process
        x = hidden_states

        for _ in range(num_steps):
            # Concatenate the three states along the feature dimension
            combined_input = torch.cat([x, z, y], dim=-1)

            # Pass through the core recursive network
            updates = self.core_network(combined_input)

            # Split the output into updates for z and y
            update_z, update_y = updates.chunk(2, dim=-1)

            # Update the states. This is the recursive step.
            # The new reasoning state z is a function of the previous states.
            # The new answer state y is also updated.
            z = self.layer_norm(z + update_z)
            y = self.layer_norm(y + update_y)

        # The final output is the original input plus the refined answer state.
        # This is a residual connection over the entire reasoning process.
        return x + y