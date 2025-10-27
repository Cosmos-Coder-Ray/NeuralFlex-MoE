"""Adaptive Reasoning Chains (ARC) for NeuralFlex-MoE"""

import torch
import torch.nn as nn

class ReasoningCell(nn.Module):
    """A single step in the recursive reasoning process"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.reasoning_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor, reasoning_state: torch.Tensor) -> torch.Tensor:
        """Performs one step of reasoning"""
        # Combine the hidden state with the current reasoning state
        combined_input = hidden_states + reasoning_state
        # Update the reasoning state
        new_reasoning_state = self.reasoning_layer(combined_input)
        return new_reasoning_state


class ARC(nn.Module):
    """Adaptive Reasoning Chains module"""

    def __init__(self, hidden_size: int, num_reasoning_steps: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_reasoning_steps = num_reasoning_steps
        self.reasoning_cell = ReasoningCell(hidden_size)
        self.gate = nn.Linear(hidden_size, 1) # Gate to decide when to stop reasoning

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Performs recursive reasoning"""
        # Initialize the reasoning state
        reasoning_state = torch.zeros_like(hidden_states)

        for _ in range(self.num_reasoning_steps):
            reasoning_state = self.reasoning_cell(hidden_states, reasoning_state)
            # The gate is a simple placeholder for a more complex stopping mechanism.
            # A sigmoid function is used to get a value between 0 and 1.
            gate_value = torch.sigmoid(self.gate(reasoning_state))
            if (gate_value > 0.8).all(): # If the gate is confident, stop reasoning
                break

        return hidden_states + reasoning_state # Add the final reasoning state to the hidden state
