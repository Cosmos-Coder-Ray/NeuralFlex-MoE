"""Continuous Learning Module (CLM) for NeuralFlex-MoE"""

import torch
import torch.nn as nn
from collections import deque
import random

class ExperienceReplayBuffer:
    """A simple experience replay buffer"""

    def __init__(self, buffer_size: int):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class EWC(nn.Module):
    """Elastic Weight Consolidation (EWC)"""

    def __init__(self, model: nn.Module, dataset: torch.utils.data.Dataset, lambda_ewc: float = 0.1):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.lambda_ewc = lambda_ewc
        self.fisher_matrix = self._compute_fisher_matrix()
        self.optimal_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}

    def _compute_fisher_matrix(self):
        """Computes the Fisher Information Matrix"""
        fisher_matrix = {}
        for name, param in self.model.named_parameters():
            fisher_matrix[name] = torch.zeros_like(param.data)

        self.model.eval()
        for inputs, targets in self.dataset:
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs.logits, targets)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher_matrix[name] += param.grad.data.pow(2)
        self.model.train()

        for name in fisher_matrix:
            fisher_matrix[name] /= len(self.dataset)

        return fisher_matrix

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Calculates the EWC penalty"""
        loss = 0
        for name, param in model.named_parameters():
            fisher = self.fisher_matrix[name]
            optimal_param = self.optimal_params[name]
            loss += (fisher * (param - optimal_param).pow(2)).sum()
        return self.lambda_ewc * loss
