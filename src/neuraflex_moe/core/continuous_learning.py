"""Continuous Learning Module (CLM) implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import deque
import copy


class ExperienceReplayBuffer:
    """Buffer for storing and sampling experiences"""
    
    def __init__(self, size: int = 10000):
        self.buffer = deque(maxlen=size)
        self.size = size
        
    def add(self, experience: Dict):
        """Add experience to buffer"""
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample random batch from buffer"""
        import random
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)
    
    def __len__(self):
        return len(self.buffer)


class ElasticWeightConsolidation:
    """Elastic Weight Consolidation for preventing catastrophic forgetting"""
    
    def __init__(self, model: nn.Module, lambda_ewc: float = 0.4):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_information = {}
        self.optimal_params = {}
        
    def compute_fisher_information(self, dataloader, device='cuda'):
        """Compute Fisher Information Matrix"""
        self.fisher_information = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_information[name] = torch.zeros_like(param.data)
        
        self.model.eval()
        
        for batch in dataloader:
            self.model.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            outputs = self.model(input_ids)
            logits = outputs['logits']
            
            # Sample from output distribution
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(logits.shape[:-1])
            
            # Compute log likelihood
            log_likelihood = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                sampled.view(-1),
                reduction='sum'
            )
            
            log_likelihood.backward()
            
            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher_information[name] += param.grad.data ** 2
        
        # Normalize
        n_samples = len(dataloader)
        for name in self.fisher_information:
            self.fisher_information[name] /= n_samples
        
        self.model.train()
        
    def save_optimal_params(self):
        """Save current parameters as optimal"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()
    
    def compute_ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss"""
        loss = 0.0
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_information and name in self.optimal_params:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                loss += (fisher * (param - optimal) ** 2).sum()
        
        return self.lambda_ewc * loss


class ContinuousLearningModule(nn.Module):
    """
    Enables model to learn from deployment interactions
    without catastrophic forgetting
    """
    
    def __init__(self, config):
        super().__init__()
        self.experience_replay_size = config.get("experience_replay_size", 10000)
        self.ewc_lambda = config.get("ewc_lambda", 0.4)
        self.batch_size = config.get("batch_size", 8)
        self.enabled = config.get("enabled", True)
        
        self.experience_replay = ExperienceReplayBuffer(size=self.experience_replay_size)
        self.ewc = None
        self.update_count = 0
        
    def initialize_ewc(self, model: nn.Module, dataloader):
        """Initialize EWC with Fisher information"""
        if not self.enabled:
            return
        
        self.ewc = ElasticWeightConsolidation(model, self.ewc_lambda)
        self.ewc.compute_fisher_information(dataloader)
        self.ewc.save_optimal_params()
    
    def add_interaction(
        self,
        input_ids: torch.Tensor,
        output_ids: torch.Tensor,
        feedback_score: Optional[float] = None,
        metadata: Optional[Dict] = None
    ):
        """Add interaction to experience replay buffer"""
        if not self.enabled:
            return
        
        experience = {
            'input_ids': input_ids.cpu(),
            'output_ids': output_ids.cpu(),
            'feedback_score': feedback_score,
            'metadata': metadata or {},
            'timestamp': self.update_count
        }
        
        self.experience_replay.add(experience)
    
    def online_learning_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda'
    ) -> Dict[str, float]:
        """Perform one online learning step"""
        if not self.enabled or len(self.experience_replay) < self.batch_size:
            return {'loss': 0.0, 'ewc_loss': 0.0}
        
        # Sample batch from experience replay
        batch = self.experience_replay.sample(self.batch_size)
        
        # Prepare batch
        input_ids = torch.stack([exp['input_ids'] for exp in batch]).to(device)
        output_ids = torch.stack([exp['output_ids'] for exp in batch]).to(device)
        
        # Forward pass
        model.train()
        outputs = model(input_ids)
        logits = outputs['logits']
        
        # Compute task loss
        task_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            output_ids.view(-1),
            ignore_index=-100
        )
        
        # Compute EWC loss
        ewc_loss = 0.0
        if self.ewc is not None:
            ewc_loss = self.ewc.compute_ewc_loss()
        
        # Total loss
        total_loss = task_loss + ewc_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        self.update_count += 1
        
        return {
            'loss': task_loss.item(),
            'ewc_loss': ewc_loss.item() if isinstance(ewc_loss, torch.Tensor) else ewc_loss,
            'total_loss': total_loss.item()
        }
    
    def update_weights_with_ewc(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        num_steps: int = 1,
        device: str = 'cuda'
    ) -> List[Dict[str, float]]:
        """Update model weights with EWC regularization"""
        if not self.enabled:
            return []
        
        losses = []
        for _ in range(num_steps):
            loss_dict = self.online_learning_step(model, optimizer, device)
            losses.append(loss_dict)
        
        return losses
    
    def consolidate_knowledge(self, model: nn.Module, dataloader):
        """Consolidate current knowledge before learning new tasks"""
        if not self.enabled:
            return
        
        if self.ewc is None:
            self.initialize_ewc(model, dataloader)
        else:
            # Update Fisher information and optimal parameters
            self.ewc.compute_fisher_information(dataloader)
            self.ewc.save_optimal_params()
    
    def get_learning_stats(self) -> Dict[str, any]:
        """Get continuous learning statistics"""
        return {
            'experience_buffer_size': len(self.experience_replay),
            'max_buffer_size': self.experience_replay_size,
            'update_count': self.update_count,
            'ewc_enabled': self.ewc is not None,
            'ewc_lambda': self.ewc_lambda
        }
