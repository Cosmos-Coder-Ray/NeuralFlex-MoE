"""Multi-Turn Reasoning with Self-Correction"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class MultiTurnReasoner(nn.Module):
    """
    Implements iterative reasoning with self-correction
    """
    
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.max_iterations = config.get("max_iterations", 5)
        self.confidence_threshold = config.get("confidence_threshold", 0.85)
        self.hidden_size = config.get("base_hidden_size", 2048)
        
        # Critique module
        self.critique_head = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Improvement scorer
        self.improvement_scorer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def think(
        self,
        query_embedding: torch.Tensor,
        previous_thoughts: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Generate reasoning step"""
        if previous_thoughts is None or len(previous_thoughts) == 0:
            # First thought
            return query_embedding
        
        # Combine previous thoughts
        thought_history = torch.cat(previous_thoughts, dim=1)
        
        # Generate next thought considering history
        combined = torch.cat([query_embedding, thought_history.mean(dim=1, keepdim=True)], dim=-1)
        next_thought = self.critique_head(combined)
        
        return next_thought
    
    def answer_with_confidence(
        self,
        query_embedding: torch.Tensor,
        reasoning_chain: List[torch.Tensor]
    ) -> tuple:
        """Generate answer with confidence score"""
        if not reasoning_chain:
            return query_embedding, 0.5
        
        # Combine reasoning chain
        reasoning_context = torch.cat(reasoning_chain, dim=1)
        final_representation = reasoning_context.mean(dim=1, keepdim=True)
        
        # Compute confidence based on consistency
        if len(reasoning_chain) > 1:
            last_thought = reasoning_chain[-1]
            prev_thought = reasoning_chain[-2]
            
            # Measure improvement
            combined = torch.cat([last_thought, prev_thought], dim=-1)
            confidence = self.improvement_scorer(combined).squeeze(-1).mean().item()
        else:
            confidence = 0.7
        
        return final_representation, confidence
    
    def critique(
        self,
        answer: torch.Tensor,
        reasoning_chain: List[torch.Tensor]
    ) -> torch.Tensor:
        """Generate critique of current answer"""
        if not reasoning_chain:
            return answer
        
        # Compare answer with reasoning chain
        reasoning_summary = torch.cat(reasoning_chain, dim=1).mean(dim=1, keepdim=True)
        combined = torch.cat([answer, reasoning_summary], dim=-1)
        
        # Generate critique
        critique = self.critique_head(combined)
        
        return critique
    
    def reason(
        self,
        query_embedding: torch.Tensor,
        tokenizer=None,
        return_text: bool = False
    ) -> Dict[str, any]:
        """
        Perform multi-turn reasoning with self-correction
        
        Args:
            query_embedding: Query representation [batch, seq_len, hidden]
            tokenizer: Optional tokenizer for text conversion
            return_text: Whether to return text representations
            
        Returns:
            Dictionary with reasoning results
        """
        reasoning_chain = []
        current_answer = None
        confidences = []
        
        for iteration in range(self.max_iterations):
            # Generate reasoning step
            thought = self.think(query_embedding, reasoning_chain)
            reasoning_chain.append(thought)
            
            # Generate answer with confidence
            answer, confidence = self.answer_with_confidence(
                query_embedding, reasoning_chain
            )
            confidences.append(confidence)
            current_answer = answer
            
            # Check if confident enough
            if confidence > self.confidence_threshold:
                return {
                    "answer": current_answer,
                    "reasoning_chain": reasoning_chain,
                    "confidence": confidence,
                    "confidences": confidences,
                    "iterations": iteration + 1,
                    "converged": True
                }
            
            # Self-critique and improve
            if iteration < self.max_iterations - 1:
                critique = self.critique(answer, reasoning_chain)
                reasoning_chain.append(critique)
        
        # Return final result even if not converged
        return {
            "answer": current_answer,
            "reasoning_chain": reasoning_chain,
            "confidence": confidences[-1] if confidences else 0.0,
            "confidences": confidences,
            "iterations": self.max_iterations,
            "converged": False
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, any]:
        """
        Forward pass with multi-turn reasoning
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Reasoning results
        """
        # Get query embedding from model
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            hidden_states = outputs.get("hidden_states", None)
            
            if hidden_states is not None:
                query_embedding = hidden_states[-1] if isinstance(hidden_states, (list, tuple)) else hidden_states
            else:
                # Fallback: use model's last hidden state
                query_embedding = outputs["logits"]
        
        # Perform reasoning
        result = self.reason(query_embedding)
        
        return result
