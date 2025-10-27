"""Model evaluation on standard benchmarks"""

import torch
from typing import Dict, List
from tqdm import tqdm


class ModelEvaluator:
    """Evaluate model on standard benchmarks"""
    
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def evaluate_perplexity(self, dataset) -> float:
        """Calculate perplexity on dataset"""
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(dataset, desc="Evaluating perplexity"):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch.get("labels", input_ids)
                
                outputs = self.model(input_ids)
                logits = outputs["logits"]
                
                # Shift for causal LM
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Calculate loss
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='sum'
                )
                
                total_loss += loss.item()
                total_tokens += shift_labels.numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        return perplexity.item()
    
    def evaluate_accuracy(self, dataset) -> float:
        """Calculate accuracy on dataset"""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataset, desc="Evaluating accuracy"):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch.get("labels", input_ids)
                
                outputs = self.model(input_ids)
                logits = outputs["logits"]
                
                predictions = logits.argmax(dim=-1)
                
                # Compare predictions with labels
                correct += (predictions[:, :-1] == labels[:, 1:]).sum().item()
                total += labels[:, 1:].numel()
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def evaluate_all(self, datasets: Dict[str, any]) -> Dict[str, float]:
        """Evaluate on multiple datasets"""
        results = {}
        
        for name, dataset in datasets.items():
            print(f"\nEvaluating on {name}...")
            
            perplexity = self.evaluate_perplexity(dataset)
            accuracy = self.evaluate_accuracy(dataset)
            
            results[name] = {
                "perplexity": perplexity,
                "accuracy": accuracy
            }
            
            print(f"{name} - Perplexity: {perplexity:.2f}, Accuracy: {accuracy:.4f}")
        
        return results
