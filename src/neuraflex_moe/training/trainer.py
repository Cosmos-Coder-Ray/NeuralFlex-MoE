"""Main training loop for NeuralFlex-MoE"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from accelerate import Accelerator
import bitsandbytes as bnb
from typing import Optional, Dict
import wandb
from tqdm import tqdm


class NeuralFlexTrainer:
    """Custom trainer for NeuralFlex-MoE with MoE-specific optimizations"""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        eval_dataset,
        tokenizer,
        config: Dict,
        output_dir: str = "./outputs"
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = output_dir
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision=config.get("mixed_precision", "bf16"),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        )
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Prepare for distributed training
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader = \
            self.accelerator.prepare(
                self.model,
                self.optimizer,
                DataLoader(train_dataset, batch_size=config.get("micro_batch_size", 2)),
                DataLoader(eval_dataset, batch_size=config.get("micro_batch_size", 2))
            )
        
        # Enable gradient checkpointing
        if config.get("gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable()
        
        # Initialize wandb
        if self.accelerator.is_main_process:
            wandb.init(
                project="neuraflex-moe",
                config=config,
                name=f"neuraflex-{config.get('model_name', 'moe')}"
            )
    
    def _setup_optimizer(self):
        """Setup 8-bit AdamW optimizer"""
        return bnb.optim.AdamW8bit(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 3e-4),
            betas=(0.9, 0.95),
            weight_decay=self.config.get("weight_decay", 0.01)
        )
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        return CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get("total_steps", 100000),
            eta_min=self.config.get("learning_rate", 3e-4) * 0.1
        )
    
    def train_step(self, batch) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        labels = batch.get("labels", input_ids)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs["logits"]
        aux_loss = outputs.get("aux_loss", 0.0)
        
        # Compute loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss()
        lm_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # Add auxiliary loss from MoE
        total_loss = lm_loss + 0.01 * aux_loss
        
        # Backward pass
        self.accelerator.backward(total_loss)
        
        # Gradient clipping
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(),
                self.config.get("max_grad_norm", 1.0)
            )
        
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return {
            "loss": lm_loss.item(),
            "aux_loss": aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss,
            "total_loss": total_loss.item(),
            "lr": self.scheduler.get_last_lr()[0]
        }
    
    def eval_step(self, batch) -> Dict[str, float]:
        """Single evaluation step"""
        self.model.eval()
        
        with torch.no_grad():
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)
            labels = batch.get("labels", input_ids)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs["logits"]
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Compute perplexity
            perplexity = torch.exp(loss)
            
            return {
                "eval_loss": loss.item(),
                "perplexity": perplexity.item()
            }
    
    def train(self):
        """Main training loop"""
        total_steps = self.config.get("total_steps", 100000)
        eval_steps = self.config.get("eval_steps", 500)
        save_steps = self.config.get("save_steps", 1000)
        
        global_step = 0
        
        progress_bar = tqdm(
            total=total_steps,
            disable=not self.accelerator.is_main_process,
            desc="Training"
        )
        
        while global_step < total_steps:
            for batch in self.train_dataloader:
                # Training step
                metrics = self.train_step(batch)
                
                global_step += 1
                progress_bar.update(1)
                
                # Log metrics
                if self.accelerator.is_main_process and global_step % 10 == 0:
                    wandb.log(metrics, step=global_step)
                    progress_bar.set_postfix(metrics)
                
                # Evaluation
                if global_step % eval_steps == 0:
                    eval_metrics = self.evaluate()
                    if self.accelerator.is_main_process:
                        wandb.log(eval_metrics, step=global_step)
                        print(f"\nEval at step {global_step}: {eval_metrics}")
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    self.save_checkpoint(global_step)
                
                if global_step >= total_steps:
                    break
        
        progress_bar.close()
        
        # Final save
        self.save_checkpoint(global_step, final=True)
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        
        total_loss = 0.0
        total_perplexity = 0.0
        num_batches = 0
        
        for batch in tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            disable=not self.accelerator.is_main_process
        ):
            metrics = self.eval_step(batch)
            total_loss += metrics["eval_loss"]
            total_perplexity += metrics["perplexity"]
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_perplexity = total_perplexity / num_batches
        
        return {
            "eval_loss": avg_loss,
            "eval_perplexity": avg_perplexity
        }
    
    def save_checkpoint(self, step: int, final: bool = False):
        """Save model checkpoint"""
        if not self.accelerator.is_main_process:
            return
        
        save_dir = f"{self.output_dir}/checkpoint-{step}" if not final else f"{self.output_dir}/final"
        
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # Save model
        unwrapped_model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # Save optimizer and scheduler
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'step': step
        }, f"{save_dir}/training_state.pt")
        
        print(f"Checkpoint saved to {save_dir}")
