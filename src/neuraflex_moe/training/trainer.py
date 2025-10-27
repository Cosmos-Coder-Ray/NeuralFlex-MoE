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
import os

from ..core.self_organizing_pathways import SelfOrganizingPathways

class NeuralFlexTrainer:
    """Custom trainer for NeuralFlex-MoE with MoE-specific optimizations and SONP"""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        eval_dataset,
        tokenizer,
        config: Dict,
        data_collator,
        output_dir: str = "./outputs"
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = output_dir
        self.data_collator = data_collator
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision=config.get("mixed_precision", "bf16"),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        )
        
        # Initialize SONP
        if self.config.get("sonp_enabled", False):
            self.sonp = SelfOrganizingPathways(config.get("sonp_config", {}))
            self.sonp_update_frequency = self.config.get("sonp_update_frequency", 100)
        else:
            self.sonp = None

        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Prepare for distributed training
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader = \
            self.accelerator.prepare(
                self.model,
                self.optimizer,
                DataLoader(train_dataset, batch_size=config.get("micro_batch_size", 2), collate_fn=self.data_collator),
                DataLoader(eval_dataset, batch_size=config.get("micro_batch_size", 2), collate_fn=self.data_collator)
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

    def _sonp_step(self):
        """Perform a Self-Organizing Neural Pathways update step."""
        if self.sonp is None or not self.sonp.enabled:
            return

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Prune underutilized pathways
        pruned_count = self.sonp.prune_pathways(unwrapped_model)
        if pruned_count > 0 and self.accelerator.is_main_process:
            print(f"[SONP] Pruned {pruned_count} pathways.")
            wandb.log({"sonp/pruned_count": pruned_count}, step=self.global_step)

        # Grow new pathways where gradients are high
        grown_count = self.sonp.grow_pathways(unwrapped_model)
        if grown_count > 0 and self.accelerator.is_main_process:
            print(f"[SONP] Grew {grown_count} new pathways.")
            wandb.log({"sonp/grown_count": grown_count}, step=self.global_step)

    def train_step(self, batch) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Forward pass
        with self.accelerator.accumulate(self.model):
            outputs = self.model(**batch)
            logits = outputs["logits"]
            aux_loss = outputs.get("aux_loss", 0.0)
            
            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Add auxiliary loss from MoE
            total_loss = lm_loss + self.config.get("moe_aux_loss_coeff", 0.01) * aux_loss
            
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
            outputs = self.model(**batch)
            logits = outputs["logits"]
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            
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
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Main training loop"""
        total_steps = self.config.get("total_steps", 100000)
        eval_steps = self.config.get("eval_steps", 500)
        save_steps = self.config.get("save_steps", 1000)
        self.global_step = 0

        if resume_from_checkpoint:
            try:
                training_state = torch.load(os.path.join(resume_from_checkpoint, "training_state.pt"))
                self.global_step = training_state['step']
                self.optimizer.load_state_dict(training_state['optimizer'])
                self.scheduler.load_state_dict(training_state['scheduler'])
                self.accelerator.load_state(os.path.join(resume_from_checkpoint, "accelerator_state"))
                print(f"Resumed from step: {self.global_step}")
            except FileNotFoundError:
                print(f"Warning: Checkpoint training state not found at {resume_from_checkpoint}. Starting from scratch.")

        progress_bar = tqdm(
            initial=self.global_step, total=total_steps,
            disable=not self.accelerator.is_main_process,
            desc="Training"
        )
        
        while self.global_step < total_steps:
            for batch in self.train_dataloader:
                # Training step
                metrics = self.train_step(batch)
                
                self.global_step += 1
                progress_bar.update(1)
                
                # Log metrics
                if self.accelerator.is_main_process and self.global_step % 10 == 0:
                    wandb.log(metrics, step=self.global_step)
                    progress_bar.set_postfix(metrics)
                
                # Evaluation
                if self.global_step % eval_steps == 0:
                    eval_metrics = self.evaluate()
                    if self.accelerator.is_main_process:
                        wandb.log(eval_metrics, step=self.global_step)
                        print(f"\nEval at step {self.global_step}: {eval_metrics}")
                
                # SONP Step
                if self.sonp and self.global_step % self.sonp_update_frequency == 0:
                    self._sonp_step()

                # Save checkpoint
                if self.global_step % save_steps == 0:
                    self.save_checkpoint(self.global_step)
                
                if self.global_step >= total_steps:
                    break
        
        progress_bar.close()
        
        # Final save
        self.save_checkpoint(self.global_step, final=True)
    
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
        os.makedirs(save_dir, exist_ok=True)

        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # Save model
        unwrapped_model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # Save optimizer, scheduler, and accelerator state
        self.accelerator.save_state(os.path.join(save_dir, "accelerator_state"))
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'step': step
        }, f"{save_dir}/training_state.pt")
        
        print(f"Checkpoint saved to {save_dir}")