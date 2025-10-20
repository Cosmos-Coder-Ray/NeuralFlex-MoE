"""Fine-tuning script with LoRA"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import argparse

from neuraflex_moe.models import NeuralFlexMoE
from neuraflex_moe.config import MODEL_CONFIG, TRAINING_CONFIG
from neuraflex_moe.training import NeuralFlexTrainer, DataPipeline
from neuraflex_moe.utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune NeuralFlex-MoE with LoRA")
    parser.add_argument("--base_model", type=str, required=True, help="Base model path")
    parser.add_argument("--output_dir", type=str, default="./finetuned", help="Output directory")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger = setup_logger("finetune", log_file=f"{args.output_dir}/finetune.log")
    logger.info("Starting LoRA fine-tuning")
    
    # Load base model
    logger.info(f"Loading base model from {args.base_model}...")
    model = NeuralFlexMoE(MODEL_CONFIG)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA
    logger.info("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare data
    logger.info("Preparing datasets...")
    data_pipeline = DataPipeline(MODEL_CONFIG)
    dataset = data_pipeline.prepare_dataset()
    
    train_size = int(0.95 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )
    
    # Update training config
    training_config = TRAINING_CONFIG.copy()
    training_config["learning_rate"] = args.learning_rate
    training_config["total_steps"] = len(train_dataset) * args.num_epochs // training_config["micro_batch_size"]
    
    # Initialize trainer
    trainer = NeuralFlexTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        config=training_config,
        output_dir=args.output_dir
    )
    
    # Start fine-tuning
    logger.info("Starting fine-tuning...")
    trainer.train()
    
    # Save LoRA weights
    logger.info(f"Saving LoRA weights to {args.output_dir}/lora_weights")
    model.save_pretrained(f"{args.output_dir}/lora_weights")
    
    logger.info("Fine-tuning completed!")


if __name__ == "__main__":
    main()
