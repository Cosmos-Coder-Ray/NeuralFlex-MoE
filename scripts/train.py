"""Main training script for NeuralFlex-MoE"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from transformers import AutoTokenizer
import argparse

from neuraflex_moe.models import NeuralFlexMoE
from neuraflex_moe.config import MODEL_CONFIG, TRAINING_CONFIG
from neuraflex_moe.training import NeuralFlexTrainer, DataPipeline
from neuraflex_moe.utils import setup_logger, optimize_memory


def parse_args():
    parser = argparse.ArgumentParser(description="Train NeuralFlex-MoE")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--model_size", type=str, default="7B", choices=["3B", "7B", "13B"])
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--total_steps", type=int, default=100000, help="Total training steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save frequency")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logger
    logger = setup_logger("train", log_file=f"{args.output_dir}/train.log")
    logger.info("Starting NeuralFlex-MoE training")
    logger.info(f"Arguments: {args}")
    
    # Update config
    config = MODEL_CONFIG.copy()
    training_config = TRAINING_CONFIG.copy()
    training_config["micro_batch_size"] = args.batch_size
    training_config["learning_rate"] = args.learning_rate
    training_config["total_steps"] = args.total_steps
    training_config["eval_steps"] = args.eval_steps
    training_config["save_steps"] = args.save_steps
    
    # Initialize model
    logger.info("Initializing model...")
    model = NeuralFlexMoE(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    # Optimize memory
    optimize_memory(model)
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare data
    logger.info("Preparing datasets...")
    data_pipeline = DataPipeline(config)
    dataset = data_pipeline.prepare_dataset()
    
    # Split dataset
    train_size = int(0.95 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )
    
    logger.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = NeuralFlexTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        config=training_config,
        output_dir=args.output_dir
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
