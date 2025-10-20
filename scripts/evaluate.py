"""Evaluation script for benchmarking"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from transformers import AutoTokenizer
import argparse

from neuraflex_moe.models import NeuralFlexMoE
from neuraflex_moe.config import MODEL_CONFIG
from neuraflex_moe.evaluation import ModelEvaluator
from neuraflex_moe.training import DataPipeline
from neuraflex_moe.utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NeuralFlex-MoE")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger = setup_logger("evaluate")
    logger.info("Starting model evaluation")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    model = NeuralFlexMoE(MODEL_CONFIG)
    
    if os.path.exists(args.model_path):
        checkpoint = torch.load(
            os.path.join(args.model_path, "pytorch_model.bin"),
            map_location=args.device
        )
        model.load_state_dict(checkpoint, strict=False)
        logger.info("Model loaded successfully")
    else:
        logger.warning("Checkpoint not found, using random weights")
    
    model = model.to(args.device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # Prepare evaluation datasets
    logger.info("Preparing evaluation datasets...")
    data_pipeline = DataPipeline(MODEL_CONFIG)
    eval_dataset = data_pipeline.prepare_dataset()
    
    # Create dataloader
    from torch.utils.data import DataLoader
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, tokenizer, device=args.device)
    
    # Run evaluation
    logger.info("Running evaluation...")
    results = evaluator.evaluate_all({"eval": eval_dataloader})
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for dataset_name, metrics in results.items():
        print(f"\n{dataset_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    print("="*50 + "\n")
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
