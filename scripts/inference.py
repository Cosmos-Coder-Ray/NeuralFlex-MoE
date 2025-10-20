"""Inference script for NeuralFlex-MoE"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from transformers import AutoTokenizer
import argparse

from neuraflex_moe.models import NeuralFlexMoE
from neuraflex_moe.config import MODEL_CONFIG
from neuraflex_moe.inference import OptimizedInference
from neuraflex_moe.utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with NeuralFlex-MoE")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling parameter")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger = setup_logger("inference")
    logger.info("Starting NeuralFlex-MoE inference")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    model = NeuralFlexMoE(MODEL_CONFIG)
    
    # Load checkpoint if exists
    if os.path.exists(args.model_path):
        checkpoint = torch.load(os.path.join(args.model_path, "pytorch_model.bin"), map_location=args.device)
        model.load_state_dict(checkpoint, strict=False)
        logger.info("Model loaded successfully")
    else:
        logger.warning(f"Checkpoint not found at {args.model_path}, using random weights")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # Initialize inference engine
    inference_engine = OptimizedInference(
        model=model,
        tokenizer=tokenizer,
        device=args.device
    )
    
    # Generate
    logger.info(f"Generating response for: {args.prompt}")
    result = inference_engine.generate(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    # Print results
    print("\n" + "="*50)
    print("PROMPT:", args.prompt)
    print("="*50)
    print("GENERATED:", result["text"])
    print("="*50)
    print(f"Tokens generated: {result['num_tokens']}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
