"""Interactive demo script for NeuralFlex-MoE"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from transformers import AutoTokenizer

from neuraflex_moe.models import NeuralFlexMoE
from neuraflex_moe.config import MODEL_CONFIG
from neuraflex_moe.inference import OptimizedInference
from neuraflex_moe.core import UncertaintyAwareGeneration
from neuraflex_moe.config import NOVEL_FEATURES_CONFIG


def main():
    print("="*60)
    print("NeuralFlex-MoE Interactive Demo")
    print("="*60)
    
    # Initialize model
    print("\n[1/3] Initializing model...")
    model = NeuralFlexMoE(MODEL_CONFIG)
    model.eval()
    
    # Load tokenizer
    print("[2/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize inference engine
    print("[3/3] Setting up inference engine...")
    inference_engine = OptimizedInference(
        model=model,
        tokenizer=tokenizer,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Initialize uncertainty-aware generation
    uag_config = NOVEL_FEATURES_CONFIG["uncertainty_aware_generation"]
    uag_config["base_hidden_size"] = MODEL_CONFIG["base_hidden_size"]
    uag_config["vocab_size"] = MODEL_CONFIG["vocab_size"]
    uag = UncertaintyAwareGeneration(uag_config)
    
    print("\n✓ Model ready! Type 'quit' to exit.\n")
    
    # Interactive loop
    while True:
        try:
            prompt = input("You: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not prompt:
                continue
            
            print("\nGenerating response...")
            
            # Generate with standard inference
            result = inference_engine.generate(
                prompt=prompt,
                max_tokens=256,
                temperature=0.7,
                top_p=0.9
            )
            
            print(f"\nNeuralFlex: {result['text']}")
            print(f"\n[Tokens: {result['num_tokens']}]")
            
            # Show uncertainty if enabled
            if uag.enabled:
                input_ids = tokenizer.encode(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(input_ids)
                    logits = outputs["logits"]
                    hidden_states = torch.zeros(
                        logits.shape[0], logits.shape[1],
                        MODEL_CONFIG["base_hidden_size"]
                    )
                    
                    confidence = uag.compute_confidence(logits, hidden_states)
                    avg_confidence = confidence.mean().item()
                    
                    print(f"[Confidence: {avg_confidence:.2%}]")
                    
                    if avg_confidence < uag.uncertainty_threshold:
                        print("[⚠️  Low confidence - consider alternative responses]")
            
            print("\n" + "-"*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    main()
