import argparse
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader

from neuraflex_moe.models.neuraflex_moe import NeuralFlexMoE
from neuraflex_moe.evaluation.evaluator import ModelEvaluator
from neuraflex_moe.training.data_pipeline import DataPipeline
from neuraflex_moe.utils.logging_utils import setup_logger

def main():
    """
    Main evaluation script for NeuralFlex-MoE.

    This script handles:
    1. Loading a trained model checkpoint.
    2. Loading the specified evaluation dataset.
    3. Running evaluation and printing key metrics (e.g., perplexity).
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained NeuralFlex-MoE model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint directory (e.g., ./outputs/final)"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the model configuration YAML file used during training (e.g., configs/neuraflex_7b.yaml)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset to evaluate on (e.g., 'wikitext', 'c4', etc.)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation"
    )
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger()

    # --- 1. Load Configuration ---
    logger.info(f"Loading configuration from: {args.config_path}")
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    model_config = config['model_config']

    # --- 2. Load Model and Tokenizer ---
    logger.info(f"Loading model from path: {args.model_path}")
    model = NeuralFlexMoE.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    logger.info(f"Model loaded on device: {device}")

    # --- 3. Prepare Dataset ---
    logger.info(f"Loading and preparing evaluation dataset: {args.dataset_name}")
    data_pipeline = DataPipeline(model_config, tokenizer_name=args.model_path)
    eval_dataset = data_pipeline.prepare_dataset(dataset_name=args.dataset_name)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
    logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

    # --- 4. Initialize Evaluator ---
    logger.info("Initializing ModelEvaluator...")
    evaluator = ModelEvaluator(model, tokenizer, device=device)

    # --- 5. Run Evaluation ---
    logger.info("Starting evaluation...")
    
    # Evaluate perplexity
    perplexity = evaluator.evaluate_perplexity(eval_dataloader)
    logger.info(f"Perplexity on {args.dataset_name}: {perplexity:.4f}")
    
    # Evaluate accuracy
    accuracy = evaluator.evaluate_accuracy(eval_dataloader)
    logger.info(f"Accuracy on {args.dataset_name}: {accuracy:.4f}")

    # You can add other evaluation tasks from lm-evaluation-harness here
    # For example:
    # from lm_eval import evaluator as lm_evaluator
    # from lm_eval.models.huggingface import HFLM
    # results = lm_evaluator.simple_evaluate(
    #     model=HFLM(pretrained=model, tokenizer=tokenizer),
    #     tasks=['arc_challenge', 'hellaswag', 'mmlu'],
    #     num_fewshot=5,
    #     device=device
    # )
    # logger.info(f"lm-harness results: {results}")

    logger.info("Evaluation complete.")

if __name__ == "__main__":
    main()