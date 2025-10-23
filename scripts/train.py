import argparse
import yaml
from transformers import AutoTokenizer
import torch
import os
import glob

# Handle Google Drive mounting for Colab
try:
    from google.colab import drive
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

from neuraflex_moe.models.neuraflex_moe import NeuralFlexMoE
from neuraflex_moe.training.data_pipeline import DataPipeline, DataCollatorForMultiModal
from neuraflex_moe.training.trainer import NeuralFlexTrainer
from neuraflex_moe.training.weight_transfer import WeightTransferSystem
from neuraflex_moe.utils.logging_utils import setup_logger
from neuraflex_moe.utils.memory_utils import print_memory_stats

def main():
    """
    Main training script for NeuralFlex-MoE, with support for resumable training and Google Drive persistence.
    """
    parser = argparse.ArgumentParser(description="Train NeuralFlex-MoE model")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the training configuration YAML file.")
    parser.add_argument("--gdrive_path", type=str, default=None, help="Base path in Google Drive to save all artifacts (e.g., 'NeuralFlex-MoE').")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Automatically resume from the latest checkpoint in the output directory.")
    args = parser.parse_args()

    logger = setup_logger()
    output_dir = "./outputs"

    # --- Google Drive Integration ---
    if args.gdrive_path and IS_COLAB:
        logger.info("Mounting Google Drive...")
        drive.mount('/content/drive')
        gdrive_base = os.path.join("/content/drive/MyDrive", args.gdrive_path)
        output_dir = gdrive_base # All outputs will now go to Google Drive
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"All artifacts will be saved to: {output_dir}")

    # --- 1. Load Configuration ---
    logger.info(f"Loading configuration from: {args.config_path}")
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model_config']
    training_config = config['training_config']
    training_config["output_dir"] = os.path.join(output_dir, training_config.get("output_dir", "model_output"))

    # --- 2. Initialize Tokenizer ---
    tokenizer_name = training_config.get("tokenizer_name", "meta-llama/Llama-2-7b-hf")
    logger.info(f"Initializing tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model_config['vocab_size'] != len(tokenizer):
        model_config['vocab_size'] = len(tokenizer)

    # --- 3. Prepare Datasets ---
    logger.info("Setting up data pipeline...")
    data_pipeline = DataPipeline(model_config, tokenizer_name=tokenizer_name)
    data_collator = data_pipeline.get_data_collator()
    
    # Adjust data path if using Google Drive
    data_path = os.path.join(output_dir, "data/cached") if args.gdrive_path else "./data/cached"
    train_dataset = data_pipeline.prepare_dataset(dataset_name=training_config['train_dataset'], data_dir=data_path)
    eval_dataset = data_pipeline.prepare_dataset(dataset_name=training_config['eval_dataset'], data_dir=data_path)
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

    # --- 4. Initialize or Load Model ---
    resume_path = None
    if args.resume_from_checkpoint:
        checkpoint_dir = training_config["output_dir"]
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda p: int(p.split('-')[-1]))
            resume_path = latest_checkpoint
            logger.info(f"Resuming training from checkpoint: {resume_path}")
            model = NeuralFlexMoE.from_pretrained(resume_path)
        else:
            logger.info("No checkpoint found. Starting training from scratch.")
            model = NeuralFlexMoE(model_config)
    else:
        model = NeuralFlexMoE(model_config)

    # --- 5. Optional Weight Transfer (only if not resuming) ---
    if not resume_path and training_config.get("use_weight_transfer", False):
        source_model_id = training_config.get("source_model_id")
        if source_model_id:
            logger.info(f"Performing weight transfer from: {source_model_id}")
            transfer_system = WeightTransferSystem()
            model = transfer_system.transfer_weights(source_model_id, model)
            model = transfer_system.replicate_to_experts(model, num_experts=model_config['num_experts'])
            logger.info("Weight transfer complete.")

    # --- 6. Initialize Trainer ---
    logger.info("Initializing NeuralFlexTrainer...")
    trainer = NeuralFlexTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        config=training_config,
        data_collator=data_collator,
        output_dir=training_config["output_dir"]
    )

    # --- 7. Start Training ---
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_path)
    logger.info("Training complete.")

    # --- 8. Final Save ---
    logger.info("Saving final model...")
    trainer.save_checkpoint(step=trainer.global_step, final=True)
    logger.info(f"Final model saved to {training_config['output_dir']}/final")

if __name__ == "__main__":
    main()
