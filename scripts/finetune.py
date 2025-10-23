import argparse
import yaml
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
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
from neuraflex_moe.utils.logging_utils import setup_logger

def main():
    """
    Fine-tuning script for NeuralFlex-MoE using PEFT (LoRA), with resumable training and Google Drive persistence.
    """
    parser = argparse.ArgumentParser(description="Fine-tune NeuralFlex-MoE with LoRA")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the pre-trained base model directory.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the model configuration YAML file.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset for fine-tuning.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained LoRA adapter.")
    parser.add_argument("--gdrive_path", type=str, default=None, help="Base path in Google Drive to save all artifacts.")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Automatically resume from the latest checkpoint.")
    # Add LoRA and training args
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA r parameter")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Micro batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")

    args = parser.parse_args()
    logger = setup_logger()
    output_dir = args.output_dir

    # --- Google Drive Integration ---
    if args.gdrive_path and IS_COLAB:
        logger.info("Mounting Google Drive...")
        drive.mount('/content/drive')
        gdrive_base = os.path.join("/content/drive/MyDrive", args.gdrive_path)
        output_dir = os.path.join(gdrive_base, args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"All artifacts will be saved to: {output_dir}")

    # --- 1. Load Configuration ---
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    model_config = config['model_config']
    training_config = config['training_config']
    training_config.update({
        'num_train_epochs': args.epochs,
        'micro_batch_size': args.batch_size,
        'learning_rate': args.lr,
        'output_dir': output_dir
    })

    # --- 2. Load Base Model and Tokenizer ---
    logger.info(f"Loading base model from: {args.base_model_path}")
    model = NeuralFlexMoE.from_pretrained(args.base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    # --- 3. Apply LoRA Configuration ---
    resume_path = None
    if args.resume_from_checkpoint:
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda p: int(p.split('-')[-1]))
            resume_path = latest_checkpoint
            logger.info(f"Resuming fine-tuning from LoRA adapter checkpoint: {resume_path}")
            model = PeftModel.from_pretrained(model, resume_path)
        else:
            logger.info("No LoRA checkpoint found. Applying new LoRA config.")
            lora_config = LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                bias="none", task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            model = get_peft_model(model, lora_config)
    else:
        lora_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # --- 4. Prepare Dataset ---
    data_pipeline = DataPipeline(model_config, tokenizer_name=args.base_model_path)
    data_collator = data_pipeline.get_data_collator()
    data_path = os.path.join(os.path.dirname(output_dir), "data/cached") if args.gdrive_path else "./data/cached"
    finetune_dataset = data_pipeline.prepare_dataset(dataset_name=args.dataset_name, data_dir=data_path)

    # --- 5. Initialize Trainer ---
    trainer = NeuralFlexTrainer(
        model=model, train_dataset=finetune_dataset, eval_dataset=finetune_dataset,
        tokenizer=tokenizer, config=training_config, data_collator=data_collator, output_dir=output_dir
    )

    # --- 6. Start Fine-Tuning ---
    logger.info("Starting LoRA fine-tuning...")
    trainer.train(resume_from_checkpoint=resume_path)
    logger.info("Fine-tuning complete.")

    # --- 7. Save Final LoRA Adapter ---
    logger.info(f"Saving final LoRA adapter to: {output_dir}/final_adapter")
    model.save_pretrained(os.path.join(output_dir, "final_adapter"))

if __name__ == "__main__":
    main()
