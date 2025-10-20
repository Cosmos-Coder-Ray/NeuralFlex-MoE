"""FREE Training Script - High Quality, Zero Cost"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, concatenate_datasets
import argparse

def load_high_quality_datasets(max_samples_per_dataset=None):
    """Load curated high-quality datasets"""
    print("Loading high-quality datasets...")
    
    datasets = []
    
    # 1. Synthetic Textbooks (2.4GB)
    try:
        cosmopedia = load_dataset("HuggingFaceTB/cosmopedia", split="train", streaming=True)
        cosmopedia = cosmopedia.take(500000) if max_samples_per_dataset else cosmopedia
        datasets.append(cosmopedia)
        print("✓ Cosmopedia loaded")
    except: print("✗ Cosmopedia failed")
    
    # 2. Math Reasoning (3.7GB)
    try:
        metamath = load_dataset("meta-math/MetaMathQA", split="train")
        datasets.append(metamath)
        print("✓ MetaMathQA loaded")
    except: print("✗ MetaMathQA failed")
    
    try:
        orca_math = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
        datasets.append(orca_math)
        print("✓ Orca-Math loaded")
    except: print("✗ Orca-Math failed")
    
    # 3. Coding (1.3GB)
    try:
        magicoder = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
        datasets.append(magicoder)
        print("✓ Magicoder loaded")
    except: print("✗ Magicoder failed")
    
    try:
        evol_code = load_dataset("theblackcat102/evol-codealpaca-v1", split="train")
        datasets.append(evol_code)
        print("✓ Evol-CodeAlpaca loaded")
    except: print("✗ Evol-CodeAlpaca failed")
    
    # 4. Instructions (1.25GB)
    try:
        alpaca = load_dataset("vicgalle/alpaca-gpt4", split="train")
        datasets.append(alpaca)
        print("✓ Alpaca-GPT4 loaded")
    except: print("✗ Alpaca-GPT4 failed")
    
    try:
        dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
        datasets.append(dolly)
        print("✓ Dolly-15K loaded")
    except: print("✗ Dolly-15K failed")
    
    # 5. Science (450MB)
    try:
        scienceqa = load_dataset("derek-thomas/ScienceQA", split="train")
        datasets.append(scienceqa)
        print("✓ ScienceQA loaded")
    except: print("✗ ScienceQA failed")
    
    # 6. Medical (550MB)
    try:
        medqa = load_dataset("bigbio/med_qa", split="train")
        datasets.append(medqa)
        print("✓ MedQA loaded")
    except: print("✗ MedQA failed")
    
    # 7. Chain-of-Thought (3GB)
    try:
        cot = load_dataset("kaist-ai/CoT-Collection", split="train[:200000]")
        datasets.append(cot)
        print("✓ CoT-Collection loaded")
    except: print("✗ CoT-Collection failed")
    
    if not datasets:
        print("⚠ No datasets loaded, using dummy data")
        from datasets import Dataset
        dummy = Dataset.from_dict({"text": ["This is a test."] * 1000})
        return dummy
    
    combined = concatenate_datasets(datasets)
    print(f"\n✓ Total samples: {len(combined):,}")
    return combined

def prepare_model_and_tokenizer(model_name="microsoft/Phi-3-mini-4k-instruct"):
    """Load model with 4-bit quantization and LoRA"""
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare for QLoRA
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--output_dir", default="./neuraflex-moe-3b")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()
    
    # Load datasets
    dataset = load_high_quality_datasets()
    
    # Load model
    model, tokenizer = prepare_model_and_tokenizer(args.base_model)
    
    # Tokenize
    def tokenize_function(examples):
        # Handle different dataset formats
        text_key = "text" if "text" in examples else "prompt"
        if text_key not in examples:
            text_key = list(examples.keys())[0]
        
        return tokenizer(
            examples[text_key],
            truncation=True,
            max_length=args.max_length,
            padding="max_length"
        )
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
        push_to_hub=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n🚀 Starting training...")
    trainer.train()
    
    # Save
    print(f"\n💾 Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("\n✅ Training complete!")
    print(f"Model saved to: {args.output_dir}")
    print("\nTo use your model:")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{args.output_dir}')")

if __name__ == "__main__":
    main()
