"""Data preparation pipeline"""

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
from typing import Dict, List


class DataPipeline:
    """Multi-source data loading and preprocessing"""
    
    def __init__(self, model_config, tokenizer_name="meta-llama/Llama-2-7b-hf"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = model_config.get("max_position_embeddings", 32768)
        
    def load_datasets(self):
        """Load multiple datasets"""
        datasets_config = {
            "text": ["wikitext", "openwebtext"],
            "code": ["codeparrot/github-code"],
            "math": ["gsm8k"],
        }
        
        all_datasets = []
        
        # Load text datasets
        try:
            wiki = load_dataset("wikitext", "wikitext-103-v1", split="train[:10%]")
            all_datasets.append(wiki)
        except:
            print("Could not load wikitext")
        
        return concatenate_datasets(all_datasets) if all_datasets else None
    
    def tokenize_function(self, examples):
        """Tokenize examples"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
    
    def prepare_dataset(self):
        """Prepare tokenized dataset"""
        dataset = self.load_datasets()
        if dataset is None:
            # Create dummy dataset for testing
            dataset = [{"text": "This is a test sentence."} for _ in range(100)]
            from datasets import Dataset as HFDataset
            dataset = HFDataset.from_list(dataset)
        
        tokenized = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized


class TextDataset(Dataset):
    """Simple text dataset"""
    
    def __init__(self, tokenized_data):
        self.data = tokenized_data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "labels": torch.tensor(item["input_ids"])
        }
