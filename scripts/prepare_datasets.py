"""Dataset preparation script for NeuralFlex-MoE"""

import yaml
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
import argparse
from pathlib import Path

def load_config(config_path="configs/dataset_config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

def prepare_pretraining_data(config, tokenizer, max_samples=None):
    """Load and prepare pretraining datasets"""
    datasets = []
    
    print("Loading pretraining datasets...")
    
    # General knowledge
    for ds_config in config['pretraining']['general_knowledge']:
        try:
            print(f"Loading {ds_config['name']}...")
            if ds_config['name'] == 'Wikipedia':
                ds = load_dataset(ds_config['source'], ds_config['config'], split='train')
            else:
                ds = load_dataset(ds_config['source'], split='train', streaming=True)
                ds = ds.take(int(max_samples * ds_config['weight'])) if max_samples else ds
            datasets.append(ds)
        except Exception as e:
            print(f"Warning: Could not load {ds_config['name']}: {e}")
    
    # Code datasets
    for ds_config in config['pretraining']['code']:
        try:
            print(f"Loading {ds_config['name']}...")
            ds = load_dataset(ds_config['source'], split='train', streaming=True)
            ds = ds.take(int(max_samples * ds_config['weight'])) if max_samples else ds
            datasets.append(ds)
        except Exception as e:
            print(f"Warning: Could not load {ds_config['name']}: {e}")
    
    return datasets

def prepare_finetuning_data(config, tokenizer):
    """Load and prepare fine-tuning datasets"""
    datasets = {}
    
    print("Loading fine-tuning datasets...")
    
    # Instruction following
    datasets['instruction'] = []
    for ds_config in config['finetuning']['instruction_following']:
        try:
            print(f"Loading {ds_config['name']}...")
            ds = load_dataset(ds_config['source'], split='train')
            datasets['instruction'].append(ds)
        except Exception as e:
            print(f"Warning: Could not load {ds_config['name']}: {e}")
    
    # Reasoning & CoT
    datasets['reasoning'] = []
    for ds_config in config['finetuning']['reasoning_cot']:
        try:
            print(f"Loading {ds_config['name']}...")
            ds = load_dataset(ds_config['source'], split='train')
            datasets['reasoning'].append(ds)
        except Exception as e:
            print(f"Warning: Could not load {ds_config['name']}: {e}")
    
    # Code generation
    datasets['code'] = []
    for ds_config in config['finetuning']['code_generation']:
        try:
            print(f"Loading {ds_config['name']}...")
            ds = load_dataset(ds_config['source'], split='train')
            datasets['code'].append(ds)
        except Exception as e:
            print(f"Warning: Could not load {ds_config['name']}: {e}")
    
    return datasets

def prepare_evaluation_data(config):
    """Load evaluation benchmarks"""
    benchmarks = {}
    
    print("Loading evaluation benchmarks...")
    
    for bench_config in config['evaluation']:
        try:
            print(f"Loading {bench_config['name']}...")
            if 'config' in bench_config:
                ds = load_dataset(bench_config['source'], bench_config['config'])
            else:
                ds = load_dataset(bench_config['source'])
            benchmarks[bench_config['name']] = ds
        except Exception as e:
            print(f"Warning: Could not load {bench_config['name']}: {e}")
    
    return benchmarks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset_config.yaml")
    parser.add_argument("--phase", choices=['pretrain', 'finetune', 'eval', 'all'], default='all')
    parser.add_argument("--output_dir", default="./data")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['base_model']['primary'])
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare datasets based on phase
    if args.phase in ['pretrain', 'all']:
        pretrain_data = prepare_pretraining_data(config, tokenizer, args.max_samples)
        print(f"Prepared {len(pretrain_data)} pretraining datasets")
    
    if args.phase in ['finetune', 'all']:
        finetune_data = prepare_finetuning_data(config, tokenizer)
        print(f"Prepared fine-tuning datasets: {list(finetune_data.keys())}")
    
    if args.phase in ['eval', 'all']:
        eval_data = prepare_evaluation_data(config)
        print(f"Prepared {len(eval_data)} evaluation benchmarks")
    
    print("\n✅ Dataset preparation complete!")
    print(f"\nRecommended training schedule:")
    for phase, details in config['training_schedule'].items():
        print(f"\n{phase}:")
        for key, value in details.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
