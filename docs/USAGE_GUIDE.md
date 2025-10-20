# NeuralFlex-MoE Usage Guide

## Table of Contents
1. [Installation](#installation)
2. [Training](#training)
3. [Inference](#inference)
4. [Fine-tuning](#fine-tuning)
5. [API Usage](#api-usage)
6. [Novel Features](#novel-features)
7. [Optimization](#optimization)
8. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites
- Python 3.11+
- CUDA 12.1+ (for GPU support)
- 32GB+ RAM
- 12GB+ VRAM (for 3B model)

### Setup
```bash
git clone https://github.com/your-org/neuraflex-moe
cd neuraflex-moe
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

### Basic Training
```bash
python scripts/train.py \
    --output_dir ./outputs \
    --model_size 7B \
    --batch_size 2 \
    --learning_rate 3e-4 \
    --total_steps 100000
```

### Advanced Training Options
```bash
python scripts/train.py \
    --output_dir ./outputs \
    --model_size 7B \
    --batch_size 2 \
    --learning_rate 3e-4 \
    --total_steps 100000 \
    --eval_steps 500 \
    --save_steps 1000 \
    --resume_from ./outputs/checkpoint-5000
```

### Distributed Training
```bash
accelerate launch --multi_gpu scripts/train.py \
    --output_dir ./outputs \
    --model_size 7B
```

## Inference

### Command Line Inference
```bash
python scripts/inference.py \
    --model_path ./outputs/final \
    --prompt "Explain quantum computing" \
    --max_tokens 512 \
    --temperature 0.7 \
    --top_p 0.9
```

### Python API
```python
from neuraflex_moe.models import NeuralFlexMoE
from neuraflex_moe.inference import OptimizedInference
from transformers import AutoTokenizer

# Load model
model = NeuralFlexMoE.from_pretrained("./outputs/final")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Initialize inference
inference = OptimizedInference(model, tokenizer)

# Generate
result = inference.generate(
    prompt="Hello, world!",
    max_tokens=256,
    temperature=0.7
)
print(result["text"])
```

## Fine-tuning

### LoRA Fine-tuning
```bash
python scripts/finetune.py \
    --base_model ./outputs/checkpoint-10000 \
    --output_dir ./finetuned \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --learning_rate 1e-4 \
    --num_epochs 3
```

### Custom Dataset
```python
from neuraflex_moe.training import DataPipeline

# Create custom data pipeline
pipeline = DataPipeline(model_config)
dataset = pipeline.prepare_dataset()

# Train with custom data
trainer = NeuralFlexTrainer(
    model=model,
    train_dataset=dataset,
    ...
)
trainer.train()
```

## API Usage

### Start Server
```bash
python -m uvicorn src.neuraflex_moe.api.server:app --host 0.0.0.0 --port 8000
```

### REST API Calls

#### Basic Generation
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "What is AI?",
       "max_tokens": 256,
       "temperature": 0.7
     }'
```

#### Uncertainty-Aware Generation
```bash
curl -X POST "http://localhost:8000/generate_with_uncertainty" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Explain relativity",
       "max_tokens": 512
     }'
```

### Python Client
```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "Write a poem about AI",
        "max_tokens": 256,
        "temperature": 0.8
    }
)

print(response.json()["text"])
```

## Novel Features

### Self-Organizing Pathways
```python
from neuraflex_moe.core import SelfOrganizingPathways

sonp = SelfOrganizingPathways(config)
output = sonp.adaptive_routing(input_tensor)

# Prune pathways
pruned = sonp.prune_pathways(model)
print(f"Pruned {pruned} connections")
```

### Temporal Context Compression
```python
from neuraflex_moe.core import TemporalContextCompressor

tcc = TemporalContextCompressor(config)
compressed, retrieved = tcc(
    current_context,
    compress=True,
    retrieve=True
)
```

### Uncertainty-Aware Generation
```python
from neuraflex_moe.core import UncertaintyAwareGeneration

uag = UncertaintyAwareGeneration(config)
result = uag.generate_with_confidence(
    model=model,
    input_ids=input_ids,
    max_length=512
)

print(f"Confidence: {result['confidence']:.2%}")
if result['uncertainty_flag']:
    print("Low confidence - alternatives available")
```

### Continuous Learning
```python
from neuraflex_moe.core import ContinuousLearningModule

clm = ContinuousLearningModule(config)

# Add interaction
clm.add_interaction(
    input_ids=input_ids,
    output_ids=output_ids,
    feedback_score=0.9
)

# Online learning step
losses = clm.online_learning_step(model, optimizer)
```

## Optimization

### Memory Optimization
```python
from neuraflex_moe.utils import optimize_memory

# Optimize memory usage
optimize_memory(model)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
```

### Quantization
```python
from neuraflex_moe.inference import ModelQuantizer

# 4-bit quantization
ModelQuantizer.quantize_4bit(
    model_path="./outputs/final",
    output_path="./outputs/quantized_4bit"
)

# 8-bit quantization
ModelQuantizer.quantize_8bit(
    model_path="./outputs/final",
    output_path="./outputs/quantized_8bit"
)
```

### Batch Inference
```python
prompts = [
    "What is machine learning?",
    "Explain neural networks",
    "What is deep learning?"
]

results = inference.batch_generate(
    prompts=prompts,
    max_tokens=256
)

for result in results:
    print(result["text"])
```

## Troubleshooting

### Out of Memory
```python
# Reduce batch size
--batch_size 1

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use smaller model
--model_size 3B
```

### Slow Training
```python
# Enable mixed precision
--mixed_precision bf16

# Use Flash Attention
# (automatically enabled if available)

# Increase batch size
--batch_size 4
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python version
python --version  # Should be 3.11+
```

### CUDA Errors
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"
```

## Advanced Usage

### Custom Configuration
```python
from neuraflex_moe.config import MODEL_CONFIG

# Modify config
custom_config = MODEL_CONFIG.copy()
custom_config["num_hidden_layers"] = 32
custom_config["num_experts"] = 32

# Create model with custom config
model = NeuralFlexMoE(custom_config)
```

### Multi-GPU Training
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    scripts/train.py
```

### Evaluation
```bash
python scripts/evaluate.py \
    --model_path ./outputs/final \
    --device cuda \
    --batch_size 4
```

## Best Practices

1. **Start Small**: Begin with 3B model for testing
2. **Monitor Memory**: Use `nvidia-smi` to track GPU usage
3. **Save Frequently**: Set `--save_steps` appropriately
4. **Use Checkpointing**: Enable gradient checkpointing for large models
5. **Log Everything**: Use wandb or tensorboard for monitoring
6. **Test First**: Run on small dataset before full training
7. **Backup Models**: Save checkpoints to multiple locations

## Performance Tips

- Use BF16 mixed precision for 2x speedup
- Enable Flash Attention 2 for memory efficiency
- Use gradient accumulation for larger effective batch size
- Quantize models for faster inference
- Use KV caching for sequential generation
- Enable xFormers for additional optimizations

---

For more information, see [README.md](README.md) and [QUICKSTART.md](QUICKSTART.md)
