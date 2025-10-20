# NeuralFlex-MoE Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Installation

```bash
# Clone the repository
git clone https://github.com/your-org/neuraflex-moe
cd neuraflex-moe

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Interactive Demo

```bash
python scripts/demo.py
```

This will start an interactive chat session with the model (using random weights for demo).

### Step 3: Train Your First Model

```bash
# Quick training run (small dataset for testing)
python scripts/train.py \
    --output_dir ./my_first_model \
    --model_size 7B \
    --batch_size 2 \
    --total_steps 1000 \
    --eval_steps 100
```

### Step 4: Run Inference

```bash
python scripts/inference.py \
    --model_path ./my_first_model/final \
    --prompt "Explain machine learning in simple terms" \
    --max_tokens 256
```

### Step 5: Start API Server

```bash
# Start the FastAPI server
python -m uvicorn src.neuraflex_moe.api.server:app --reload

# Test the API
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello, world!", "max_tokens": 100}'
```

## 📊 Training Options

### Small Model (3B parameters)
```bash
python scripts/train.py --model_size 3B --batch_size 4
```

### Medium Model (7B parameters)
```bash
python scripts/train.py --model_size 7B --batch_size 2
```

### Large Model (13B parameters)
```bash
python scripts/train.py --model_size 13B --batch_size 1
```

## 🎯 Fine-tuning with LoRA

```bash
python scripts/finetune.py \
    --base_model ./my_first_model/checkpoint-1000 \
    --output_dir ./finetuned_model \
    --lora_r 32 \
    --lora_alpha 64 \
    --num_epochs 3
```

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -f docker/Dockerfile -t neuraflex:latest .

# Run container
docker run -p 8000:8000 --gpus all neuraflex:latest

# Or use docker-compose
docker-compose -f docker/docker-compose.yml up -d
```

## 🧪 Run Tests

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model.py -v

# Run with coverage
pytest --cov=src tests/
```

## 📈 Monitor Training

### Weights & Biases
```bash
# Login to wandb
wandb login

# Training will automatically log to wandb
python scripts/train.py --output_dir ./outputs
```

### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir ./logs

# Open browser to http://localhost:6006
```

## 🔧 Configuration

Edit `configs/neuraflex_7b.yaml` to customize:

```yaml
model:
  num_hidden_layers: 24
  num_experts: 16
  num_experts_per_tok: 2

training:
  learning_rate: 3.0e-4
  batch_size: 2
  total_steps: 100000

novel_features:
  uncertainty_aware_generation:
    enabled: true
    uncertainty_threshold: 0.7
```

## 💡 Tips

1. **Memory Issues?** Enable gradient checkpointing:
   ```python
   model.gradient_checkpointing_enable()
   ```

2. **Slow Training?** Use mixed precision:
   ```bash
   python scripts/train.py --mixed_precision bf16
   ```

3. **Want Faster Inference?** Use quantization:
   ```python
   from neuraflex_moe.inference import ModelQuantizer
   ModelQuantizer.quantize_4bit(model_path, output_path)
   ```

## 📚 Next Steps

- Read the full [README.md](README.md)
- Check out [NeuralFlex-Prompt.md](NeuralFlex-Prompt.md) for architecture details
- Explore [Key-Notes.txt](Key-Notes.txt) for core innovations
- Join our [Discord community](https://discord.gg/neuraflex)

## ❓ Common Issues

### CUDA Out of Memory
- Reduce batch size: `--batch_size 1`
- Enable gradient checkpointing
- Use smaller model variant

### Slow Training
- Enable Flash Attention 2
- Use mixed precision (bf16)
- Increase batch size if memory allows

### Import Errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

## 🆘 Getting Help

- GitHub Issues: [github.com/neuraflex/neuraflex-moe/issues](https://github.com/neuraflex/neuraflex-moe/issues)
- Discord: [discord.gg/neuraflex](https://discord.gg/neuraflex)
- Email: contact@neuraflex.ai

---

**Happy Training! 🎉**
