# Models Directory

Trained models and checkpoints are saved here.

## Structure

```
models/
├── checkpoints/      # Training checkpoints
├── final/            # Final trained models
├── quantized/        # Quantized models (4-bit, 8-bit)
├── onnx/             # ONNX exported models
└── gguf/             # GGUF format for llama.cpp
```

## Model Formats

### PyTorch (.pt, .pth)
Standard PyTorch format with full model state.

```python
model = torch.load("models/final/model.pt")
```

### SafeTensors (.safetensors)
Safer format, prevents arbitrary code execution.

```python
from safetensors.torch import load_file
state_dict = load_file("models/final/model.safetensors")
```

### Hugging Face Format
Compatible with transformers library.

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("./models/final")
```

## Quantized Models

### 4-bit GPTQ
Best compression, minimal quality loss.
- Size: ~2GB for 7B model
- Speed: 2-3x faster
- VRAM: ~4GB

### 8-bit
Good balance of size and quality.
- Size: ~4GB for 7B model
- Speed: 1.5-2x faster
- VRAM: ~6GB

### ONNX
Cross-platform deployment.
- Optimized for inference
- Works with ONNX Runtime
- CPU and GPU support

### GGUF
For llama.cpp (CPU inference).
- Runs on CPU efficiently
- Multiple quantization levels
- Great for edge devices

## Checkpoint Management

Checkpoints are saved every 1000 steps:
- `checkpoint-1000/`
- `checkpoint-2000/`
- etc.

Keep only the last 3 checkpoints to save space.

## Model Cards

Each model should have a `README.md` with:
- Training details
- Performance metrics
- Usage examples
- Limitations

## Storage Requirements

- Full model (7B): ~14GB
- 4-bit quantized: ~2GB
- 8-bit quantized: ~4GB
- ONNX: ~14GB
- GGUF: ~2-8GB (depends on quantization)

**Recommended**: 50GB free space for training
