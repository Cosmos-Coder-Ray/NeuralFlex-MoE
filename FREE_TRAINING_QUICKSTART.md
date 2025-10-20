# 🆓 FREE Training Quick Start

## 🎯 **Zero Investment Strategy**

Train a **3B parameter model** with **Phi-3 quality** using **15GB of curated datasets** for **$0**.

---

## 📚 **High-Quality Datasets (15GB Total)**

### **Top 10 Must-Have Datasets**

| Dataset | Size | Samples | Quality | Purpose |
|---------|------|---------|---------|---------|
| **Cosmopedia** | 2GB | 30M | ⭐⭐⭐⭐⭐ | Synthetic textbooks |
| **MetaMathQA** | 1.5GB | 395K | ⭐⭐⭐⭐⭐ | Math reasoning |
| **Orca-Math** | 800MB | 200K | ⭐⭐⭐⭐⭐ | GPT-4 math |
| **Magicoder** | 500MB | 75K | ⭐⭐⭐⭐⭐ | Code generation |
| **Alpaca-GPT4** | 200MB | 52K | ⭐⭐⭐⭐⭐ | Instructions |
| **CoT-Collection** | 2GB | 200K | ⭐⭐⭐⭐⭐ | Chain-of-thought |
| **ScienceQA** | 300MB | 21K | ⭐⭐⭐⭐⭐ | Science reasoning |
| **MedQA** | 200MB | 12K | ⭐⭐⭐⭐⭐ | Medical knowledge |
| **Evol-CodeAlpaca** | 300MB | 110K | ⭐⭐⭐⭐⭐ | Advanced coding |
| **Dolly-15K** | 50MB | 15K | ⭐⭐⭐⭐⭐ | Human instructions |

**Total: ~8GB core datasets, ~15GB with all specializations**

---

## 💻 **Free GPU Platforms**

### **Platform Rotation Strategy**

```
Week 1-2: Kaggle (30h/week) → Pretraining
Week 3:   Kaggle (30h/week) → Continued training
Week 4:   Lightning AI (22h/month) → Fine-tuning
Week 5:   Google Colab (12h sessions) → Evaluation
```

| Platform | GPU | VRAM | Free Hours | Sign Up |
|----------|-----|------|------------|---------|
| **Kaggle** | P100/T4 | 16GB | 30h/week | kaggle.com |
| **Lightning AI** | A10G | 24GB | 22h/month | lightning.ai |
| **Google Colab** | T4/A100 | 15-40GB | 12h sessions | colab.google.com |
| **Paperspace** | Free GPU | 8GB | Limited | paperspace.com |

---

## 🚀 **One-Command Training**

### **Kaggle Notebook Setup**

```python
# Cell 1: Install dependencies
!pip install -q transformers accelerate peft bitsandbytes datasets

# Cell 2: Run training
!git clone https://github.com/your-org/neuraflex-moe
%cd neuraflex-moe
!python scripts/train_free.py \
    --base_model microsoft/Phi-3-mini-4k-instruct \
    --output_dir ./neuraflex-3b \
    --num_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation 16
```

---

## 📊 **Expected Results**

| Metric | Target | Training Time | Cost |
|--------|--------|---------------|------|
| **MMLU** | 70% | 4-5 weeks | $0 |
| **GSM8K** | 75% | 4-5 weeks | $0 |
| **HumanEval** | 60% | 4-5 weeks | $0 |
| **Model Size** | 3B params | - | - |
| **Inference** | 200+ tok/s | - | - |
| **VRAM** | 6GB | - | - |

---

## 📥 **Dataset Loading Code**

```python
from datasets import load_dataset, concatenate_datasets

# Core datasets (8GB)
datasets = [
    load_dataset("HuggingFaceTB/cosmopedia", split="train[:500000]"),
    load_dataset("meta-math/MetaMathQA", split="train"),
    load_dataset("microsoft/orca-math-word-problems-200k", split="train"),
    load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train"),
    load_dataset("vicgalle/alpaca-gpt4", split="train"),
    load_dataset("kaist-ai/CoT-Collection", split="train[:200000]"),
]

train_data = concatenate_datasets(datasets)
print(f"Total: {len(train_data):,} samples")
```

---

## 🎓 **Why These Datasets?**

### **1. Cosmopedia** (Mixtral-8x7B generated)
- Synthetic textbooks covering all domains
- Clean, structured, educational content
- Used by top models

### **2. MetaMathQA** (Augmented GSM8K)
- 395K math problems with solutions
- Step-by-step reasoning
- Proven to improve math performance

### **3. Orca-Math** (GPT-4 generated)
- 200K word problems
- Detailed explanations
- Microsoft's quality standard

### **4. Magicoder** (OSS-Instruct method)
- 75K high-quality code samples
- Multiple programming languages
- State-of-the-art code generation

### **5. Alpaca-GPT4** (GPT-4 instructions)
- 52K diverse instructions
- High-quality responses
- Widely used baseline

### **6. CoT-Collection** (1.84M reasoning examples)
- Chain-of-thought reasoning
- Multiple reasoning types
- Improves complex problem solving

---

## 🔧 **Optimization Settings**

```python
# Memory efficient
load_in_4bit=True
gradient_checkpointing=True
per_device_train_batch_size=1
gradient_accumulation_steps=16

# Speed optimized
bf16=True
optim="paged_adamw_8bit"

# LoRA efficient
lora_r=64
lora_alpha=128
```

---

## 📈 **Training Schedule**

```
Day 1:     Setup + Weight transfer (2h)
Week 1-2:  Pretraining on core datasets (60h Kaggle)
Week 3:    Continued pretraining (30h Kaggle)
Week 4:    Specialized fine-tuning (22h Lightning AI)
Week 5:    Evaluation + optimization (12h Colab)

Total: ~126 GPU hours, $0 cost
```

---

## 🌟 **Key Advantages**

✅ **$0 Cost** - 100% free  
✅ **15GB Data** - Small, manageable  
✅ **High Quality** - Curated datasets  
✅ **Phi-3 Approach** - Proven methodology  
✅ **Consumer Hardware** - 6GB VRAM  
✅ **Open Source** - Fully shareable  
✅ **Fast Training** - 4-5 weeks  

---

## 📝 **Complete File List**

- `configs/free_training_config.yaml` - Configuration
- `scripts/train_free.py` - Training script
- `docs/FREE_TRAINING_GUIDE.md` - Detailed guide
- `FREE_TRAINING_QUICKSTART.md` - This file

---

## 🎯 **Start Now**

1. **Sign up**: Kaggle.com (free account)
2. **Create notebook**: New → GPU P100
3. **Clone repo**: `!git clone https://github.com/your-org/neuraflex-moe`
4. **Run**: `!python scripts/train_free.py`
5. **Wait**: 4-5 weeks of training
6. **Deploy**: Share on Hugging Face

---

## 💡 **Pro Tips**

- **Rotate platforms** to maximize free GPU hours
- **Save checkpoints** frequently (every 500 steps)
- **Use streaming** for large datasets
- **Monitor memory** with `nvidia-smi`
- **Test locally** with small samples first

---

**Train a world-class LLM for FREE! 🚀**

Questions? Check `docs/FREE_TRAINING_GUIDE.md` for details.
