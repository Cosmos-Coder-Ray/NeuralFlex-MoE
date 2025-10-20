# Dataset & Model Weight Guide for NeuralFlex-MoE

## 🎯 Complete Training Strategy

### **Phase 1: Weight Transfer (Day 1)** ⭐ START HERE

Transfer weights from existing models to bootstrap your architecture.

#### **Recommended Base Model: Mistral-7B**
```bash
python scripts/weight_transfer.py \
    --source_model mistralai/Mistral-7B-v0.1 \
    --target_config configs/neuraflex_7b.yaml \
    --output_dir ./models/neuraflex-init
```

**Why Mistral-7B?**
- ✅ Similar architecture (GQA, RoPE, SwiGLU)
- ✅ Sliding window attention (matches your design)
- ✅ 32K context window
- ✅ Excellent performance baseline
- ✅ Easy weight mapping to MoE layers

**Alternatives:**
- **Qwen2-7B**: Better reasoning, 128K context
- **Phi-3-Mini**: For 3B variant, efficient
- **Llama-2-7B**: Widely supported, stable

---

### **Phase 2: Pretraining (2-4 weeks)**

Continue training on diverse datasets to develop general capabilities.

#### **Dataset Mix (100B-500B tokens)**

| Category | Weight | Datasets | Purpose |
|----------|--------|----------|---------|
| **General Knowledge** | 60% | RedPajama-v2, RefinedWeb, C4, Wikipedia | Language understanding |
| **Code** | 20% | The Stack v2, StarCoder | Programming ability |
| **Math** | 10% | OpenWebMath, MATH, MetaMath | Reasoning skills |
| **Instruction** | 10% | Orca, UltraChat | Following directions |

#### **Quick Start:**
```bash
# Prepare datasets
python scripts/prepare_datasets.py --phase pretrain --max_samples 1000000

# Start pretraining
python scripts/train.py \
    --base_model ./models/neuraflex-init \
    --output_dir ./outputs/pretrain \
    --total_steps 100000 \
    --batch_size 2 \
    --learning_rate 3e-4
```

#### **Key Datasets:**

**1. RedPajama-Data-v2** (30TB, filtered)
```python
dataset = load_dataset("togethercomputer/RedPajama-Data-V2", streaming=True)
```
- High-quality web text
- Deduplicated and filtered
- Multiple domains

**2. RefinedWeb** (5TB)
```python
dataset = load_dataset("tiiuae/falcon-refinedweb", streaming=True)
```
- Cleanest web dataset
- Used by Falcon models
- Excellent quality

**3. The Stack v2** (3TB code)
```python
dataset = load_dataset("bigcode/the-stack-v2", streaming=True)
```
- 600+ programming languages
- Permissive licenses only
- High-quality code

**4. OpenWebMath** (14.7B tokens)
```python
dataset = load_dataset("open-web-math/open-web-math")
```
- Mathematical content
- LaTeX formatted
- Reasoning training

---

### **Phase 3: Fine-tuning (3-7 days)**

Specialize on specific tasks and instruction following.

#### **A. Instruction Following (52K-200K samples)**

**Alpaca-GPT4** (52K, GPT-4 generated)
```python
dataset = load_dataset("vicgalle/alpaca-gpt4")
```
- High-quality instructions
- Diverse tasks
- GPT-4 responses

**OpenAssistant** (161K conversations)
```python
dataset = load_dataset("OpenAssistant/oasst1")
```
- Human-written conversations
- Multi-turn dialogues
- 35 languages

#### **B. Reasoning & Chain-of-Thought (800K-2M samples)**

**CoT Collection** (1.84M examples)
```python
dataset = load_dataset("kaist-ai/CoT-Collection")
```
- Chain-of-thought reasoning
- Multiple reasoning types
- Step-by-step solutions

**Orca-Math** (200K problems)
```python
dataset = load_dataset("microsoft/orca-math-word-problems-200k")
```
- Math word problems
- Detailed explanations
- GPT-4 generated

**PRM800K** (800K process rewards)
```python
dataset = load_dataset("openai/prm800k")
```
- Process reward modeling
- Step-by-step verification
- Math reasoning

#### **C. Code Generation (75K-100K samples)**

**Magicoder** (75K)
```python
dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K")
```
- OSS-Instruct method
- High-quality code
- Diverse problems

**Evol-Instruct-Code** (80K)
```python
dataset = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1")
```
- Evolved instructions
- Complex coding tasks
- Multiple languages

#### **Fine-tuning Command:**
```bash
python scripts/finetune.py \
    --base_model ./outputs/pretrain/checkpoint-100000 \
    --output_dir ./outputs/finetune \
    --datasets alpaca-gpt4,openassistant,cot-collection \
    --num_epochs 3 \
    --learning_rate 1e-5
```

---

### **Phase 4: Novel Feature Training**

#### **For Uncertainty-Aware Generation (UAG)**

**TruthfulQA** (Test hallucination)
```python
dataset = load_dataset("truthful_qa", "generation")
```
- Tests truthfulness
- Identifies hallucinations
- Calibration data

**Training:**
```bash
python scripts/train_uag.py \
    --model ./outputs/finetune/final \
    --dataset truthful_qa \
    --uncertainty_threshold 0.7
```

#### **For Adaptive Reasoning (ARC)**

**GSM8K** (Math reasoning)
```python
dataset = load_dataset("gsm8k", "main")
```

**StrategyQA** (Multi-hop)
```python
dataset = load_dataset("strategy_qa")
```

**Training:**
```bash
python scripts/train_arc.py \
    --model ./outputs/finetune/final \
    --datasets gsm8k,strategy_qa \
    --max_reasoning_steps 5
```

---

## 📊 **Evaluation Benchmarks**

Test your model on standard benchmarks:

```bash
python scripts/evaluate.py \
    --model ./outputs/final \
    --benchmarks mmlu,hellaswag,humaneval,gsm8k
```

### **Target Performance:**

| Benchmark | Target | Dataset |
|-----------|--------|---------|
| MMLU | 75% | `cais/mmlu` |
| HellaSwag | 85% | `hellaswag` |
| HumanEval | 75% | `openai_humaneval` |
| GSM8K | 80% | `gsm8k` |
| TruthfulQA | 65% | `truthful_qa` |
| BBH | 70% | `lukaemon/bbh` |

---

## 💰 **Cost Estimates**

| Phase | Duration | Tokens/Samples | Cost (8xA100) |
|-------|----------|----------------|---------------|
| Weight Transfer | 1 day | - | $50 |
| Pretraining | 2-4 weeks | 100B-500B | $500-2000 |
| Fine-tuning | 3-7 days | 500K-2M | $100-300 |
| Alignment | 2-3 days | 100K-300K | $50-100 |
| **TOTAL** | **1-2 months** | - | **$700-2450** |

---

## 🚀 **Quick Start Commands**

### **1. Prepare Everything:**
```bash
# Download config
cat configs/dataset_config.yaml

# Prepare datasets
python scripts/prepare_datasets.py --phase all

# Transfer weights
python scripts/weight_transfer.py \
    --source_model mistralai/Mistral-7B-v0.1 \
    --output_dir ./models/init
```

### **2. Start Training:**
```bash
# Pretraining
python scripts/train.py \
    --base_model ./models/init \
    --output_dir ./outputs/pretrain \
    --total_steps 100000

# Fine-tuning
python scripts/finetune.py \
    --base_model ./outputs/pretrain/final \
    --output_dir ./outputs/finetune \
    --num_epochs 3

# Evaluation
python scripts/evaluate.py \
    --model ./outputs/finetune/final \
    --benchmarks all
```

---

## 📚 **Dataset Sources Summary**

### **Pretraining (Large Scale)**
- ✅ RedPajama-v2: `togethercomputer/RedPajama-Data-V2`
- ✅ RefinedWeb: `tiiuae/falcon-refinedweb`
- ✅ The Stack v2: `bigcode/the-stack-v2`
- ✅ OpenWebMath: `open-web-math/open-web-math`

### **Fine-tuning (High Quality)**
- ✅ Alpaca-GPT4: `vicgalle/alpaca-gpt4`
- ✅ OpenAssistant: `OpenAssistant/oasst1`
- ✅ CoT Collection: `kaist-ai/CoT-Collection`
- ✅ Orca-Math: `microsoft/orca-math-word-problems-200k`
- ✅ Magicoder: `ise-uiuc/Magicoder-OSS-Instruct-75K`

### **Evaluation (Benchmarks)**
- ✅ MMLU: `cais/mmlu`
- ✅ HellaSwag: `hellaswag`
- ✅ HumanEval: `openai_humaneval`
- ✅ GSM8K: `gsm8k`
- ✅ TruthfulQA: `truthful_qa`

---

## 🎯 **Recommended Path**

1. **Start with Mistral-7B weights** (1 day, $50)
2. **Pretrain on 100B tokens** (2 weeks, $500)
   - 60% RedPajama + RefinedWeb
   - 20% The Stack v2
   - 10% OpenWebMath
   - 10% Orca + UltraChat
3. **Fine-tune on instructions** (5 days, $200)
   - Alpaca-GPT4 + OpenAssistant
   - CoT Collection + Orca-Math
   - Magicoder
4. **Train novel features** (3 days, $100)
   - UAG on TruthfulQA
   - ARC on GSM8K + StrategyQA
5. **Evaluate on benchmarks** (1 day, $50)

**Total: ~1 month, $900**

---

## 📖 **Additional Resources**

- **Dataset Config**: `configs/dataset_config.yaml`
- **Preparation Script**: `scripts/prepare_datasets.py`
- **Training Scripts**: `scripts/train.py`, `scripts/finetune.py`
- **Evaluation**: `scripts/evaluate.py`

---

**Ready to train your revolutionary LLM! 🚀**
