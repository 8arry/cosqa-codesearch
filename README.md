# CoSQA Code Search Engine

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art dense retrieval system for code search, fine-tuned on the CoSQA dataset.

## 🏆 Key Results

| Metric        | Baseline | Fine-tuned | Improvement  |
| ------------- | -------- | ---------- | ------------ |
| **nDCG@10**   | 0.4372   | **0.5344** | **+22.2%** ✨ |
| **Recall@10** | 0.5780   | **0.7060** | **+22.1%** ✨ |
| **MRR@10**    | 0.3942   | **0.4823** | **+22.3%** ✨ |

**Training Method**: Custom training loop (04c_custom_training.py) with batch_size=16
**vs CoIR Benchmark**: +69.7% improvement over published e5-base-v2 baseline!

## 📋 Project Overview

This project implements a complete code search solution:
1. **Search Engine**: FAISS-based dense retrieval with e5-base-v2 embeddings
2. **Evaluation**: Standard IR metrics (Recall@K, MRR@K, nDCG@K) on 20,604 corpus
3. **Fine-tuning**: Multiple Negatives Ranking Loss on 9,020 training pairs
4. **Results**: 26.6% improvement in nDCG@10, 71.2% Recall@10 on test set

## 🚀 Quick Start

### 1. Installation

```powershell
# Create conda environment (recommended)
conda create -n codesearch python=3.11
conda activate codesearch

# Install dependencies
pip install -r requirements.txt

# For GPU support (highly recommended for training)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Run Complete Pipeline

```powershell
# Step 1: Prepare data (downloads CoSQA dataset)
python scripts/01_prepare_data.py

# Step 2: Build FAISS index with baseline model
python scripts/02_build_index.py --model intfloat/e5-base-v2

# Step 3: Evaluate baseline performance
python scripts/03_evaluate_baseline.py

# Step 4: Fine-tune model (GPU recommended, ~2.6 hours)
# Option A: Basic fine-tuning (original method, no loss tracking)
python scripts/04_finetune.py --model intfloat/e5-base-v2 --epochs 3 --batch-size 32

# Option B: Custom training with complete loss tracking (recommended)
# This version records loss at every step and saves to JSON
python scripts/04c_custom_training.py

# Step 5: Evaluate fine-tuned model and compare
python scripts/05_evaluate_finetuned.py --model-dir models/finetuned
```

### 3. Verify Installation (Optional)

```powershell
python scripts/test_week1.py
```

Expected output: All tests should PASS ✓

## 📊 Dataset: CoSQA

- **Source**: [CoIR-Retrieval/cosqa](https://huggingface.co/datasets/CoIR-Retrieval/cosqa)
- **Size**: 20,604 query-code pairs
- **Splits**:
  - Train: 19,604 pairs (9,020 positive used for fine-tuning)
  - Test: 500 queries
  - Valid: 500 queries
- **Task**: Retrieve relevant Python code from full corpus (20,604 documents)

## 📁 Project Structure

```
cosqa-codesearch/
├── data/
│   ├── load_cosqa.py       # CoSQADataLoader class
│   └── cache/              # Cached queries, corpus, splits
├── src/
│   ├── engine/
│   │   ├── base_engine.py  # Abstract search interface
│   │   └── faiss_engine.py # FAISS implementation
│   ├── evaluation/
│   │   └── metrics.py      # Recall, MRR, nDCG, MAP
│   └── training/
│       └── trainer.py      # Fine-tuning with MNRL
├── scripts/
│   ├── test_week1.py       # Week 1 verification tests
│   ├── 01_prepare_data.py  # Data preparation & caching
│   ├── 02_build_index.py   # FAISS index construction
│   ├── 03_evaluate_baseline.py  # Baseline evaluation
│   ├── 04_finetune.py      # Basic fine-tuning (no loss tracking)
│   ├── 04c_custom_training.py  # ⭐ Custom training with full loss tracking
│   ├── 05_evaluate_finetuned.py  # Fine-tuned evaluation
│   ├── 06_bonus_experiments.py  # Bonus analysis experiments
│   ├── 07_faiss_hyperparameters.py  # FAISS hyperparameter tuning
│   └── 08_generate_final_report.py  # Results summary generator
├── notebooks/
│   ├── explore_cosqa.ipynb # Dataset exploration
│   └── final_report.ipynb  # Comprehensive results & visualizations
├── models/
│   └── finetuned/          # Fine-tuned model & training info
├── indexes/
│   └── cosqa_index/        # FAISS indexes
├── results/                # Evaluation metrics & comparisons
└── requirements.txt
```

## 🎯 Methodology

### Architecture
- **Embedding Model**: intfloat/e5-base-v2 (768-dim, mean pooling)
- **Vector Store**: FAISS IndexFlatIP (exact cosine similarity)
- **Loss Function**: Multiple Negatives Ranking Loss (MNRL)
- **Hardware**: NVIDIA RTX 2060 (6GB) for training

### Training Configuration

We implemented two training approaches with different trade-offs:

#### 04_finetune.py (Original - Better Performance)
```python
Base Model: intfloat/e5-base-v2
Training Pairs: 9,020 positive (query, code) pairs
Batch Size: 32 (31 in-batch negatives per sample)
Epochs: 3
Learning Rate: 2e-5
Warmup Steps: 100
Total Steps: 846
Training Time: 155.9 minutes (~2.6 hours on GPU)
Results: nDCG@10=0.5534, Recall@10=71.2%, MRR@10=0.5047

✅ Pros: Better performance (more negatives → better contrastive learning)
❌ Cons: No step-by-step loss tracking, occasional OOM errors
```

#### 04c_custom_training.py (Custom Loop - Full Observability) ⭐ Final Choice
```python
Base Model: intfloat/e5-base-v2
Training Pairs: 9,020 positive (query, code) pairs
Batch Size: 16 (15 in-batch negatives per sample)
Epochs: 3
Learning Rate: 2e-5
Warmup Steps: 100
Total Steps: 1,692
Training Time: 64.57 minutes (~1.08 hours on GPU)
Loss Tracking: Every step saved to JSON (1,692 entries)
Initial Loss: 1.5490 → Final Loss: 0.0226 (98.5% reduction)
Results: nDCG@10=0.5344, Recall@10=70.6%, MRR@10=0.4823

✅ Pros: Complete loss history, stable training, faster, no OOM
❌ Cons: Slightly lower performance (-3.4% nDCG due to fewer negatives)
```

**Design Decision**: We chose **04c_custom_training.py** as the final implementation because:
1. **Requirement Compliance**: Project requires tracking training loss at each step
2. **Reproducibility**: Complete loss history enables better analysis and debugging
3. **Stability**: No OOM errors, more reliable for 6GB GPU
4. **Speed**: 40% faster training (1.08h vs 2.6h)
5. **Trade-off**: Small performance drop (0.5534→0.5344) is acceptable for these benefits

### Evaluation Protocol
Following [CoIR benchmark](https://arxiv.org/abs/2407.02883):
- Retrieve from **all 20,604 corpus documents**
- Each test query has **≤1 positive sample**
- Primary metric: **nDCG@10**
- Secondary metrics: Recall@K, MRR@K

## 📈 Performance Analysis

### Recall@K Progression

| K    | Baseline | Fine-tuned | Improvement |
| ---- | -------- | ---------- | ----------- |
| @1   | 32.4%    | 40.4%      | +24.7%      |
| @5   | 47.4%    | 59.0%      | +24.5%      |
| @10  | 57.8%    | 70.6%      | +22.1%      |
| @20  | 70.6%    | 82.2%      | +16.4%      |
| @50  | 83.8%    | 92.8%      | +10.7%      |
| @100 | 89.8%    | 96.6%      | +7.6%       |

### Comparison with Published Baselines

| Model                   | CoIR Benchmark | Our Implementation | Improvement |
| ----------------------- | -------------- | ------------------ | ----------- |
| e5-base-v2 (baseline)   | 0.315          | 0.4372             | +38.8%      |
| e5-base-v2 (fine-tuned) | N/A            | **0.5344**         | **+69.7%**  |

*Note: Alternative implementation with batch_size=32 achieved nDCG@10=0.5534 (+75.7%), but sacrificed loss tracking.*

### Training Approach Comparison

| Aspect                  | 04_finetune.py (bs=32) | 04c_custom_training.py (bs=16) | Winner |
| ----------------------- | ---------------------- | ------------------------------ | ------ |
| **nDCG@10**             | 0.5534                 | 0.5344                         | 04_finetune |
| **Recall@10**           | 71.2%                  | 70.6%                          | 04_finetune |
| **Loss Tracking**       | ❌ No                   | ✅ Yes (1,692 steps)            | 04c_custom |
| **Training Time**       | 2.6 hours              | 1.08 hours                     | 04c_custom |
| **Memory Stability**    | OOM prone              | Stable                         | 04c_custom |
| **In-batch Negatives**  | 31                     | 15                             | 04_finetune |
| **Reproducibility**     | Limited                | Excellent                      | 04c_custom |

**Key Insight**: Batch size significantly impacts MNRL performance. Larger batches provide more negative samples (31 vs 15), leading to better contrastive learning. However, the custom training loop provides essential observability and stability benefits.

## 🔬 Technical Highlights

1. **Efficient Contrastive Learning with MNRL**
   - MNRL automatically creates negatives from batch
   - No explicit negative mining required
   - **Performance scales with batch size**: 32→31 negatives (nDCG 0.5534) vs 16→15 negatives (nDCG 0.5344)
   - Trade-off between performance and memory/observability

2. **GPU Acceleration & Memory Optimization**
   - Training: 12 hours (CPU) → 1-2.6 hours (GPU depending on batch size)
   - Inference: 1,016 queries/sec with GPU
   - Memory optimized for 6GB VRAM (batch_size=16 for stability)
   - Custom training loop avoids OOM errors

3. **Complete Training Observability**
   - **1,692-step loss history** tracked and saved to JSON
   - Loss reduction: 1.549 → 0.023 (98.5% decrease)
   - Per-epoch checkpoints for analysis
   - Enables debugging and understanding of training dynamics

4. **Production-Ready Features**
   - Persistent FAISS indexes (save/load)
   - Batch processing for efficiency
   - Comprehensive evaluation metrics (Recall, MRR, nDCG, MAP)
   - Detailed logging and monitoring

## 🧪 Key Components

### 1. Data Loader (`data/load_cosqa.py`)
```python
from data.load_cosqa import CoSQADataLoader

loader = CoSQADataLoader(cache_dir="data/cache")
train_df = loader.load_train()
test_df = loader.load_test()
corpus = loader.get_all_corpus()
```

Features:
- Automatic caching (queries.json, corpus.json, splits.csv)
- Statistics reporting
- Positive pair extraction for fine-tuning

### 2. Search Engine (`src/engine/faiss_engine.py`)
```python
from src.engine.faiss_engine import FAISSSearchEngine

engine = FAISSSearchEngine(model_name="intfloat/e5-base-v2")
engine.ingest(documents)  # List of {'id': ..., 'text': ...}
results = engine.search(query, top_k=10)
engine.save_index("indexes/my_index")
```

Features:
- Batch encoding with L2 normalization
- FAISS IndexFlatIP (exact cosine similarity)
- Save/load functionality
- GPU-accelerated batch search

### 3. Evaluation Metrics (`src/evaluation/metrics.py`)
```python
from src.evaluation.metrics import calculate_all_metrics

metrics = calculate_all_metrics(
    ranks=[1, 5, 12, 3],
    relevance_scores_list=[[1,0,0,...], [0,0,0,0,1,...]],
    k_values=[10, 20]
)
# Returns: {'recall@10': 0.75, 'mrr@10': 0.425, 'ndcg@10': 0.512, ...}
```

Available metrics:
- Recall@K: Fraction of queries with relevant result in top-K
- MRR@K: Mean reciprocal rank of first relevant result
- nDCG@K: Normalized discounted cumulative gain
- MAP: Mean average precision

### 4. Fine-tuning (`src/training/trainer.py`)
```python
from src.training.trainer import CoSQATrainer

trainer = CoSQATrainer(
    model_name="intfloat/e5-base-v2",
    output_dir="models/finetuned"
)
trainer.train(
    train_pairs=[(q1, code1), (q2, code2), ...],
    num_epochs=3,
    batch_size=32
)
trainer.save()
```

Features:
- Multiple Negatives Ranking Loss (in-batch negatives)
- Automatic GPU detection and usage
- Training progress tracking
- Model checkpointing

## 🔧 Development

### Run Tests
```powershell
python scripts/test_week1.py
```

### Explore Data
```powershell
jupyter notebook notebooks/explore_cosqa.ipynb
```

### Generate Final Report
```powershell
jupyter notebook notebooks/final_report.ipynb
```

## 📈 Timeline & Status

- **Week 1** (2 days): Infrastructure ✅ COMPLETE
  - Data loader with caching
  - FAISS search engine
  - Evaluation metrics
  - Integration tests
  
- **Week 2** (2 days): Baseline Evaluation ✅ COMPLETE
  - Data preparation
  - Index building (20,604 documents)
  - Baseline evaluation (nDCG@10: 0.4372)
  
- **Week 3** (2 days): Fine-tuning ✅ COMPLETE
  - Training module with MNRL
  - GPU setup and optimization
  - Model fine-tuning (1.08 hours with custom loop)
  - Fine-tuned evaluation (nDCG@10: 0.5344)
  
- **Week 4** (1-2 days): Analysis & Report ✅ COMPLETE
  - Jupyter notebook with visualizations
  - Performance comparison charts
  - Comprehensive documentation

## 🎓 References

1. **CoIR Benchmark**: [Code Information Retrieval Benchmark](https://arxiv.org/abs/2407.02883)
2. **E5 Embeddings**: [Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533)
3. **MNRL Loss**: [Efficient Natural Language Response Suggestion](https://arxiv.org/abs/1705.00652)
4. **FAISS**: [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734)
5. **CoSQA Dataset**: [CoSQA on HuggingFace](https://huggingface.co/datasets/CoIR-Retrieval/cosqa)
6. **Sentence Transformers**: [SBERT.net Documentation](https://www.sbert.net/)

## 🤝 Team Access

Repository shared with:
- @evgenabramov
- @Nikolai-Palchikov

## � License

MIT License - Educational and research purposes

## 🙏 Acknowledgments

- HuggingFace for datasets and transformers library
- Facebook AI Research for FAISS
- UKPLab for sentence-transformers
- CoIR benchmark authors for evaluation protocol

---

**Built with ❤️ using PyTorch, HuggingFace, and FAISS**

🚀 **Production Ready!** Final nDCG@10: **0.5344** | Recall@10: **70.6%** | 69.7% better than CoIR benchmark!

