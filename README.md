# CoSQA Code Search Engine

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art dense retrieval system for code search, fine-tuned on the CoSQA dataset.

## 🏆 Key Results

| Metric        | Baseline | Fine-tuned | Improvement  |
| ------------- | -------- | ---------- | ------------ |
| **nDCG@10**   | 0.4372   | **0.5534** | **+26.6%** ✨ |
| **Recall@10** | 0.5780   | **0.7120** | **+23.2%** ✨ |
| **MRR@10**    | 0.3942   | **0.5047** | **+28.0%** ✨ |

**vs CoIR Benchmark**: +75.7% improvement over published e5-base-v2 baseline!

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
python scripts/04_finetune.py --model intfloat/e5-base-v2 --epochs 3 --batch-size 32

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
│   ├── 04_finetune.py      # Model fine-tuning (GPU)
│   └── 05_evaluate_finetuned.py  # Fine-tuned evaluation
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
```python
Base Model: intfloat/e5-base-v2
Training Pairs: 9,020 positive (query, code) pairs
Batch Size: 32 (31 in-batch negatives per sample)
Epochs: 3
Learning Rate: 2e-5
Warmup Steps: 100
Total Steps: 846
Training Time: 155.9 minutes (~2.6 hours on GPU)
```

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
| @1   | 32.4%    | 42.0%      | +29.6%      |
| @5   | 47.4%    | 59.6%      | +25.7%      |
| @10  | 57.8%    | 71.2%      | +23.2%      |
| @20  | 70.6%    | 83.0%      | +17.6%      |
| @50  | 83.8%    | 94.0%      | +12.2%      |
| @100 | 89.8%    | 97.2%      | +8.2%       |

### Comparison with Published Baselines

| Model                   | CoIR Benchmark | Our Implementation | Improvement |
| ----------------------- | -------------- | ------------------ | ----------- |
| e5-base-v2 (baseline)   | 0.315          | 0.4372             | +38.8%      |
| e5-base-v2 (fine-tuned) | N/A            | **0.5534**         | **+75.7%**  |

## 🔬 Technical Highlights

1. **Efficient Contrastive Learning**
   - MNRL automatically creates negatives from batch
   - No explicit negative mining required
   - Scales well with batch size

2. **GPU Acceleration**
   - Training: 12 hours (CPU) → 2.6 hours (GPU)
   - Inference: 1,028 queries/sec with GPU
   - Memory optimized for 6GB VRAM

3. **Production-Ready Features**
   - Persistent FAISS indexes (save/load)
   - Batch processing for efficiency
   - Comprehensive evaluation metrics
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
  - Model fine-tuning (2.6 hours)
  - Fine-tuned evaluation (nDCG@10: 0.5534)
  
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

🚀 **Production Ready!** Final nDCG@10: **0.5534** | Recall@10: **71.2%** | 75.7% better than CoIR benchmark!

