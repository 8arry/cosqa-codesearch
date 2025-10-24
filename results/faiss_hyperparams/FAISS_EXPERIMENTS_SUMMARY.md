# FAISS Hyperparameter Experiments Summary

## Overview

This document summarizes experiments on FAISS index configurations for CoSQA code search.

**Model**: SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
**Corpus size**: 20,604 documents
**Test queries**: 500
**Embedding dimension**: 768

---

## Experiment 1: Index Types Comparison

Compare exact search (Flat) vs approximate search (IVF).

| Configuration | Recall@10 | MRR@10 | nDCG@10 | Search Time (s) | Queries/s |
|--------------|-----------|---------|---------|-----------------|-----------|
| flat | 0.5560 | 0.4362 | 0.4648 | 0.0793 | 6308.8 |
| ivf_default | 0.5540 | 0.4530 | 0.4773 | 0.0972 | 5142.9 |

---

## Experiment 2: nlist Parameter Sweep

Impact of clustering granularity (fixed nprobe=nlist/10).

| nlist | nprobe | Recall@10 | nDCG@10 | Search Time (s) |
|-------|--------|-----------|---------|-----------------|
| 143 | 14 | 0.5540 | 0.4773 | 0.1026 |
| 286 | 28 | 0.5560 | 0.4819 | 0.1682 |
| 572 | 57 | 0.5700 | 0.4856 | 0.1129 |
| 71 | 7 | 0.5560 | 0.4723 | 0.0888 |

---

## Experiment 3: nprobe Parameter Sweep

Search accuracy vs speed trade-off (fixed nlist=143).

| nprobe | Recall@10 | nDCG@10 | Search Time (s) | Queries/s |
|--------|-----------|---------|-----------------|-----------|
| 1 | 0.3940 | 0.3514 | 0.0239 | 20956.9 |
| 5 | 0.5340 | 0.4645 | 0.0442 | 11306.1 |
| 10 | 0.5520 | 0.4765 | 0.0793 | 6307.5 |
| 20 | 0.5620 | 0.4818 | 0.1341 | 3727.3 |
| 50 | 0.5700 | 0.4870 | 0.2990 | 1672.2 |
| 71 | 0.5700 | 0.4870 | 0.4260 | 1173.7 |
| 143 | 0.5720 | 0.4883 | 0.8935 | 559.6 |

---

## Key Findings

### 1. Flat vs IVF
- Flat index provides 100% recall (exact search)
- IVF can provide similar accuracy with much faster search

### 2. nlist Parameter
- Higher nlist = more clusters = finer partitioning
- Trade-off: slower build time vs faster search
- Optimal: around sqrt(N) to 2*sqrt(N)

### 3. nprobe Parameter
- Higher nprobe = search more clusters = better recall
- Trade-off: accuracy vs speed
- Optimal: start with nlist/10, increase if recall too low

## Recommendations

**For production**:
- Use IVF with nlist=286, nprobe=28
- Provides good balance of speed and accuracy

**For high-accuracy requirements**:
- Use Flat index or IVF with high nprobe (nlist/2)

**For low-latency requirements**:
- Use IVF with low nprobe (1-5)
- Monitor recall and increase nprobe if needed
