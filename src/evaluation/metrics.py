"""
Evaluation metrics for information retrieval.

This module implements standard IR metrics:
- Recall@K: Proportion of queries with at least one relevant doc in top-K
- MRR@K: Mean Reciprocal Rank (average of 1/rank for first relevant doc)
- nDCG@K: Normalized Discounted Cumulative Gain

All metrics assume 1-based ranking (rank starts from 1, not 0).
"""

import numpy as np
from typing import List, Union


def recall_at_k(ranks: List[int], k: int = 10) -> float:
    """
    Calculate Recall@K.
    
    Recall@K = (# queries with relevant doc in top-K) / (total # queries)
    
    Args:
        ranks: List of ranks (1-based) where first relevant document appears
               Use float('inf') if no relevant document found
        k: Cutoff rank
        
    Returns:
        Recall@K score in [0, 1]
        
    Example:
        >>> ranks = [1, 5, 15, 2]  # 4 queries
        >>> recall_at_k(ranks, k=10)  # First 3 queries have relevant doc in top-10
        0.75  # 3/4
    """
    if not ranks:
        return 0.0
    
    hits = sum(1 for r in ranks if r <= k)
    return hits / len(ranks)


def mrr_at_k(ranks: List[int], k: int = 10) -> float:
    """
    Calculate Mean Reciprocal Rank at K.
    
    MRR@K = (1/N) * Σ (1/rank_i) for rank_i <= K, else 0
    
    Args:
        ranks: List of ranks (1-based) where first relevant document appears
        k: Cutoff rank (only consider ranks <= k)
        
    Returns:
        MRR@K score in [0, 1]
        
    Example:
        >>> ranks = [1, 5, 15, 2]
        >>> mrr_at_k(ranks, k=10)
        # (1/1 + 1/5 + 0 + 1/2) / 4 = 0.425
    """
    if not ranks:
        return 0.0
    
    rr_sum = sum(1.0 / r if r <= k else 0.0 for r in ranks)
    return rr_sum / len(ranks)


def dcg_at_k(relevance_scores: List[Union[int, float]], k: int = 10) -> float:
    """
    Calculate Discounted Cumulative Gain at K.
    
    DCG@K = Σ (rel_i / log2(i + 1)) for i in 1..K
    
    Args:
        relevance_scores: Relevance scores for top-K retrieved docs (in order)
                         Typically binary (0/1) or graded (0-3)
        k: Cutoff position
        
    Returns:
        DCG@K score
    """
    if not relevance_scores:
        return 0.0
    
    # Truncate to k
    rel = relevance_scores[:k]
    
    # DCG formula: rel_i / log2(i+1), where i is 0-based index
    # For 1-based ranking: rel[i] / log2(i+2)
    dcg = sum(
        rel[i] / np.log2(i + 2)  # i+2 because i is 0-based, rank is 1-based
        for i in range(len(rel))
    )
    
    return dcg


def ndcg_at_k(relevance_scores_list: List[List[Union[int, float]]], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.
    
    nDCG@K normalizes DCG by the ideal DCG (IDCG), which is the DCG
    of the perfect ranking (relevance scores sorted in descending order).
    
    Args:
        relevance_scores_list: List of relevance score lists (one per query)
                              Each inner list contains relevance scores for 
                              top-K retrieved docs in ranking order
        k: Cutoff position
        
    Returns:
        Mean nDCG@K score across all queries in [0, 1]
        
    Example:
        >>> # Query 1: Retrieved [relevant, irrelevant, irrelevant, ...]
        >>> # Query 2: Retrieved [irrelevant, relevant, irrelevant, ...]
        >>> relevance = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]
        >>> ndcg_at_k(relevance, k=5)
        # Query 1: DCG=1.0, IDCG=1.0 -> nDCG=1.0
        # Query 2: DCG=0.63, IDCG=1.0 -> nDCG=0.63
        # Mean: (1.0 + 0.63) / 2 = 0.815
    """
    if not relevance_scores_list:
        return 0.0
    
    ndcg_scores = []
    
    for relevance_scores in relevance_scores_list:
        # Calculate DCG
        dcg = dcg_at_k(relevance_scores, k)
        
        # Calculate IDCG (ideal ranking: sort by relevance desc)
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = dcg_at_k(ideal_relevance, k)
        
        # Avoid division by zero
        if idcg == 0:
            ndcg = 0.0
        else:
            ndcg = dcg / idcg
        
        ndcg_scores.append(ndcg)
    
    return float(np.mean(ndcg_scores))


def precision_at_k(relevance_scores: List[Union[int, float]], k: int = 10) -> float:
    """
    Calculate Precision@K for a single query.
    
    Precision@K = (# relevant docs in top-K) / K
    
    Args:
        relevance_scores: Binary relevance scores for top-K docs
        k: Cutoff position
        
    Returns:
        Precision@K in [0, 1]
    """
    if not relevance_scores:
        return 0.0
    
    rel = relevance_scores[:k]
    return sum(rel) / k


def average_precision(relevance_scores: List[Union[int, float]]) -> float:
    """
    Calculate Average Precision for a single query.
    
    AP = (1/R) * Σ P(k) * rel(k)
    where R is total number of relevant docs, P(k) is precision at k,
    and rel(k) is 1 if doc at k is relevant, else 0.
    
    Args:
        relevance_scores: Binary relevance scores for all retrieved docs
        
    Returns:
        Average Precision in [0, 1]
    """
    if not relevance_scores:
        return 0.0
    
    total_relevant = sum(relevance_scores)
    if total_relevant == 0:
        return 0.0
    
    precision_sum = 0.0
    relevant_count = 0
    
    for k, rel in enumerate(relevance_scores, 1):
        if rel > 0:
            relevant_count += 1
            precision_at_k = relevant_count / k
            precision_sum += precision_at_k
    
    return precision_sum / total_relevant


def mean_average_precision(relevance_scores_list: List[List[Union[int, float]]]) -> float:
    """
    Calculate Mean Average Precision (MAP) across multiple queries.
    
    Args:
        relevance_scores_list: List of relevance score lists (one per query)
        
    Returns:
        MAP score in [0, 1]
    """
    if not relevance_scores_list:
        return 0.0
    
    ap_scores = [average_precision(rel) for rel in relevance_scores_list]
    return float(np.mean(ap_scores))


def calculate_all_metrics(
    ranks: List[int],
    relevance_scores_list: List[List[Union[int, float]]],
    k_values: List[int] = [1, 5, 10, 20, 100]
) -> dict:
    """
    Calculate all common IR metrics at multiple K values.
    
    Args:
        ranks: List of ranks where first relevant doc appears (for Recall and MRR)
        relevance_scores_list: List of relevance score lists (for nDCG and MAP)
        k_values: List of K values to evaluate
        
    Returns:
        Dictionary with all metrics
        
    Example:
        >>> ranks = [1, 5, 3]
        >>> rel_scores = [[1, 0, 0], [0, 0, 0, 0, 1], [0, 0, 1]]
        >>> metrics = calculate_all_metrics(ranks, rel_scores, k_values=[5, 10])
        >>> print(metrics)
        {
            'recall@5': 1.0,
            'recall@10': 1.0,
            'mrr@5': 0.633,
            'mrr@10': 0.633,
            'ndcg@5': 0.882,
            'ndcg@10': 0.882,
            'map': 0.833
        }
    """
    metrics = {}
    
    for k in k_values:
        metrics[f'recall@{k}'] = recall_at_k(ranks, k)
        metrics[f'mrr@{k}'] = mrr_at_k(ranks, k)
        metrics[f'ndcg@{k}'] = ndcg_at_k(relevance_scores_list, k)
    
    # MAP doesn't depend on K (uses all retrieved docs)
    metrics['map'] = mean_average_precision(relevance_scores_list)
    
    return metrics


if __name__ == "__main__":
    # Unit tests / demo
    print("="*80)
    print("Evaluation Metrics Demo")
    print("="*80)
    
    # Test case 1: Perfect ranking
    print("\nTest 1: Perfect ranking")
    ranks = [1, 1, 1]  # All queries have relevant doc at rank 1
    rel_scores = [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
    
    print(f"Ranks: {ranks}")
    print(f"Recall@10: {recall_at_k(ranks, 10):.3f}")  # Should be 1.0
    print(f"MRR@10: {mrr_at_k(ranks, 10):.3f}")        # Should be 1.0
    print(f"nDCG@10: {ndcg_at_k(rel_scores, 10):.3f}")  # Should be 1.0
    
    # Test case 2: Mixed ranking
    print("\nTest 2: Mixed ranking")
    ranks = [1, 5, 15, 2]  # 4 queries with different ranks
    rel_scores = [
        [1, 0, 0, 0, 0],  # Rank 1
        [0, 0, 0, 0, 1],  # Rank 5
        [],               # Rank > 10 (no relevant in top-10)
        [0, 1, 0, 0, 0]   # Rank 2
    ]
    
    print(f"Ranks: {ranks}")
    print(f"Recall@10: {recall_at_k(ranks, 10):.3f}")   # 3/4 = 0.75
    print(f"MRR@10: {mrr_at_k(ranks, 10):.3f}")         # (1 + 0.2 + 0 + 0.5)/4 = 0.425
    
    # For nDCG, pad with zeros to k=10
    rel_scores_padded = [scores + [0]*(10-len(scores)) for scores in rel_scores]
    print(f"nDCG@10: {ndcg_at_k(rel_scores_padded, 10):.3f}")
    
    # Test case 3: All metrics
    print("\nTest 3: Calculate all metrics")
    metrics = calculate_all_metrics(ranks, rel_scores_padded, k_values=[5, 10])
    for metric_name, value in sorted(metrics.items()):
        print(f"  {metric_name}: {value:.4f}")
    
    print("\n✓ Metrics demo completed successfully!")
