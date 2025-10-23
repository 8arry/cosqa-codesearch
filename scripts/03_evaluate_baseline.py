"""
Script 3: Evaluate Baseline Model

This script evaluates the baseline model on the CoSQA test set.
It retrieves from all 20,604 corpus documents and computes IR metrics.

Run: python scripts/03_evaluate_baseline.py --index-name cosqa_index
"""

import sys
import argparse
from pathlib import Path
import time
import json
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.load_cosqa import CoSQADataLoader
from src.engine.faiss_engine import FAISSSearchEngine
from src.evaluation.metrics import calculate_all_metrics
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseline model on CoSQA test set")
    parser.add_argument(
        "--index-name",
        type=str,
        default="cosqa_index",
        help="Name of the saved index (default: cosqa_index)"
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="indexes",
        help="Directory containing the index (default: indexes)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="Cache directory for data (default: data/cache)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of results to retrieve per query (default: 100)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "valid"],
        help="Which split to evaluate on (default: test)"
    )
    return parser.parse_args()


def evaluate(engine, test_df, top_k=100):
    """
    Evaluate search engine on test set.
    
    Returns:
        dict: Contains metrics and detailed results
    """
    print(f"\nEvaluating on {len(test_df)} queries...")
    print(f"Retrieving top-{top_k} results per query from {engine.get_num_documents():,} documents")
    
    # Prepare queries
    queries = test_df['query_text'].tolist()
    query_ids = test_df['query_id'].tolist()
    relevant_doc_ids = test_df['corpus_id'].tolist()  # Column is 'corpus_id', not 'doc_id'
    
    # Batch search
    print("\nPerforming batch search...")
    start_time = time.time()
    all_results = engine.batch_search(queries, top_k=top_k)
    search_time = time.time() - start_time
    
    print(f"✓ Search completed in {search_time:.2f}s")
    print(f"  Throughput: {len(queries)/search_time:.1f} queries/sec")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
    ranks = []  # Position of first relevant document (1-indexed)
    relevance_scores_list = []  # Binary relevance for each retrieved document
    
    for i, (query_id, relevant_doc_id, results) in enumerate(zip(query_ids, relevant_doc_ids, all_results)):
        # Find rank of relevant document
        retrieved_ids = [r['id'] for r in results]
        
        if relevant_doc_id in retrieved_ids:
            rank = retrieved_ids.index(relevant_doc_id) + 1  # 1-indexed
        else:
            rank = float('inf')  # Not found in top-K
        
        ranks.append(rank)
        
        # Create binary relevance scores
        relevance_scores = [1 if doc_id == relevant_doc_id else 0 for doc_id in retrieved_ids]
        relevance_scores_list.append(relevance_scores)
    
    # Calculate metrics at different K values
    k_values = [1, 5, 10, 20, 50, 100]
    metrics = calculate_all_metrics(ranks, relevance_scores_list, k_values=k_values)
    
    # Add additional info
    metrics['total_queries'] = len(queries)
    metrics['search_time_sec'] = search_time
    metrics['queries_per_sec'] = len(queries) / search_time
    
    # Count how many queries found relevant doc
    found_in_100 = sum(1 for r in ranks if r <= 100)
    metrics['found_in_top100'] = found_in_100
    metrics['found_in_top100_pct'] = found_in_100 / len(queries) * 100
    
    return metrics, ranks, all_results


def print_metrics(metrics):
    """Print metrics in a formatted way."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nDataset Statistics:")
    print(f"  Total queries:        {metrics['total_queries']}")
    print(f"  Search time:          {metrics['search_time_sec']:.2f}s")
    print(f"  Throughput:           {metrics['queries_per_sec']:.1f} queries/sec")
    print(f"  Found in top-100:     {metrics['found_in_top100']} ({metrics['found_in_top100_pct']:.1f}%)")
    
    print(f"\n{'Metric':<20} {'@1':<10} {'@5':<10} {'@10':<10} {'@20':<10} {'@50':<10} {'@100':<10}")
    print("-"*80)
    
    # Recall
    recall_values = [metrics.get(f'recall@{k}', 0) for k in [1, 5, 10, 20, 50, 100]]
    print(f"{'Recall':<20} {recall_values[0]:<10.4f} {recall_values[1]:<10.4f} {recall_values[2]:<10.4f} {recall_values[3]:<10.4f} {recall_values[4]:<10.4f} {recall_values[5]:<10.4f}")
    
    # MRR
    mrr_values = [metrics.get(f'mrr@{k}', 0) for k in [1, 5, 10, 20, 50, 100]]
    print(f"{'MRR':<20} {mrr_values[0]:<10.4f} {mrr_values[1]:<10.4f} {mrr_values[2]:<10.4f} {mrr_values[3]:<10.4f} {mrr_values[4]:<10.4f} {mrr_values[5]:<10.4f}")
    
    # nDCG
    ndcg_values = [metrics.get(f'ndcg@{k}', 0) for k in [1, 5, 10, 20, 50, 100]]
    print(f"{'nDCG':<20} {ndcg_values[0]:<10.4f} {ndcg_values[1]:<10.4f} {ndcg_values[2]:<10.4f} {ndcg_values[3]:<10.4f} {ndcg_values[4]:<10.4f} {ndcg_values[5]:<10.4f}")
    
    # MAP
    if 'map' in metrics:
        print(f"\n{'MAP':<20} {metrics['map']:.4f}")
    
    print("\n" + "="*80)
    print("PRIMARY METRICS (for CoIR benchmark)")
    print("="*80)
    print(f"  Recall@10:  {metrics.get('recall@10', 0):.4f}")
    print(f"  MRR@10:     {metrics.get('mrr@10', 0):.4f}")
    print(f"  nDCG@10:    {metrics.get('ndcg@10', 0):.4f}  ← Primary metric")


def main():
    args = parse_args()
    
    print("="*80)
    print("STEP 3: Evaluate Baseline Model")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  Index:      {args.index_name}")
    print(f"  Split:      {args.split}")
    print(f"  Top-K:      {args.top_k}")
    
    # Load index
    print("\n" + "-"*80)
    print("Loading search index...")
    print("-"*80)
    
    index_path = project_root / args.index_dir / args.index_name
    
    if not index_path.exists():
        print(f"\n✗ Error: Index not found at {index_path}")
        print(f"\nPlease run scripts/02_build_index.py first to build the index.")
        return 1
    
    engine = FAISSSearchEngine()
    engine.load_index(str(index_path))
    
    print(f"\n✓ Index loaded successfully")
    print(f"  Documents: {engine.get_num_documents():,}")
    
    # Load test data
    print("\n" + "-"*80)
    print(f"Loading {args.split} data...")
    print("-"*80)
    
    loader = CoSQADataLoader(cache_dir=args.cache_dir)
    
    if args.split == "test":
        test_df = loader.load_test()
    else:
        test_df = loader.load_valid()
    
    print(f"\n✓ Loaded {len(test_df)} query-document pairs")
    
    # Evaluate
    print("\n" + "-"*80)
    print("Running evaluation...")
    print("-"*80)
    
    metrics, ranks, all_results = evaluate(engine, test_df, top_k=args.top_k)
    
    # Print results
    print_metrics(metrics)
    
    # Save results
    print("\n" + "-"*80)
    print("Saving results...")
    print("-"*80)
    
    output_dir = project_root / args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    # Save metrics
    metrics_file = output_dir / f"baseline_metrics_{args.split}.json"
    with open(metrics_file, 'w') as f:
        # Convert to serializable format
        serializable_metrics = {k: float(v) if isinstance(v, (int, float)) else v 
                               for k, v in metrics.items()}
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"✓ Metrics saved to: {metrics_file}")
    
    # Save detailed results
    results_file = output_dir / f"baseline_results_{args.split}.jsonl"
    with open(results_file, 'w') as f:
        for i, (idx, row) in enumerate(test_df.iterrows()):
            result_entry = {
                'query_id': row['query_id'],
                'query_text': row['query_text'],
                'relevant_doc_id': row['corpus_id'],  # Use 'corpus_id'
                'rank': int(ranks[i]) if ranks[i] != float('inf') else None,
                'top_10_results': [
                    {'id': r['id'], 'score': float(r['score'])}
                    for r in all_results[i][:10]
                ]
            }
            f.write(json.dumps(result_entry) + '\n')
    
    print(f"✓ Detailed results saved to: {results_file}")
    
    # Summary
    print("\n" + "="*80)
    print("✓ Baseline evaluation complete!")
    print("="*80)
    
    print(f"\nKey results:")
    print(f"  nDCG@10:    {metrics.get('ndcg@10', 0):.4f}")
    print(f"  Recall@10:  {metrics.get('recall@10', 0):.4f}")
    print(f"  MRR@10:     {metrics.get('mrr@10', 0):.4f}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"\nNext step: Run scripts/04_finetune.py to fine-tune the model (Week 3)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
