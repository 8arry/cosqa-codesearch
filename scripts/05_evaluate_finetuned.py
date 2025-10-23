"""
Script 5: Evaluate Fine-tuned Model

This script evaluates the fine-tuned model on the CoSQA test set
and compares performance with the baseline model.

Run: python scripts/05_evaluate_finetuned.py --model-dir models/finetuned
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
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model on CoSQA test set")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/finetuned",
        help="Directory containing fine-tuned model (default: models/finetuned)"
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
    parser.add_argument(
        "--baseline-metrics",
        type=str,
        default=None,
        help="Path to baseline metrics JSON for comparison (default: results/baseline_metrics_test.json)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for encoding (default: 64)"
    )
    return parser.parse_args()


def evaluate(engine, test_df, top_k=100):
    """Evaluate search engine on test set."""
    print(f"\nEvaluating on {len(test_df)} queries...")
    print(f"Retrieving top-{top_k} results per query from {engine.get_num_documents():,} documents")
    
    # Prepare queries
    queries = test_df['query_text'].tolist()
    query_ids = test_df['query_id'].tolist()
    relevant_doc_ids = test_df['corpus_id'].tolist()
    
    # Batch search
    print("\nPerforming batch search...")
    start_time = time.time()
    all_results = engine.batch_search(queries, top_k=top_k)
    search_time = time.time() - start_time
    
    print(f"✓ Search completed in {search_time:.2f}s")
    print(f"  Throughput: {len(queries)/search_time:.1f} queries/sec")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
    ranks = []
    relevance_scores_list = []
    
    for i, (query_id, relevant_doc_id, results) in enumerate(zip(query_ids, relevant_doc_ids, all_results)):
        retrieved_ids = [r['id'] for r in results]
        
        if relevant_doc_id in retrieved_ids:
            rank = retrieved_ids.index(relevant_doc_id) + 1
        else:
            rank = float('inf')
        
        ranks.append(rank)
        
        relevance_scores = [1 if doc_id == relevant_doc_id else 0 for doc_id in retrieved_ids]
        relevance_scores_list.append(relevance_scores)
    
    # Calculate metrics at different K values
    k_values = [1, 5, 10, 20, 50, 100]
    metrics = calculate_all_metrics(ranks, relevance_scores_list, k_values=k_values)
    
    # Add additional info
    metrics['total_queries'] = len(queries)
    metrics['search_time_sec'] = search_time
    metrics['queries_per_sec'] = len(queries) / search_time
    
    found_in_100 = sum(1 for r in ranks if r <= 100)
    metrics['found_in_top100'] = found_in_100
    metrics['found_in_top100_pct'] = found_in_100 / len(queries) * 100
    
    return metrics, ranks, all_results


def print_metrics(metrics, baseline_metrics=None):
    """Print metrics in a formatted way with optional baseline comparison."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nDataset Statistics:")
    print(f"  Total queries:        {metrics['total_queries']}")
    print(f"  Search time:          {metrics['search_time_sec']:.2f}s")
    print(f"  Throughput:           {metrics['queries_per_sec']:.1f} queries/sec")
    print(f"  Found in top-100:     {metrics['found_in_top100']} ({metrics['found_in_top100_pct']:.1f}%)")
    
    if baseline_metrics:
        print("\n" + "="*80)
        print("COMPARISON WITH BASELINE")
        print("="*80)
        
        # Print comparison table
        print(f"\n{'Metric':<20} {'Baseline':<12} {'Fine-tuned':<12} {'Δ Absolute':<12} {'Δ Relative':<12}")
        print("-"*80)
        
        for metric in ['recall@1', 'recall@5', 'recall@10', 'recall@20', 'mrr@10', 'ndcg@10']:
            baseline_val = baseline_metrics.get(metric, 0)
            finetuned_val = metrics.get(metric, 0)
            abs_diff = finetuned_val - baseline_val
            rel_diff = (abs_diff / baseline_val * 100) if baseline_val > 0 else 0
            
            diff_symbol = "↑" if abs_diff > 0 else ("↓" if abs_diff < 0 else "→")
            
            print(f"{metric:<20} {baseline_val:<12.4f} {finetuned_val:<12.4f} {diff_symbol} {abs_diff:<10.4f} {rel_diff:>+6.1f}%")
    
    print("\n" + "="*80)
    print("DETAILED METRICS")
    print("="*80)
    
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
    print("STEP 5: Evaluate Fine-tuned Model")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  Model dir:  {args.model_dir}")
    print(f"  Split:      {args.split}")
    print(f"  Top-K:      {args.top_k}")
    
    # Check if fine-tuned model exists
    model_path = Path(args.model_dir)
    if not model_path.exists():
        print(f"\n✗ Error: Fine-tuned model not found at {model_path}")
        print(f"\nPlease run scripts/04_finetune.py first to fine-tune the model.")
        return 1
    
    # Load corpus and build index with fine-tuned model
    print("\n" + "-"*80)
    print("Loading corpus and building index with fine-tuned model...")
    print("-"*80)
    
    loader = CoSQADataLoader(cache_dir=args.cache_dir)
    corpus = loader.get_all_corpus()
    
    print(f"\n✓ Loaded {len(corpus):,} code snippets")
    
    # Initialize engine with fine-tuned model
    print(f"\nInitializing search engine with fine-tuned model...")
    engine = FAISSSearchEngine(
        model_name=str(model_path),
        batch_size=args.batch_size
    )
    
    # Index corpus
    print(f"\nIndexing corpus...")
    start_time = time.time()
    engine.ingest(corpus)
    index_time = time.time() - start_time
    
    print(f"\n✓ Index built in {index_time:.2f}s ({index_time/60:.2f} minutes)")
    print(f"  Documents indexed: {engine.get_num_documents():,}")
    
    # Load test data
    print("\n" + "-"*80)
    print(f"Loading {args.split} data...")
    print("-"*80)
    
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
    
    # Load baseline metrics if available
    baseline_metrics = None
    if args.baseline_metrics:
        baseline_path = Path(args.baseline_metrics)
    else:
        baseline_path = project_root / args.output_dir / f"baseline_metrics_{args.split}.json"
    
    if baseline_path.exists():
        print(f"\nLoading baseline metrics from: {baseline_path}")
        with open(baseline_path, 'r') as f:
            baseline_metrics = json.load(f)
    
    # Print results
    print_metrics(metrics, baseline_metrics)
    
    # Save results
    print("\n" + "-"*80)
    print("Saving results...")
    print("-"*80)
    
    output_dir = project_root / args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    # Save metrics
    metrics_file = output_dir / f"finetuned_metrics_{args.split}.json"
    with open(metrics_file, 'w') as f:
        serializable_metrics = {k: float(v) if isinstance(v, (int, float)) else v 
                               for k, v in metrics.items()}
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"✓ Metrics saved to: {metrics_file}")
    
    # Save detailed results
    results_file = output_dir / f"finetuned_results_{args.split}.jsonl"
    with open(results_file, 'w') as f:
        for i, (idx, row) in enumerate(test_df.iterrows()):
            result_entry = {
                'query_id': row['query_id'],
                'query_text': row['query_text'],
                'relevant_doc_id': row['corpus_id'],
                'rank': int(ranks[i]) if ranks[i] != float('inf') else None,
                'top_10_results': [
                    {'id': r['id'], 'score': float(r['score'])}
                    for r in all_results[i][:10]
                ]
            }
            f.write(json.dumps(result_entry) + '\n')
    
    print(f"✓ Detailed results saved to: {results_file}")
    
    # Save comparison report if baseline exists
    if baseline_metrics:
        comparison = {
            'baseline': {k: baseline_metrics.get(k, 0) for k in ['recall@10', 'mrr@10', 'ndcg@10']},
            'finetuned': {k: metrics.get(k, 0) for k in ['recall@10', 'mrr@10', 'ndcg@10']},
            'improvement': {}
        }
        
        for metric in ['recall@10', 'mrr@10', 'ndcg@10']:
            baseline_val = baseline_metrics.get(metric, 0)
            finetuned_val = metrics.get(metric, 0)
            abs_diff = finetuned_val - baseline_val
            rel_diff = (abs_diff / baseline_val * 100) if baseline_val > 0 else 0
            
            comparison['improvement'][metric] = {
                'absolute': abs_diff,
                'relative_pct': rel_diff
            }
        
        comparison_file = output_dir / f"comparison_{args.split}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"✓ Comparison report saved to: {comparison_file}")
    
    # Summary
    print("\n" + "="*80)
    print("✓ Fine-tuned model evaluation complete!")
    print("="*80)
    
    print(f"\nKey results:")
    print(f"  nDCG@10:    {metrics.get('ndcg@10', 0):.4f}")
    print(f"  Recall@10:  {metrics.get('recall@10', 0):.4f}")
    print(f"  MRR@10:     {metrics.get('mrr@10', 0):.4f}")
    
    if baseline_metrics:
        print(f"\nImprovement over baseline:")
        for metric in ['ndcg@10', 'recall@10', 'mrr@10']:
            baseline_val = baseline_metrics.get(metric, 0)
            finetuned_val = metrics.get(metric, 0)
            abs_diff = finetuned_val - baseline_val
            rel_diff = (abs_diff / baseline_val * 100) if baseline_val > 0 else 0
            symbol = "↑" if abs_diff > 0 else ("↓" if abs_diff < 0 else "→")
            print(f"  {metric:<12} {symbol} {abs_diff:>+7.4f} ({rel_diff:>+6.1f}%)")
    
    print(f"\nResults saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
