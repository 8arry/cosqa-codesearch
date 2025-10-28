"""
Bonus Experiment: Function Names Only vs Full Code Bodies

This script compares retrieval performance when indexing:
1. Only function names (e.g., "sort_list", "add_numbers")
2. Full code bodies (complete function definitions)

This directly answers the bonus question:
"How do the metrics change when you apply the model to function names 
instead of whole bodies?"

Author: CoSQA Team
Date: October 2025
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from tqdm import tqdm

from data.load_cosqa import CoSQADataLoader
from src.engine.faiss_engine import FAISSSearchEngine
from src.evaluation.metrics import calculate_all_metrics


def extract_function_name(code: str) -> str:
    """Extract function name from Python code snippet."""
    # Match: def function_name(...)
    match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
    if match:
        return match.group(1)
    return "unknown_function"  # Fallback for code without clear function def


def main():
    print("\n" + "=" * 80)
    print("BONUS EXPERIMENT: Function Names Only vs Full Code Bodies")
    print("=" * 80)
    print("\nQuestion: How do metrics change when applying the model to")
    print("          function names instead of whole bodies?")
    print("=" * 80)
    
    # Setup paths
    output_dir = project_root / 'results' / 'bonus'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n📁 Loading CoSQA dataset...")
    loader = CoSQADataLoader(cache_dir=str(project_root / 'data' / 'cache'))
    test_df = loader.load_test()
    corpus_list = loader.get_all_corpus()  # Returns list of {'id': ..., 'text': ...}
    
    # Convert to dict for easier manipulation
    corpus_dict = {item['id']: item['text'] for item in corpus_list}
    
    print(f"   Test queries: {len(test_df['query_id'].unique())}")
    print(f"   Corpus size: {len(corpus_dict)}")
    
    # Extract function names from corpus
    print("\n🔍 Extracting function names from code...")
    function_names_corpus = {}
    stats = {'has_name': 0, 'no_name': 0, 'avg_length': []}
    
    for doc_id, code in tqdm(corpus_dict.items(), desc="Processing"):
        func_name = extract_function_name(code)
        function_names_corpus[doc_id] = func_name
        
        if func_name != "unknown_function":
            stats['has_name'] += 1
            stats['avg_length'].append(len(func_name))
        else:
            stats['no_name'] += 1
    
    print(f"\n   Functions with names: {stats['has_name']} ({stats['has_name']/len(corpus_dict)*100:.1f}%)")
    print(f"   Without clear names: {stats['no_name']}")
    if stats['avg_length']:
        print(f"   Average name length: {np.mean(stats['avg_length']):.1f} characters")
    
    # Use fine-tuned model
    model_name = str(project_root / 'models' / 'finetuned')
    if not Path(model_name).exists():
        print(f"\n⚠️  Fine-tuned model not found at {model_name}")
        print("   Using baseline model: intfloat/e5-base-v2")
        model_name = "intfloat/e5-base-v2"
    
    # Configuration A: Index only function names
    print("\n" + "-" * 80)
    print("Configuration A: Indexing ONLY Function Names")
    print("-" * 80)
    print("Example: 'sort_list', 'calculate_sum', 'process_data'")
    
    engine_names_only = FAISSSearchEngine(model_name=model_name)
    docs_names_only = [
        {'id': doc_id, 'text': func_name}
        for doc_id, func_name in function_names_corpus.items()
    ]
    
    print(f"\nIngesting {len(docs_names_only)} function names...")
    engine_names_only.ingest(docs_names_only)
    print("✓ Index built")
    
    # Configuration B: Index full code bodies
    print("\n" + "-" * 80)
    print("Configuration B: Indexing Full Code Bodies")
    print("-" * 80)
    print("Example: 'def sort_list(items):\\n    return sorted(items)'")
    
    engine_full_code = FAISSSearchEngine(model_name=model_name)
    docs_full_code = [
        {'id': doc_id, 'text': code}
        for doc_id, code in corpus_dict.items()
    ]
    
    print(f"\nIngesting {len(docs_full_code)} full code snippets...")
    engine_full_code.ingest(docs_full_code)
    print("✓ Index built")
    
    # Prepare test queries and ground truth
    print("\n" + "-" * 80)
    print("Evaluation Setup")
    print("-" * 80)
    
    queries = test_df['query_text'].tolist()
    query_ids = test_df['query_id'].tolist()
    
    ground_truth = {}
    for _, row in test_df.iterrows():
        qid = row['query_id']
        if row['score'] == 1:
            if qid not in ground_truth:
                ground_truth[qid] = []
            ground_truth[qid].append(row['corpus_id'])
    
    unique_queries = test_df['query_id'].unique()
    print(f"\n   Queries: {len(unique_queries)}")
    print(f"   Total relevance pairs: {sum(len(v) for v in ground_truth.values())}")
    
    # Evaluate Configuration A: Function names only
    print("\n" + "=" * 80)
    print("EVALUATING: Function Names Only")
    print("=" * 80)
    
    results_names_only = engine_names_only.batch_search(queries, top_k=100)
    
    ranks_names = []
    relevance_lists_names = []
    
    for qid, results in zip(query_ids, results_names_only):
        relevant_docs = set(ground_truth.get(qid, []))
        
        rank = None
        relevance_scores = []
        for i, result in enumerate(results, 1):
            doc_id = result['id']
            is_relevant = 1 if doc_id in relevant_docs else 0
            relevance_scores.append(is_relevant)
            if is_relevant and rank is None:
                rank = i
        
        if rank is not None:
            ranks_names.append(rank)
            relevance_lists_names.append(relevance_scores)
    
    metrics_names = calculate_all_metrics(
        ranks=ranks_names,
        relevance_scores_list=relevance_lists_names,
        k_values=[1, 5, 10, 20, 50, 100]
    )
    
    print("\n📊 Results (Function Names Only):")
    print(f"   nDCG@10:   {metrics_names.get('ndcg@10', 0):.4f}")
    print(f"   Recall@10: {metrics_names.get('recall@10', 0):.4f}")
    print(f"   MRR@10:    {metrics_names.get('mrr@10', 0):.4f}")
    print(f"   Recall@1:  {metrics_names.get('recall@1', 0):.4f}")
    
    # Evaluate Configuration B: Full code bodies
    print("\n" + "=" * 80)
    print("EVALUATING: Full Code Bodies")
    print("=" * 80)
    
    results_full_code = engine_full_code.batch_search(queries, top_k=100)
    
    ranks_full = []
    relevance_lists_full = []
    
    for qid, results in zip(query_ids, results_full_code):
        relevant_docs = set(ground_truth.get(qid, []))
        
        rank = None
        relevance_scores = []
        for i, result in enumerate(results, 1):
            doc_id = result['id']
            is_relevant = 1 if doc_id in relevant_docs else 0
            relevance_scores.append(is_relevant)
            if is_relevant and rank is None:
                rank = i
        
        if rank is not None:
            ranks_full.append(rank)
            relevance_lists_full.append(relevance_scores)
    
    metrics_full = calculate_all_metrics(
        ranks=ranks_full,
        relevance_scores_list=relevance_lists_full,
        k_values=[1, 5, 10, 20, 50, 100]
    )
    
    print("\n📊 Results (Full Code Bodies):")
    print(f"   nDCG@10:   {metrics_full.get('ndcg@10', 0):.4f}")
    print(f"   Recall@10: {metrics_full.get('recall@10', 0):.4f}")
    print(f"   MRR@10:    {metrics_full.get('mrr@10', 0):.4f}")
    print(f"   Recall@1:  {metrics_full.get('recall@1', 0):.4f}")
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON: Function Names vs Full Code")
    print("=" * 80)
    
    comparison = {}
    for metric in ['ndcg@10', 'recall@10', 'mrr@10', 'recall@1', 'recall@5', 'recall@20']:
        names_val = metrics_names.get(metric, 0)
        full_val = metrics_full.get(metric, 0)
        diff = full_val - names_val
        pct_change = (diff / names_val * 100) if names_val > 0 else 0
        
        comparison[metric] = {
            'function_names_only': float(names_val),
            'full_code_bodies': float(full_val),
            'difference': float(diff),
            'percent_change': float(pct_change)
        }
        
        print(f"\n{metric.upper()}:")
        print(f"   Function Names Only: {names_val:.4f}")
        print(f"   Full Code Bodies:    {full_val:.4f}")
        print(f"   Difference:          {diff:+.4f} ({pct_change:+.1f}%)")
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    ndcg_improve = comparison['ndcg@10']['percent_change']
    recall_improve = comparison['recall@10']['percent_change']
    
    print(f"\n✅ Full code bodies outperform function names by:")
    print(f"   - nDCG@10:   {ndcg_improve:+.1f}%")
    print(f"   - Recall@10: {recall_improve:+.1f}%")
    
    print(f"\n💡 Interpretation:")
    if ndcg_improve > 50:
        print(f"   HUGE improvement with full code! Function names alone are")
        print(f"   insufficient for semantic code search. The model needs the")
        print(f"   complete context (parameters, logic, comments) to understand")
        print(f"   what the code actually does.")
    elif ndcg_improve > 20:
        print(f"   Significant improvement with full code. Function names provide")
        print(f"   some signal but the function body is crucial for accurate retrieval.")
    else:
        print(f"   Moderate improvement. Function names surprisingly informative,")
        print(f"   but full code still better for precise matching.")
    
    # Save results
    output_file = output_dir / 'experiment_names_vs_bodies.json'
    results_data = {
        'experiment': 'Function Names Only vs Full Code Bodies',
        'description': 'Compare retrieval performance when indexing only function names vs complete code',
        'corpus_stats': {
            'total_documents': len(corpus_dict),
            'functions_with_names': stats['has_name'],
            'average_name_length': float(np.mean(stats['avg_length'])) if stats['avg_length'] else 0
        },
        'metrics': {
            'function_names_only': {k: float(v) for k, v in metrics_names.items()},
            'full_code_bodies': {k: float(v) for k, v in metrics_full.items()}
        },
        'comparison': comparison,
        'key_insight': {
            'ndcg_improvement_percent': float(ndcg_improve),
            'recall_improvement_percent': float(recall_improve),
            'conclusion': 'Full code bodies significantly outperform function names alone' if ndcg_improve > 20 else 'Moderate improvement with full code'
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
