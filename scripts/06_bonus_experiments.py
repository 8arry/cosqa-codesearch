"""
Bonus Experiments for CoSQA Code Search

This script runs additional experiments to analyze:
1. Function name vs function body comparison
2. Query type analysis (what/how/why questions)
3. Code complexity impact on retrieval

Author: CoSQA Team
Date: October 2025
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from data.load_cosqa import CoSQADataLoader
from src.engine.faiss_engine import FAISSSearchEngine
from src.evaluation.metrics import calculate_all_metrics


def extract_function_name(code: str) -> str:
    """Extract function name from Python code snippet."""
    # Match: def function_name(...)
    match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
    if match:
        return match.group(1)
    return ""


def remove_function_name(code: str) -> str:
    """Remove function name from code (replace with generic placeholder)."""
    # Replace function name with 'func'
    modified = re.sub(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', r'def func(', code)
    return modified


def analyze_function_names(corpus: Dict[str, str]) -> Dict:
    """Analyze function names in corpus."""
    stats = {
        'total': len(corpus),
        'has_function': 0,
        'no_function': 0,
        'function_names': [],
        'avg_name_length': 0
    }
    
    name_lengths = []
    for doc_id, code in corpus.items():
        func_name = extract_function_name(code)
        if func_name:
            stats['has_function'] += 1
            stats['function_names'].append(func_name)
            name_lengths.append(len(func_name))
        else:
            stats['no_function'] += 1
    
    if name_lengths:
        stats['avg_name_length'] = np.mean(name_lengths)
    
    return stats


def experiment_1_function_name_impact(
    model_name: str,
    test_df: pd.DataFrame,
    corpus: Dict[str, str],
    output_dir: Path
):
    """
    Experiment 1: Function Name vs Function Body
    
    Compares retrieval performance:
    1. With original function names
    2. With function names removed (replaced with 'func')
    
    This helps understand how much the function name contributes to search quality.
    """
    print("\n" + "=" * 80)
    print("Experiment 1: Function Name Impact")
    print("=" * 80)
    
    # Analyze function names first
    print("\n📊 Analyzing function names in corpus...")
    name_stats = analyze_function_names(corpus)
    print(f"Total corpus: {name_stats['total']}")
    print(f"Has function definition: {name_stats['has_function']} ({name_stats['has_function']/name_stats['total']*100:.1f}%)")
    print(f"No function definition: {name_stats['no_function']} ({name_stats['no_function']/name_stats['total']*100:.1f}%)")
    print(f"Average function name length: {name_stats['avg_name_length']:.1f} characters")
    
    # Save function name statistics
    with open(output_dir / 'function_name_stats.json', 'w') as f:
        json.dump({
            'total_corpus': name_stats['total'],
            'has_function': name_stats['has_function'],
            'no_function': name_stats['no_function'],
            'percentage_with_function': name_stats['has_function'] / name_stats['total'] * 100,
            'avg_function_name_length': float(name_stats['avg_name_length']),
            'sample_function_names': name_stats['function_names'][:50]  # First 50 samples
        }, f, indent=2)
    
    print("\n1️⃣ Building index WITH original function names...")
    engine_with_names = FAISSSearchEngine(model_name=model_name)
    docs_with_names = [{'id': k, 'text': v} for k, v in corpus.items()]
    engine_with_names.ingest(docs_with_names)
    
    print("\n2️⃣ Building index WITHOUT function names (replaced with 'func')...")
    engine_without_names = FAISSSearchEngine(model_name=model_name)
    corpus_no_names = {k: remove_function_name(v) for k, v in corpus.items()}
    docs_without_names = [{'id': k, 'text': v} for k, v in corpus_no_names.items()]
    engine_without_names.ingest(docs_without_names)
    
    print("\n🔍 Evaluating both configurations...")
    
    # Prepare test queries
    queries = test_df['query_text'].tolist()
    query_ids = test_df['query_id'].tolist()
    
    # Build ground truth mapping
    ground_truth = {}
    for _, row in test_df.iterrows():
        qid = row['query_id']
        if row['score'] == 1:
            if qid not in ground_truth:
                ground_truth[qid] = []
            ground_truth[qid].append(row['corpus_id'])
    
    # Evaluate WITH function names
    print("\n  Searching WITH function names...")
    all_results_with = engine_with_names.batch_search(queries, top_k=100)
    
    ranks_with = []
    relevance_lists_with = []
    
    for qid, results in zip(query_ids, all_results_with):
        relevant_docs = set(ground_truth.get(qid, []))
        
        # Find rank of first relevant document
        rank = None
        relevance_scores = []
        for i, result in enumerate(results, 1):
            doc_id = result['id']
            is_relevant = 1 if doc_id in relevant_docs else 0
            relevance_scores.append(is_relevant)
            if is_relevant and rank is None:
                rank = i
        
        if rank is not None:
            ranks_with.append(rank)
            relevance_lists_with.append(relevance_scores)
    
    metrics_with = calculate_all_metrics(
        ranks=ranks_with,
        relevance_scores_list=relevance_lists_with,
        k_values=[1, 5, 10, 20, 50, 100]
    )
    
    # Evaluate WITHOUT function names
    print("\n  Searching WITHOUT function names...")
    all_results_without = engine_without_names.batch_search(queries, top_k=100)
    
    ranks_without = []
    relevance_lists_without = []
    
    for qid, results in zip(query_ids, all_results_without):
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
            ranks_without.append(rank)
            relevance_lists_without.append(relevance_scores)
    
    metrics_without = calculate_all_metrics(
        ranks=ranks_without,
        relevance_scores_list=relevance_lists_without,
        k_values=[1, 5, 10, 20, 50, 100]
    )
    
    # Compare results
    print("\n" + "=" * 80)
    print("Results: Function Name Impact")
    print("=" * 80)
    
    comparison = {}
    print(f"\n{'Metric':<15} {'With Names':<12} {'Without Names':<12} {'Difference':<12} {'Impact':<10}")
    print("-" * 70)
    
    key_metrics = ['recall@10', 'mrr@10', 'ndcg@10', 'recall@1', 'recall@100']
    for metric in key_metrics:
        with_val = metrics_with[metric]
        without_val = metrics_without[metric]
        diff = with_val - without_val
        impact_pct = (diff / with_val * 100) if with_val > 0 else 0
        
        comparison[metric] = {
            'with_names': with_val,
            'without_names': without_val,
            'difference': diff,
            'impact_percentage': impact_pct
        }
        
        impact_str = f"{impact_pct:+.1f}%"
        print(f"{metric:<15} {with_val:<12.4f} {without_val:<12.4f} {diff:<+12.4f} {impact_str:<10}")
    
    # Save results
    results = {
        'experiment': 'function_name_impact',
        'description': 'Compare retrieval with and without function names',
        'metrics_with_names': metrics_with,
        'metrics_without_names': metrics_without,
        'comparison': comparison,
        'statistics': {
            'total_queries': len(queries),
            'queries_with_results_with_names': len(ranks_with),
            'queries_with_results_without_names': len(ranks_without)
        }
    }
    
    output_file = output_dir / 'experiment1_function_name_impact.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Key findings
    print("\n🔍 Key Findings:")
    ndcg_impact = comparison['ndcg@10']['impact_percentage']
    if abs(ndcg_impact) < 2:
        print(f"  • Function names have MINIMAL impact on search quality ({ndcg_impact:+.1f}%)")
        print(f"  • Model relies more on function body/semantics than naming")
    elif ndcg_impact < -5:
        print(f"  • Function names HURT search quality ({ndcg_impact:+.1f}%)")
        print(f"  • Removing names improves retrieval (possibly reduces overfitting to names)")
    elif ndcg_impact > 5:
        print(f"  • Function names HELP search quality ({ndcg_impact:+.1f}%)")
        print(f"  • Descriptive names provide valuable semantic signals")
    else:
        print(f"  • Function names have MODERATE impact ({ndcg_impact:+.1f}%)")
    
    return results


def experiment_2_query_type_analysis(
    test_df: pd.DataFrame,
    results_file: Path,
    output_dir: Path
):
    """
    Experiment 2: Query Type Analysis
    
    Categorizes queries by type (what/how/why) and analyzes performance.
    """
    print("\n" + "=" * 80)
    print("Experiment 2: Query Type Analysis")
    print("=" * 80)
    
    # Load baseline or finetuned results
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        evaluation_results = [json.loads(line) for line in f]
    
    # Build results lookup
    results_lookup = {r['query_id']: r for r in evaluation_results}
    
    # Categorize queries
    query_types = defaultdict(list)
    
    for _, row in test_df.iterrows():
        query = row['query_text'].lower()
        qid = row['query_id']
        
        if qid not in results_lookup:
            continue
        
        result = results_lookup[qid]
        
        # Categorize by question type
        if query.startswith(('what', 'which')):
            query_types['what'].append(result)
        elif query.startswith('how'):
            query_types['how'].append(result)
        elif query.startswith('why'):
            query_types['why'].append(result)
        elif '?' in query:
            query_types['other_question'].append(result)
        else:
            query_types['statement'].append(result)
    
    print(f"\n📊 Query Distribution:")
    print("-" * 40)
    for qtype, results in sorted(query_types.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {qtype:<20} {len(results):>4} queries ({len(results)/len(evaluation_results)*100:.1f}%)")
    
    # Analyze performance by query type
    print(f"\n📈 Performance by Query Type:")
    print("-" * 40)
    
    type_analysis = {}
    for qtype, results_list in query_types.items():
        if len(results_list) < 5:  # Skip types with too few samples
            continue
        
        # Calculate metrics for this type
        ranks = [r['rank'] for r in results_list if r['rank'] is not None]
        found = len(ranks)  # If rank exists, it was found
        
        avg_rank = np.mean(ranks) if ranks else 0
        success_rate = found / len(results_list) if results_list else 0
        
        type_analysis[qtype] = {
            'count': len(results_list),
            'found': found,
            'success_rate': success_rate,
            'avg_rank': float(avg_rank),
            'median_rank': float(np.median(ranks)) if ranks else 0
        }
        
        print(f"\n  {qtype}:")
        print(f"    Queries: {len(results_list)}")
        print(f"    Found in top-100: {found} ({success_rate*100:.1f}%)")
        print(f"    Average rank: {avg_rank:.1f}")
        print(f"    Median rank: {np.median(ranks) if ranks else 0:.1f}")
    
    # Save results
    output_file = output_dir / 'experiment2_query_type_analysis.json'
    with open(output_file, 'w') as f:
        json.dump({
            'experiment': 'query_type_analysis',
            'distribution': {k: len(v) for k, v in query_types.items()},
            'analysis': type_analysis
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    return type_analysis


def experiment_3_code_complexity(
    test_df: pd.DataFrame,
    corpus: Dict[str, str],
    results_file: Path,
    output_dir: Path
):
    """
    Experiment 3: Code Complexity Impact
    
    Analyzes how code complexity affects retrieval performance.
    """
    print("\n" + "=" * 80)
    print("Experiment 3: Code Complexity Impact")
    print("=" * 80)
    
    # Load results
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        evaluation_results = [json.loads(line) for line in f]
    
    results_lookup = {r['query_id']: r for r in evaluation_results}
    
    # Calculate complexity metrics for each code
    complexity_data = []
    
    for _, row in test_df.iterrows():
        if row['score'] != 1:  # Only positive pairs
            continue
        
        qid = row['query_id']
        corpus_id = row['corpus_id']
        
        if qid not in results_lookup or corpus_id not in corpus:
            continue
        
        code = corpus[corpus_id]
        result = results_lookup[qid]
        
        # Simple complexity metrics
        lines = code.split('\n')
        num_lines = len([l for l in lines if l.strip()])
        num_chars = len(code)
        num_functions = code.count('def ')
        num_comments = code.count('#') + code.count('"""') + code.count("'''")
        indentation_levels = max([len(l) - len(l.lstrip()) for l in lines if l.strip()], default=0)
        
        # Check if found (rank exists and is not None)
        rank = result.get('rank')
        found = rank is not None
        
        complexity_data.append({
            'query_id': qid,
            'corpus_id': corpus_id,
            'num_lines': num_lines,
            'num_chars': num_chars,
            'num_functions': num_functions,
            'num_comments': num_comments,
            'max_indentation': indentation_levels,
            'rank': rank if found else None,
            'found': found
        })
    
    df = pd.DataFrame(complexity_data)
    
    # Categorize by complexity
    df['complexity_bin'] = pd.cut(df['num_lines'], bins=[0, 5, 10, 20, 100], 
                                   labels=['very_short', 'short', 'medium', 'long'])
    
    print(f"\n📊 Code Complexity Distribution:")
    print("-" * 40)
    for bin_name, group in df.groupby('complexity_bin'):
        print(f"  {bin_name:<15} {len(group):>4} samples ({len(group)/len(df)*100:.1f}%)")
    
    # Analyze performance by complexity
    print(f"\n📈 Performance by Code Complexity:")
    print("-" * 40)
    
    complexity_analysis = {}
    for bin_name, group in df.groupby('complexity_bin'):
        found = group['found'].sum()
        ranks = group[group['rank'].notna()]['rank'].values
        
        complexity_analysis[bin_name] = {
            'count': len(group),
            'avg_lines': float(group['num_lines'].mean()),
            'found': int(found),
            'success_rate': float(found / len(group)),
            'avg_rank': float(np.mean(ranks)) if len(ranks) > 0 else 0,
            'median_rank': float(np.median(ranks)) if len(ranks) > 0 else 0
        }
        
        print(f"\n  {bin_name} ({group['num_lines'].min()}-{group['num_lines'].max()} lines):")
        print(f"    Samples: {len(group)}")
        print(f"    Avg lines: {group['num_lines'].mean():.1f}")
        print(f"    Found: {found}/{len(group)} ({found/len(group)*100:.1f}%)")
        if len(ranks) > 0:
            print(f"    Avg rank: {np.mean(ranks):.1f}")
    
    # Save results
    output_file = output_dir / 'experiment3_code_complexity.json'
    with open(output_file, 'w') as f:
        json.dump({
            'experiment': 'code_complexity_impact',
            'analysis': complexity_analysis,
            'statistics': {
                'total_samples': len(df),
                'avg_lines': float(df['num_lines'].mean()),
                'median_lines': float(df['num_lines'].median()),
                'max_lines': int(df['num_lines'].max())
            }
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    return complexity_analysis


def main():
    parser = argparse.ArgumentParser(description="Run bonus experiments on CoSQA")
    parser.add_argument('--model', type=str, default='intfloat/e5-base-v2',
                       help='Model to use for experiments')
    parser.add_argument('--results-file', type=str, default='results/finetuned_results_test.jsonl',
                       help='Results file for analysis experiments')
    parser.add_argument('--output-dir', type=str, default='results/bonus',
                       help='Output directory for experiment results')
    parser.add_argument('--experiments', type=str, default='1,2,3',
                       help='Comma-separated list of experiments to run (1,2,3)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("CoSQA Bonus Experiments")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    print("\n📦 Loading CoSQA data...")
    loader = CoSQADataLoader(cache_dir="data/cache")
    test_df = loader.load_test()
    corpus_list = loader.get_all_corpus()
    
    # Convert to dict for easier lookup
    corpus = {doc['id']: doc['text'] for doc in corpus_list}
    
    print(f"✓ Loaded {len(test_df)} test samples")
    print(f"✓ Loaded {len(corpus)} corpus documents")
    
    # Parse experiments to run
    experiments_to_run = [int(x.strip()) for x in args.experiments.split(',')]
    
    # Run experiments
    if 1 in experiments_to_run:
        experiment_1_function_name_impact(
            model_name=args.model,
            test_df=test_df,
            corpus=corpus,
            output_dir=output_dir
        )
    
    if 2 in experiments_to_run:
        experiment_2_query_type_analysis(
            test_df=test_df,
            results_file=Path(args.results_file),
            output_dir=output_dir
        )
    
    if 3 in experiments_to_run:
        experiment_3_code_complexity(
            test_df=test_df,
            corpus=corpus,
            results_file=Path(args.results_file),
            output_dir=output_dir
        )
    
    print("\n" + "=" * 80)
    print("✅ All experiments complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
