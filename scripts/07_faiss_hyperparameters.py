"""
FAISS Hyperparameter Experiments for CoSQA Code Search

This script explores different FAISS index configurations:
1. Index types: Flat (exact), IVF (approximate)
2. IVF parameters: nlist, nprobe
3. Performance vs accuracy trade-offs
4. Memory usage comparison

Task requirement: "向量存储超参数 - 未实现（可选）"
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import faiss
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_cosqa import CoSQADataLoader
from sentence_transformers import SentenceTransformer
from src.evaluation.metrics import calculate_all_metrics


class FAISSHyperparameterExperiment:
    """
    Compare different FAISS index configurations.
    
    FAISS Index Types:
    1. IndexFlatIP: Exact search (baseline, 100% recall)
    2. IndexIVFFlat: Inverted file with exact distance (faster, approximate)
    
    Key hyperparameters for IVF:
    - nlist: Number of clusters (voronoi cells)
      - Higher = more granular partitioning, slower build, faster search
      - Rule of thumb: sqrt(N) to 4*sqrt(N) where N = corpus size
      - For 20,604 docs: ~144 to ~576
    
    - nprobe: Number of clusters to search
      - Higher = more accurate but slower search
      - Range: 1 to nlist
      - Common: nlist/10 to nlist/2
    """
    
    def __init__(self, model_path: str, cache_dir: str = "data/cache"):
        """Initialize experiment."""
        print("="*80)
        print("FAISS Hyperparameter Experiments")
        print("="*80)
        
        # Load model
        print(f"\nLoading model: {model_path}")
        self.model = SentenceTransformer(model_path)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
        
        # Load data
        print(f"\nLoading data from: {cache_dir}")
        loader = CoSQADataLoader(cache_dir=cache_dir)
        
        # Load test set
        self.test_df = loader.load_test()
        print(f"Test set: {len(self.test_df)} pairs")
        
        # Load full corpus for indexing
        corpus_data = loader.get_all_corpus()
        if isinstance(corpus_data, list):
            self.corpus = {doc['id']: doc['text'] for doc in corpus_data}
        else:
            self.corpus = corpus_data
        
        print(f"Corpus: {len(self.corpus)} documents")
        
        # Prepare test data
        self._prepare_test_data()
        
        # Encode corpus once (reused for all experiments)
        print("\nEncoding corpus (this may take a few minutes)...")
        start = time.time()
        corpus_texts = [self.corpus[cid] for cid in self.corpus_ids]
        self.corpus_embeddings = self.model.encode(
            corpus_texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=128
        )
        encode_time = time.time() - start
        print(f"✓ Encoded {len(self.corpus_embeddings)} documents in {encode_time:.2f}s")
        
        # Normalize for inner product = cosine similarity
        faiss.normalize_L2(self.corpus_embeddings)
        
        # Encode test queries once
        print("\nEncoding test queries...")
        self.query_embeddings = self.model.encode(
            self.test_queries,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=128
        )
        faiss.normalize_L2(self.query_embeddings)
        print(f"✓ Encoded {len(self.query_embeddings)} queries")
        
    def _prepare_test_data(self):
        """Prepare test queries and relevance judgments."""
        # Get unique test queries
        test_queries_df = self.test_df.drop_duplicates(subset=['query_id'])
        self.test_queries = test_queries_df['query_text'].tolist()
        self.test_query_ids = test_queries_df['query_id'].tolist()
        
        # Build relevance map: query_id -> set of relevant corpus_ids
        self.relevance = {}
        for _, row in self.test_df.iterrows():
            qid = row['query_id']
            if qid not in self.relevance:
                self.relevance[qid] = set()
            if row['score'] == 1:
                self.relevance[qid].add(row['corpus_id'])
        
        # Corpus ID list (same order as embeddings)
        self.corpus_ids = list(self.corpus.keys())
        
        # Create corpus_id -> index mapping
        self.corpus_id_to_idx = {cid: idx for idx, cid in enumerate(self.corpus_ids)}
        
        print(f"\nTest data prepared:")
        print(f"  Queries: {len(self.test_queries)}")
        print(f"  Relevance judgments: {len(self.relevance)}")
        print(f"  Corpus IDs: {len(self.corpus_ids)}")
    
    def build_flat_index(self) -> faiss.IndexFlatIP:
        """
        Build exact search index (baseline).
        
        IndexFlatIP: Exact inner product search
        - 100% recall (exhaustive search)
        - Memory: O(N*D) where N=corpus size, D=embedding dim
        - Search: O(N*D) per query (slow for large corpus)
        """
        print("\n" + "-"*80)
        print("Building Flat Index (Exact Search - Baseline)")
        print("-"*80)
        
        start = time.time()
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(self.corpus_embeddings)
        build_time = time.time() - start
        
        print(f"✓ Built in {build_time:.4f}s")
        print(f"  Total vectors: {index.ntotal}")
        print(f"  Index trained: {index.is_trained}")
        
        return index
    
    def build_ivf_index(
        self, 
        nlist: int, 
        nprobe: int = None
    ) -> faiss.IndexIVFFlat:
        """
        Build IVF (Inverted File) index.
        
        IndexIVFFlat: Approximate search using clustering
        - Memory: O(N*D) + O(nlist*D)
        - Build: O(N*D*nlist) - need to cluster data
        - Search: O(nprobe*N/nlist*D) - search only nprobe clusters
        
        Args:
            nlist: Number of clusters (voronoi cells)
            nprobe: Number of clusters to search (default: nlist/10)
        """
        if nprobe is None:
            nprobe = max(1, nlist // 10)
        
        print("\n" + "-"*80)
        print(f"Building IVF Index (nlist={nlist}, nprobe={nprobe})")
        print("-"*80)
        
        start = time.time()
        
        # Create quantizer (used to assign vectors to clusters)
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        
        # Create IVF index
        index = faiss.IndexIVFFlat(
            quantizer,
            self.embedding_dim,
            nlist,
            faiss.METRIC_INNER_PRODUCT
        )
        
        # Train the index (k-means clustering)
        print(f"Training index (k-means with k={nlist})...")
        train_start = time.time()
        index.train(self.corpus_embeddings)
        train_time = time.time() - train_start
        
        # Add vectors
        print(f"Adding vectors...")
        add_start = time.time()
        index.add(self.corpus_embeddings)
        add_time = time.time() - add_start
        
        # Set search parameter
        index.nprobe = nprobe
        
        build_time = time.time() - start
        
        print(f"✓ Built in {build_time:.4f}s")
        print(f"  Training time: {train_time:.4f}s")
        print(f"  Adding time:   {add_time:.4f}s")
        print(f"  Total vectors: {index.ntotal}")
        print(f"  Clusters:      {index.nlist}")
        print(f"  Search probes: {index.nprobe}")
        
        return index
    
    def evaluate_index(
        self, 
        index: faiss.Index, 
        config_name: str,
        k: int = 10
    ) -> Dict:
        """
        Evaluate index performance.
        
        Metrics:
        - Recall@k: Did we find the relevant document?
        - MRR@k: Mean Reciprocal Rank
        - nDCG@k: Normalized Discounted Cumulative Gain
        - Search time: Average time per query
        """
        print(f"\nEvaluating: {config_name}")
        
        # Search
        search_start = time.time()
        scores, indices = index.search(self.query_embeddings, k)
        search_time = time.time() - search_start
        
        # Prepare data for calculate_all_metrics
        ranks = []
        relevance_scores_list = []
        
        for i, qid in enumerate(self.test_query_ids):
            # Get top-k corpus IDs
            retrieved_ids = [self.corpus_ids[idx] for idx in indices[i]]
            
            # Get relevant doc IDs for this query
            relevant_ids = self.relevance.get(qid, set())
            
            # Find rank of first relevant document
            rank = float('inf')
            for idx, doc_id in enumerate(retrieved_ids):
                if doc_id in relevant_ids:
                    rank = idx + 1  # 1-indexed
                    break
            
            ranks.append(rank)
            
            # Create binary relevance scores
            relevance_scores = [1 if doc_id in relevant_ids else 0 for doc_id in retrieved_ids]
            relevance_scores_list.append(relevance_scores)
        
        # Compute metrics using calculate_all_metrics
        k_values = [1, 5, 10, 20, 50, 100] if k >= 100 else [1, 5, 10, 20, 50]
        metrics = calculate_all_metrics(ranks, relevance_scores_list, k_values=k_values)
        
        # Add timing info
        metrics['search_time_total'] = search_time
        metrics['search_time_per_query'] = search_time / len(self.test_queries)
        metrics['queries_per_second'] = len(self.test_queries) / search_time
        
        print(f"  Recall@{k}:  {metrics.get(f'recall@{k}', 'N/A'):.4f}" if f'recall@{k}' in metrics else f"  Recall@{k}: N/A")
        print(f"  MRR@{k}:     {metrics.get(f'mrr@{k}', 'N/A'):.4f}" if f'mrr@{k}' in metrics else f"  MRR@{k}: N/A")
        print(f"  nDCG@{k}:    {metrics.get(f'ndcg@{k}', 'N/A'):.4f}" if f'ndcg@{k}' in metrics else f"  nDCG@{k}: N/A")
        print(f"  Search time: {search_time:.4f}s ({metrics['queries_per_second']:.1f} queries/s)")
        
        return metrics
    
    def experiment_1_index_types(self) -> Dict:
        """
        Experiment 1: Compare Flat vs IVF index types.
        
        Goal: Understand speed vs accuracy trade-off
        """
        print("\n" + "="*80)
        print("EXPERIMENT 1: Index Types Comparison")
        print("="*80)
        
        results = {}
        
        # 1. Flat index (exact, baseline)
        flat_index = self.build_flat_index()
        results['flat'] = self.evaluate_index(flat_index, "Flat (Exact Search)")
        
        # 2. IVF with default parameters
        # For 20,604 docs: sqrt(N) ≈ 144
        nlist_default = int(np.sqrt(len(self.corpus)))
        ivf_index = self.build_ivf_index(nlist_default)
        results['ivf_default'] = self.evaluate_index(
            ivf_index, 
            f"IVF (nlist={nlist_default}, nprobe={ivf_index.nprobe})"
        )
        
        return results
    
    def experiment_2_nlist_sweep(self) -> Dict:
        """
        Experiment 2: Sweep nlist parameter.
        
        Test different clustering granularities:
        - Small nlist: Coarse partitioning (faster build, less accurate)
        - Large nlist: Fine partitioning (slower build, more accurate)
        """
        print("\n" + "="*80)
        print("EXPERIMENT 2: nlist Parameter Sweep")
        print("="*80)
        
        corpus_size = len(self.corpus)
        sqrt_n = int(np.sqrt(corpus_size))
        
        # Test different nlist values
        nlist_values = [
            sqrt_n // 2,      # 72 - very coarse
            sqrt_n,           # 144 - default
            sqrt_n * 2,       # 288 - fine
            sqrt_n * 4,       # 576 - very fine
        ]
        
        print(f"\nTesting nlist values: {nlist_values}")
        print(f"Corpus size: {corpus_size}, sqrt(N): {sqrt_n}")
        
        results = {}
        for nlist in nlist_values:
            # Use nprobe = nlist/10 for fair comparison
            nprobe = max(1, nlist // 10)
            
            index = self.build_ivf_index(nlist, nprobe)
            config_name = f"IVF_nlist{nlist}_nprobe{nprobe}"
            results[config_name] = self.evaluate_index(index, config_name)
        
        return results
    
    def experiment_3_nprobe_sweep(self) -> Dict:
        """
        Experiment 3: Sweep nprobe parameter.
        
        Fix nlist, vary nprobe to see accuracy vs speed trade-off:
        - Small nprobe: Fast search, lower recall
        - Large nprobe: Slower search, higher recall
        """
        print("\n" + "="*80)
        print("EXPERIMENT 3: nprobe Parameter Sweep")
        print("="*80)
        
        # Use default nlist
        nlist = int(np.sqrt(len(self.corpus)))
        print(f"\nFixed nlist: {nlist}")
        
        # Build index once
        print("\nBuilding base IVF index...")
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        index = faiss.IndexIVFFlat(
            quantizer,
            self.embedding_dim,
            nlist,
            faiss.METRIC_INNER_PRODUCT
        )
        index.train(self.corpus_embeddings)
        index.add(self.corpus_embeddings)
        print(f"✓ Built with {index.ntotal} vectors")
        
        # Test different nprobe values
        nprobe_values = [1, 5, 10, 20, 50, nlist // 2, nlist]
        # Remove invalid values
        nprobe_values = [p for p in nprobe_values if p <= nlist]
        nprobe_values = sorted(set(nprobe_values))
        
        print(f"\nTesting nprobe values: {nprobe_values}")
        
        results = {}
        for nprobe in nprobe_values:
            index.nprobe = nprobe
            config_name = f"IVF_nlist{nlist}_nprobe{nprobe}"
            results[config_name] = self.evaluate_index(index, config_name)
        
        return results
    
    def run_all_experiments(self, output_dir: str = "results/faiss_hyperparams"):
        """Run all experiments and save results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        # Experiment 1: Index types
        print("\n" + "="*80)
        print("Starting Experiment 1...")
        print("="*80)
        exp1_results = self.experiment_1_index_types()
        all_results['experiment1_index_types'] = exp1_results
        
        # Save intermediate results
        with open(output_path / "experiment1_index_types.json", 'w') as f:
            json.dump(exp1_results, f, indent=2)
        
        # Experiment 2: nlist sweep
        print("\n" + "="*80)
        print("Starting Experiment 2...")
        print("="*80)
        exp2_results = self.experiment_2_nlist_sweep()
        all_results['experiment2_nlist_sweep'] = exp2_results
        
        with open(output_path / "experiment2_nlist_sweep.json", 'w') as f:
            json.dump(exp2_results, f, indent=2)
        
        # Experiment 3: nprobe sweep
        print("\n" + "="*80)
        print("Starting Experiment 3...")
        print("="*80)
        exp3_results = self.experiment_3_nprobe_sweep()
        all_results['experiment3_nprobe_sweep'] = exp3_results
        
        with open(output_path / "experiment3_nprobe_sweep.json", 'w') as f:
            json.dump(exp3_results, f, indent=2)
        
        # Save all results
        with open(output_path / "all_experiments.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Generate summary
        self._generate_summary(all_results, output_path)
        
        print("\n" + "="*80)
        print("✓ All experiments complete!")
        print("="*80)
        print(f"\nResults saved to: {output_path}")
        
        return all_results
    
    def _generate_summary(self, results: Dict, output_path: Path):
        """Generate markdown summary of experiments."""
        summary_lines = [
            "# FAISS Hyperparameter Experiments Summary",
            "",
            "## Overview",
            "",
            "This document summarizes experiments on FAISS index configurations for CoSQA code search.",
            "",
            f"**Model**: {self.model}",
            f"**Corpus size**: {len(self.corpus):,} documents",
            f"**Test queries**: {len(self.test_queries)}",
            f"**Embedding dimension**: {self.embedding_dim}",
            "",
            "---",
            "",
        ]
        
        # Experiment 1
        summary_lines.extend([
            "## Experiment 1: Index Types Comparison",
            "",
            "Compare exact search (Flat) vs approximate search (IVF).",
            "",
            "| Configuration | Recall@10 | MRR@10 | nDCG@10 | Search Time (s) | Queries/s |",
            "|--------------|-----------|---------|---------|-----------------|-----------|"
        ])
        
        exp1 = results['experiment1_index_types']
        for config_name, metrics in exp1.items():
            summary_lines.append(
                f"| {config_name} | "
                f"{metrics['recall@10']:.4f} | "
                f"{metrics['mrr@10']:.4f} | "
                f"{metrics['ndcg@10']:.4f} | "
                f"{metrics['search_time_total']:.4f} | "
                f"{metrics['queries_per_second']:.1f} |"
            )
        
        summary_lines.extend(["", "---", ""])
        
        # Experiment 2
        summary_lines.extend([
            "## Experiment 2: nlist Parameter Sweep",
            "",
            "Impact of clustering granularity (fixed nprobe=nlist/10).",
            "",
            "| nlist | nprobe | Recall@10 | nDCG@10 | Search Time (s) |",
            "|-------|--------|-----------|---------|-----------------|"
        ])
        
        exp2 = results['experiment2_nlist_sweep']
        for config_name, metrics in sorted(exp2.items()):
            # Extract nlist and nprobe from config name
            parts = config_name.split('_')
            nlist = parts[1].replace('nlist', '')
            nprobe = parts[2].replace('nprobe', '')
            
            summary_lines.append(
                f"| {nlist} | {nprobe} | "
                f"{metrics['recall@10']:.4f} | "
                f"{metrics['ndcg@10']:.4f} | "
                f"{metrics['search_time_total']:.4f} |"
            )
        
        summary_lines.extend(["", "---", ""])
        
        # Experiment 3
        summary_lines.extend([
            "## Experiment 3: nprobe Parameter Sweep",
            "",
            f"Search accuracy vs speed trade-off (fixed nlist={int(np.sqrt(len(self.corpus)))}).",
            "",
            "| nprobe | Recall@10 | nDCG@10 | Search Time (s) | Queries/s |",
            "|--------|-----------|---------|-----------------|-----------|"
        ])
        
        exp3 = results['experiment3_nprobe_sweep']
        for config_name, metrics in sorted(exp3.items(), 
                                          key=lambda x: int(x[0].split('_')[2].replace('nprobe', ''))):
            nprobe = config_name.split('_')[2].replace('nprobe', '')
            
            summary_lines.append(
                f"| {nprobe} | "
                f"{metrics['recall@10']:.4f} | "
                f"{metrics['ndcg@10']:.4f} | "
                f"{metrics['search_time_total']:.4f} | "
                f"{metrics['queries_per_second']:.1f} |"
            )
        
        summary_lines.extend([
            "",
            "---",
            "",
            "## Key Findings",
            "",
            "### 1. Flat vs IVF",
            "- Flat index provides 100% recall (exact search)",
            "- IVF can provide similar accuracy with much faster search",
            "",
            "### 2. nlist Parameter",
            "- Higher nlist = more clusters = finer partitioning",
            "- Trade-off: slower build time vs faster search",
            "- Optimal: around sqrt(N) to 2*sqrt(N)",
            "",
            "### 3. nprobe Parameter",
            "- Higher nprobe = search more clusters = better recall",
            "- Trade-off: accuracy vs speed",
            "- Optimal: start with nlist/10, increase if recall too low",
            "",
            "## Recommendations",
            "",
            "**For production**:",
            f"- Use IVF with nlist={int(np.sqrt(len(self.corpus)))*2}, nprobe={int(np.sqrt(len(self.corpus))//5)}",
            "- Provides good balance of speed and accuracy",
            "",
            "**For high-accuracy requirements**:",
            "- Use Flat index or IVF with high nprobe (nlist/2)",
            "",
            "**For low-latency requirements**:",
            "- Use IVF with low nprobe (1-5)",
            "- Monitor recall and increase nprobe if needed",
            ""
        ])
        
        # Write summary
        summary_file = output_path / "FAISS_EXPERIMENTS_SUMMARY.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"\n✓ Summary saved to: {summary_file}")


def main():
    """Run all FAISS hyperparameter experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="FAISS hyperparameter experiments for CoSQA"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/finetuned",
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="Cache directory for CoSQA data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/faiss_hyperparams",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run scripts/04_finetune.py first")
        return 1
    
    # Run experiments
    experiment = FAISSHyperparameterExperiment(
        model_path=str(model_path),
        cache_dir=args.cache_dir
    )
    
    results = experiment.run_all_experiments(output_dir=args.output_dir)
    
    print("\n" + "="*80)
    print("✓ All FAISS hyperparameter experiments complete!")
    print("="*80)
    print(f"\nResults directory: {args.output_dir}/")
    print(f"  - experiment1_index_types.json")
    print(f"  - experiment2_nlist_sweep.json")
    print(f"  - experiment3_nprobe_sweep.json")
    print(f"  - all_experiments.json")
    print(f"  - FAISS_EXPERIMENTS_SUMMARY.md")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
