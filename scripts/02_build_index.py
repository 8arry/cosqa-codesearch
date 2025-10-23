"""
Script 2: Build Search Index

This script builds a FAISS index over all CoSQA corpus documents
using a pre-trained embedding model.

Run: python scripts/02_build_index.py --model intfloat/e5-base-v2
"""

import sys
import argparse
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.load_cosqa import CoSQADataLoader
from src.engine.faiss_engine import FAISSSearchEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS search index for CoSQA corpus")
    parser.add_argument(
        "--model",
        type=str,
        default="intfloat/e5-base-v2",
        help="Model name from HuggingFace (default: intfloat/e5-base-v2)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for encoding (default: 64)"
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default="cosqa_index",
        help="Name for the saved index (default: cosqa_index)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="Cache directory for data (default: data/cache)"
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="indexes",
        help="Directory to save index (default: indexes)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*80)
    print("STEP 2: Build Search Index")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  Model:      {args.model}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Index name: {args.index_name}")
    
    # Load corpus
    print("\n" + "-"*80)
    print("Loading CoSQA corpus...")
    print("-"*80)
    
    loader = CoSQADataLoader(cache_dir=args.cache_dir)
    corpus = loader.get_all_corpus()
    
    print(f"\n✓ Loaded {len(corpus):,} code snippets")
    
    # Prepare documents for indexing (corpus is already in the right format)
    documents = corpus  # Already list of {'id': ..., 'text': ...}
    
    # Initialize search engine
    print("\n" + "-"*80)
    print("Initializing search engine...")
    print("-"*80)
    
    start_time = time.time()
    engine = FAISSSearchEngine(
        model_name=args.model,
        batch_size=args.batch_size
    )
    init_time = time.time() - start_time
    print(f"\n✓ Search engine initialized in {init_time:.2f}s")
    
    # Build index
    print("\n" + "-"*80)
    print("Building FAISS index...")
    print("-"*80)
    print(f"\nThis will encode {len(documents):,} documents...")
    print(f"Estimated time: ~{len(documents) / args.batch_size / 10:.1f}-{len(documents) / args.batch_size / 5:.1f} minutes")
    
    start_time = time.time()
    engine.ingest(documents)
    build_time = time.time() - start_time
    
    print(f"\n✓ Index built in {build_time:.2f}s ({build_time/60:.2f} minutes)")
    print(f"  Documents indexed: {engine.get_num_documents():,}")
    print(f"  Throughput: {len(documents)/build_time:.1f} docs/sec")
    
    # Save index
    print("\n" + "-"*80)
    print("Saving index...")
    print("-"*80)
    
    index_dir = project_root / args.index_dir
    index_dir.mkdir(exist_ok=True)
    
    index_path = index_dir / args.index_name
    engine.save_index(str(index_path))
    
    print(f"\n✓ Index saved to: {index_path}")
    
    # Test search
    print("\n" + "-"*80)
    print("Testing search...")
    print("-"*80)
    
    test_query = "how to read a file in python"
    print(f"\nQuery: '{test_query}'")
    
    results = engine.search(test_query, top_k=3)
    
    print(f"\nTop 3 results:")
    for i, result in enumerate(results, 1):
        code_snippet = result['text'][:100] + "..." if len(result['text']) > 100 else result['text']
        print(f"\n{i}. ID: {result['id']} | Score: {result['score']:.4f}")
        print(f"   {code_snippet}")
    
    # Summary
    print("\n" + "="*80)
    print("✓ Index building complete!")
    print("="*80)
    
    print(f"\nIndex statistics:")
    print(f"  Total documents:  {engine.get_num_documents():,}")
    print(f"  Model:            {args.model}")
    print(f"  Index file:       {index_path}")
    print(f"  Build time:       {build_time:.2f}s ({build_time/60:.2f} min)")
    
    print(f"\nNext step: Run scripts/03_evaluate_baseline.py to evaluate on test set")


if __name__ == "__main__":
    main()
