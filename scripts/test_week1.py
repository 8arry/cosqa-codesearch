"""
Test script to verify Week 1 implementation.

This script tests:
1. Data loading from CoSQA
2. FAISS search engine functionality
3. Evaluation metrics calculation

Run: python scripts/test_week1.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from data.load_cosqa import CoSQADataLoader
from src.engine.faiss_engine import FAISSSearchEngine
from src.evaluation.metrics import (
    recall_at_k,
    mrr_at_k,
    ndcg_at_k,
    calculate_all_metrics
)


def test_data_loader():
    """Test CoSQA data loading."""
    print("="*80)
    print("TEST 1: Data Loading")
    print("="*80)
    
    try:
        loader = CoSQADataLoader(cache_dir="data/cache")
        
        # Test loading train data
        print("\n1.1 Loading train data...")
        train_df = loader.load_train()
        print(f"✓ Train data shape: {train_df.shape}")
        assert len(train_df) > 0, "Train data is empty"
        assert 'query_text' in train_df.columns, "Missing query_text column"
        assert 'code_text' in train_df.columns, "Missing code_text column"
        
        # Test loading test data
        print("\n1.2 Loading test data...")
        test_df = loader.load_test()
        print(f"✓ Test data shape: {test_df.shape}")
        assert len(test_df) > 0, "Test data is empty"
        
        # Test getting all corpus
        print("\n1.3 Loading all corpus...")
        corpus = loader.get_all_corpus()
        print(f"✓ Total corpus: {len(corpus)}")
        assert len(corpus) > 0, "Corpus is empty"
        
        # Test getting positive pairs
        print("\n1.4 Getting positive training pairs...")
        positive_pairs = loader.get_positive_pairs('train')
        print(f"✓ Positive pairs: {len(positive_pairs)}")
        assert len(positive_pairs) > 0, "No positive pairs found"
        
        # Print statistics
        print("\n1.5 Dataset statistics:")
        stats = loader.get_statistics()
        print(f"  Total queries: {stats['total_queries']}")
        print(f"  Total corpus: {stats['total_corpus']}")
        print(f"  Train positive: {stats['train']['positive']}")
        print(f"  Test positive: {stats['test']['positive']}")
        
        print("\n✓ Data loading tests PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Data loading tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_search_engine():
    """Test FAISS search engine."""
    print("\n" + "="*80)
    print("TEST 2: Search Engine")
    print("="*80)
    
    try:
        # Create demo documents
        demo_docs = [
            {'id': 'd1', 'text': 'def sort_list(lst): return sorted(lst)'},
            {'id': 'd2', 'text': 'def reverse_string(s): return s[::-1]'},
            {'id': 'd3', 'text': 'def add_numbers(a, b): return a + b'},
            {'id': 'd4', 'text': 'def multiply(x, y): return x * y'},
            {'id': 'd5', 'text': 'def find_max(numbers): return max(numbers)'},
        ]
        
        # Initialize engine with a small model for testing
        print("\n2.1 Initializing search engine...")
        engine = FAISSSearchEngine(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=32
        )
        print("✓ Engine initialized")
        
        # Test ingestion
        print("\n2.2 Testing document ingestion...")
        engine.ingest(demo_docs)
        assert engine.get_num_documents() == 5, "Wrong number of documents indexed"
        print("✓ Ingested 5 documents")
        
        # Test single search
        print("\n2.3 Testing single search...")
        query = "how to sort a list"
        results = engine.search(query, top_k=3)
        assert len(results) == 3, "Wrong number of results"
        assert results[0]['id'] == 'd1', "Wrong top result (should be d1)"
        print(f"✓ Query: '{query}'")
        print(f"  Top result: {results[0]['id']} (score: {results[0]['score']:.4f})")
        
        # Test batch search
        print("\n2.4 Testing batch search...")
        queries = ["sort a list", "reverse text", "add two numbers"]
        batch_results = engine.batch_search(queries, top_k=2)
        assert len(batch_results) == 3, "Wrong number of batch results"
        assert all(len(r) == 2 for r in batch_results), "Wrong results per query"
        print("✓ Batch search for 3 queries")
        
        # Test save and load
        print("\n2.5 Testing save/load...")
        engine.save_index("test_index")
        
        engine2 = FAISSSearchEngine(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        engine2.load_index("test_index")
        assert engine2.get_num_documents() == 5, "Wrong number after loading"
        
        # Verify search works after loading
        results2 = engine2.search(query, top_k=3)
        assert results2[0]['id'] == results[0]['id'], "Results differ after reload"
        print("✓ Index saved and loaded successfully")
        
        print("\n✓ Search engine tests PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Search engine tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_metrics():
    """Test evaluation metrics."""
    print("\n" + "="*80)
    print("TEST 3: Evaluation Metrics")
    print("="*80)
    
    try:
        # Test case 1: Perfect ranking
        print("\n3.1 Testing perfect ranking...")
        ranks = [1, 1, 1]
        rel_scores = [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
        
        recall = recall_at_k(ranks, k=10)
        mrr = mrr_at_k(ranks, k=10)
        ndcg = ndcg_at_k(rel_scores, k=10)
        
        assert abs(recall - 1.0) < 0.001, f"Recall should be 1.0, got {recall}"
        assert abs(mrr - 1.0) < 0.001, f"MRR should be 1.0, got {mrr}"
        assert abs(ndcg - 1.0) < 0.001, f"nDCG should be 1.0, got {ndcg}"
        print("✓ Perfect ranking: Recall=1.0, MRR=1.0, nDCG=1.0")
        
        # Test case 2: Mixed ranking
        print("\n3.2 Testing mixed ranking...")
        ranks = [1, 5, 15, 2]
        # ranks [1, 5, 15, 2]:
        # @k=5: hits are [1, 5, 2] = 3/4 = 0.75
        # @k=10: hits are [1, 5, 2] = 3/4 = 0.75
        # MRR@10: (1/1 + 1/5 + 0 + 1/2) / 4 = (1.0 + 0.2 + 0 + 0.5) / 4 = 0.425
        
        recall_5 = recall_at_k(ranks, k=5)
        recall_10 = recall_at_k(ranks, k=10)
        mrr_10 = mrr_at_k(ranks, k=10)
        
        assert abs(recall_5 - 0.75) < 0.001, f"Recall@5 should be 0.75, got {recall_5}"
        assert abs(recall_10 - 0.75) < 0.001, f"Recall@10 should be 0.75, got {recall_10}"
        assert abs(mrr_10 - 0.425) < 0.001, f"MRR@10 should be 0.425, got {mrr_10}"
        print(f"✓ Mixed ranking: Recall@5={recall_5}, Recall@10={recall_10}, MRR@10={mrr_10}")
        
        # Test case 3: All metrics
        print("\n3.3 Testing calculate_all_metrics...")
        rel_scores_padded = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0] * 10,
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        
        metrics = calculate_all_metrics(ranks, rel_scores_padded, k_values=[5, 10])
        
        assert 'recall@5' in metrics, "Missing recall@5"
        assert 'mrr@10' in metrics, "Missing mrr@10"
        assert 'ndcg@10' in metrics, "Missing ndcg@10"
        
        print("✓ All metrics calculated:")
        for metric_name, value in sorted(metrics.items()):
            print(f"    {metric_name}: {value:.4f}")
        
        print("\n✓ Evaluation metrics tests PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Evaluation metrics tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═"*78 + "╗")
    print("║" + " "*25 + "WEEK 1 VERIFICATION TEST" + " "*29 + "║")
    print("╚" + "═"*78 + "╝")
    print("\n")
    
    results = {
        "Data Loading": test_data_loader(),
        "Search Engine": test_search_engine(),
        "Evaluation Metrics": test_evaluation_metrics(),
    }
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
        if not passed:
            all_passed = False
    
    print("="*80)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Week 1 implementation is complete.")
        print("\nYou can now proceed to:")
        print("  - scripts/02_build_index.py (build full CoSQA index)")
        print("  - scripts/03_evaluate_baseline.py (evaluate baseline model)")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
