"""
Script 1: Prepare CoSQA Data

This script downloads and caches the CoSQA dataset.
All data will be stored in data/cache/ for future use.

Run: python scripts/01_prepare_data.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.load_cosqa import CoSQADataLoader


def main():
    print("="*80)
    print("STEP 1: Prepare CoSQA Data")
    print("="*80)
    
    # Initialize data loader
    cache_dir = project_root / "data" / "cache"
    print(f"\nCache directory: {cache_dir}")
    
    loader = CoSQADataLoader(cache_dir=str(cache_dir))
    
    # Load all data (this will download and cache if not already present)
    print("\n1. Loading queries...")
    queries = loader.get_all_queries()
    print(f"✓ Loaded {len(queries)} queries")
    
    print("\n2. Loading corpus...")
    corpus = loader.get_all_corpus()
    print(f"✓ Loaded {len(corpus)} code snippets")
    
    print("\n3. Loading train split...")
    train_df = loader.load_train()
    print(f"✓ Loaded {len(train_df)} training pairs")
    
    print("\n4. Loading test split...")
    test_df = loader.load_test()
    print(f"✓ Loaded {len(test_df)} test pairs")
    
    print("\n5. Loading valid split...")
    valid_df = loader.load_valid()
    print(f"✓ Loaded {len(valid_df)} validation pairs")
    
    # Print statistics
    print("\n" + "="*80)
    print("Dataset Statistics")
    print("="*80)
    
    stats = loader.get_statistics()
    
    print(f"\nTotal queries:    {stats['total_queries']:,}")
    print(f"Total corpus:     {stats['total_corpus']:,}")
    
    print(f"\nTrain split:")
    print(f"  Total pairs:    {stats['train']['total']:,}")
    print(f"  Positive pairs: {stats['train']['positive']:,} ({stats['train']['positive']/stats['train']['total']*100:.1f}%)")
    print(f"  Negative pairs: {stats['train']['negative']:,} ({stats['train']['negative']/stats['train']['total']*100:.1f}%)")
    
    print(f"\nTest split:")
    print(f"  Total pairs:    {stats['test']['total']:,}")
    print(f"  Positive pairs: {stats['test']['positive']:,}")
    
    print(f"\nValid split:")
    print(f"  Total pairs:    {stats['valid']['total']:,}")
    print(f"  Positive pairs: {stats['valid']['positive']:,}")
    
    # Extract positive pairs for fine-tuning
    print("\n" + "="*80)
    print("Extracting Positive Pairs for Fine-tuning")
    print("="*80)
    
    positive_pairs = loader.get_positive_pairs('train')
    print(f"\n✓ Extracted {len(positive_pairs)} positive (query, code) pairs")
    print(f"  These will be used for fine-tuning in Week 3")
    
    print("\n" + "="*80)
    print("✓ Data preparation complete!")
    print("="*80)
    print(f"\nAll data cached to: {cache_dir}")
    print("\nNext step: Run scripts/02_build_index.py to build the search index")
    

if __name__ == "__main__":
    main()
