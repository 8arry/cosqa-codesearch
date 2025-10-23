"""
CoSQA Dataset Loader

This module provides utilities to load and preprocess the CoSQA dataset
from HuggingFace. The CoSQA dataset consists of three configurations:
1. queries: Natural language queries
2. corpus: Python code snippets
3. default (train/test/valid): Query-Corpus pairs with relevance scores

Author: Your Name
Date: October 2025
"""

import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


@dataclass
class CoSQASample:
    """A single CoSQA training/test sample with full information."""
    query_id: str
    query_text: str
    corpus_id: str
    code_text: str
    score: int  # 0 or 1
    partition: str  # 'train', 'test', or 'valid'


class CoSQADataLoader:
    """
    Unified data loader for CoSQA dataset.
    
    Usage:
        loader = CoSQADataLoader(cache_dir="./data/cache")
        train_data = loader.load_train()
        test_data = loader.load_test()
        all_corpus = loader.get_all_corpus()
    """
    
    def __init__(self, cache_dir: Optional[str] = None, force_reload: bool = False):
        """
        Initialize CoSQA data loader.
        
        Args:
            cache_dir: Directory to cache processed data
            force_reload: If True, reload from HuggingFace even if cache exists
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.force_reload = force_reload
        
        # Will be loaded lazily
        self._queries_dict = None
        self._corpus_dict = None
        self._train_pairs = None
        self._test_pairs = None
        self._valid_pairs = None
    
    def _load_queries(self) -> Dict[str, str]:
        """Load all queries and return as {query_id: query_text} dict."""
        if self._queries_dict is not None and not self.force_reload:
            return self._queries_dict
        
        cache_file = self.cache_dir / "queries.json"
        
        if cache_file.exists() and not self.force_reload:
            print(f"Loading queries from cache: {cache_file}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                self._queries_dict = json.load(f)
            return self._queries_dict
        
        print("Loading queries from HuggingFace...")
        queries_dataset = load_dataset("CoIR-Retrieval/cosqa", "queries", split="queries")
        
        self._queries_dict = {
            item['_id']: item['text'] 
            for item in tqdm(queries_dataset, desc="Processing queries")
        }
        
        # Cache to disk
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(self._queries_dict, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Loaded {len(self._queries_dict)} queries")
        return self._queries_dict
    
    def _load_corpus(self) -> Dict[str, str]:
        """Load all corpus and return as {corpus_id: code_text} dict."""
        if self._corpus_dict is not None and not self.force_reload:
            return self._corpus_dict
        
        cache_file = self.cache_dir / "corpus.json"
        
        if cache_file.exists() and not self.force_reload:
            print(f"Loading corpus from cache: {cache_file}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                self._corpus_dict = json.load(f)
            return self._corpus_dict
        
        print("Loading corpus from HuggingFace...")
        corpus_dataset = load_dataset("CoIR-Retrieval/cosqa", "corpus", split="corpus")
        
        self._corpus_dict = {
            item['_id']: item['text'] 
            for item in tqdm(corpus_dataset, desc="Processing corpus")
        }
        
        # Cache to disk
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(self._corpus_dict, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Loaded {len(self._corpus_dict)} code snippets")
        return self._corpus_dict
    
    def _load_split(self, split: str) -> pd.DataFrame:
        """
        Load a specific split (train/test/valid) and merge with queries and corpus.
        
        Args:
            split: One of 'train', 'test', 'valid'
            
        Returns:
            DataFrame with columns: query_id, corpus_id, score, query_text, code_text
        """
        cache_file = self.cache_dir / f"{split}_full.csv"
        
        if cache_file.exists() and not self.force_reload:
            print(f"Loading {split} data from cache: {cache_file}")
            return pd.read_csv(cache_file)
        
        print(f"Loading {split} split from HuggingFace...")
        pairs_dataset = load_dataset("CoIR-Retrieval/cosqa", "default", split=split)
        
        # Load queries and corpus if not already loaded
        queries_dict = self._load_queries()
        corpus_dict = self._load_corpus()
        
        # Merge pairs with actual text
        data = []
        for item in tqdm(pairs_dataset, desc=f"Processing {split} pairs"):
            query_id = item['query-id']
            corpus_id = item['corpus-id']
            score = item['score']
            
            data.append({
                'query_id': query_id,
                'corpus_id': corpus_id,
                'score': score,
                'query_text': queries_dict.get(query_id, ''),
                'code_text': corpus_dict.get(corpus_id, '')
            })
        
        df = pd.DataFrame(data)
        
        # Cache to disk
        df.to_csv(cache_file, index=False)
        print(f"✓ Loaded {len(df)} {split} samples")
        
        return df
    
    def load_train(self) -> pd.DataFrame:
        """
        Load training data with full information.
        
        Returns:
            DataFrame with columns: query_id, corpus_id, score, query_text, code_text
        """
        return self._load_split('train')
    
    def load_test(self) -> pd.DataFrame:
        """Load test data with full information."""
        return self._load_split('test')
    
    def load_valid(self) -> pd.DataFrame:
        """Load validation data with full information."""
        return self._load_split('valid')
    
    def get_all_corpus(self) -> List[Dict[str, str]]:
        """
        Get all corpus documents for indexing.
        
        Returns:
            List of dicts with keys: 'id', 'text'
        """
        corpus_dict = self._load_corpus()
        return [
            {'id': corpus_id, 'text': code_text}
            for corpus_id, code_text in corpus_dict.items()
        ]
    
    def get_all_queries(self) -> List[Dict[str, str]]:
        """
        Get all queries.
        
        Returns:
            List of dicts with keys: 'id', 'text'
        """
        queries_dict = self._load_queries()
        return [
            {'id': query_id, 'text': query_text}
            for query_id, query_text in queries_dict.items()
        ]
    
    def get_positive_pairs(self, split: str = 'train') -> List[Tuple[str, str]]:
        """
        Get positive (query, code) pairs for training.
        
        Args:
            split: One of 'train', 'test', 'valid'
            
        Returns:
            List of (query_text, code_text) tuples where score=1
        """
        df = self._load_split(split)
        positive_df = df[df['score'] == 1]
        
        return list(zip(positive_df['query_text'], positive_df['code_text']))
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        train_df = self.load_train()
        test_df = self.load_test()
        valid_df = self.load_valid()
        
        stats = {
            'total_queries': len(self._load_queries()),
            'total_corpus': len(self._load_corpus()),
            'train': {
                'total': len(train_df),
                'positive': (train_df['score'] == 1).sum(),
                'negative': (train_df['score'] == 0).sum(),
            },
            'test': {
                'total': len(test_df),
                'positive': (test_df['score'] == 1).sum(),
                'negative': (test_df['score'] == 0).sum(),
            },
            'valid': {
                'total': len(valid_df),
                'positive': (valid_df['score'] == 1).sum(),
                'negative': (valid_df['score'] == 0).sum(),
            }
        }
        
        return stats


# Convenience functions for quick loading
def load_cosqa_queries() -> Dict[str, str]:
    """Quick load all queries as {query_id: query_text} dict."""
    loader = CoSQADataLoader()
    return loader._load_queries()


def load_cosqa_corpus() -> Dict[str, str]:
    """Quick load all corpus as {corpus_id: code_text} dict."""
    loader = CoSQADataLoader()
    return loader._load_corpus()


def load_cosqa_train() -> pd.DataFrame:
    """Quick load training data."""
    loader = CoSQADataLoader()
    return loader.load_train()


def load_cosqa_test() -> pd.DataFrame:
    """Quick load test data."""
    loader = CoSQADataLoader()
    return loader.load_test()


def load_cosqa_valid() -> pd.DataFrame:
    """Quick load validation data."""
    loader = CoSQADataLoader()
    return loader.load_valid()


if __name__ == "__main__":
    # Demo usage
    print("="*80)
    print("CoSQA Data Loader Demo")
    print("="*80)
    
    loader = CoSQADataLoader(cache_dir="data/cache")
    
    # Get statistics
    print("\nDataset Statistics:")
    stats = loader.get_statistics()
    print(json.dumps(stats, indent=2))
    
    # Load train data
    print("\nLoading training data...")
    train_df = loader.load_train()
    print(f"Train shape: {train_df.shape}")
    print("\nFirst positive sample:")
    positive_sample = train_df[train_df['score'] == 1].iloc[0]
    print(f"Query: {positive_sample['query_text']}")
    print(f"Code: {positive_sample['code_text'][:200]}...")
    
    # Get positive pairs for training
    print("\nGetting positive pairs for training...")
    positive_pairs = loader.get_positive_pairs('train')
    print(f"Total positive training pairs: {len(positive_pairs)}")
    
    print("\n✓ Data loader demo completed successfully!")
