"""
FAISS-based search engine implementation.
"""

import os
import pickle
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .base_engine import BaseSearchEngine


class FAISSSearchEngine(BaseSearchEngine):
    """
    FAISS-based dense retrieval search engine.
    
    This implementation uses:
    - Sentence-Transformers for encoding
    - FAISS IndexFlatIP for exact cosine similarity search
    - L2 normalization for vectors
    
    Usage:
        engine = FAISSSearchEngine(model_name="intfloat/e5-base-v2")
        engine.ingest(documents)
        results = engine.search("how to sort a list in python", top_k=10)
    """
    
    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        normalize: bool = True,
        use_gpu: bool = False,
        batch_size: int = 64
    ):
        """
        Initialize FAISS search engine.
        
        Args:
            model_name: HuggingFace model name or local path
            normalize: Whether to L2 normalize embeddings (for cosine similarity)
            use_gpu: Whether to use GPU for FAISS (requires faiss-gpu)
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.normalize = normalize
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        
        # Load model
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index (IndexFlatIP for inner product)
        self.index = faiss.IndexFlatIP(self.dim)
        if use_gpu and faiss.get_num_gpus() > 0:
            print(f"Using GPU for FAISS (found {faiss.get_num_gpus()} GPUs)")
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        
        # Store document metadata
        self.documents = []  # List of {'id': ..., 'text': ...}
        self.id_to_idx = {}  # Map document ID to index in FAISS
    
    def _encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of shape (len(texts), dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False  # We'll normalize manually
        )
        
        # L2 normalize for cosine similarity
        if self.normalize:
            faiss.normalize_L2(embeddings)
        
        return embeddings.astype('float32')
    
    def ingest(self, documents: List[Dict[str, str]]) -> None:
        """
        Index a batch of documents.
        
        Args:
            documents: List of dicts with keys 'id' and 'text'
        """
        if not documents:
            print("Warning: No documents to ingest")
            return
        
        print(f"Ingesting {len(documents)} documents...")
        
        # Extract texts and IDs
        texts = [doc['text'] for doc in documents]
        doc_ids = [doc['id'] for doc in documents]
        
        # Encode documents
        embeddings = self._encode(texts, show_progress=True)
        
        # Add to FAISS index
        start_idx = len(self.documents)
        self.index.add(embeddings)
        
        # Store metadata
        for i, doc in enumerate(documents):
            self.documents.append(doc)
            self.id_to_idx[doc['id']] = start_idx + i
        
        print(f"✓ Indexed {len(documents)} documents. Total: {len(self.documents)}")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        return_scores: bool = True
    ) -> List[Dict]:
        """
        Search for relevant documents.
        
        Args:
            query: Natural language query
            top_k: Number of top results
            return_scores: Whether to include similarity scores
            
        Returns:
            List of result dicts with 'id', 'text', 'score', 'rank'
        """
        if len(self.documents) == 0:
            print("Warning: No documents indexed")
            return []
        
        # Encode query
        query_embedding = self._encode([query], show_progress=False)
        
        # Search in FAISS
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        
        # Format results
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue
            
            doc = self.documents[idx]
            result = {
                'id': doc['id'],
                'text': doc['text'],
                'rank': rank
            }
            
            if return_scores:
                result['score'] = float(score)
            
            results.append(result)
        
        return results
    
    def batch_search(
        self,
        queries: List[str],
        top_k: int = 10,
        return_scores: bool = True
    ) -> List[List[Dict]]:
        """
        Batch search for multiple queries (optimized).
        
        Args:
            queries: List of query strings
            top_k: Number of top results per query
            return_scores: Whether to include scores
            
        Returns:
            List of result lists
        """
        if len(self.documents) == 0:
            print("Warning: No documents indexed")
            return [[] for _ in queries]
        
        # Encode all queries at once
        query_embeddings = self._encode(queries, show_progress=True)
        
        # Batch search
        scores_batch, indices_batch = self.index.search(
            query_embeddings,
            min(top_k, len(self.documents))
        )
        
        # Format results for each query
        all_results = []
        for scores, indices in zip(scores_batch, indices_batch):
            results = []
            for rank, (idx, score) in enumerate(zip(indices, scores), 1):
                if idx == -1:
                    continue
                
                doc = self.documents[idx]
                result = {
                    'id': doc['id'],
                    'text': doc['text'],
                    'rank': rank
                }
                
                if return_scores:
                    result['score'] = float(score)
                
                results.append(result)
            
            all_results.append(results)
        
        return all_results
    
    def save_index(self, path: str) -> None:
        """
        Save index and metadata to disk.
        
        Args:
            path: Directory path to save index
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = path / "faiss_index.bin"
        if self.use_gpu:
            # Move to CPU before saving
            index_cpu = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(index_cpu, str(index_file))
        else:
            faiss.write_index(self.index, str(index_file))
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'dim': self.dim,
            'normalize': self.normalize,
            'documents': self.documents,
            'id_to_idx': self.id_to_idx
        }
        
        metadata_file = path / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✓ Index saved to {path}")
    
    def load_index(self, path: str) -> None:
        """
        Load index and metadata from disk.
        
        Args:
            path: Directory path to load index from
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Index path does not exist: {path}")
        
        # Load FAISS index
        index_file = path / "faiss_index.bin"
        index_cpu = faiss.read_index(str(index_file))
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_all_gpus(index_cpu)
        else:
            self.index = index_cpu
        
        # Load metadata
        metadata_file = path / "metadata.pkl"
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        self.documents = metadata['documents']
        self.id_to_idx = metadata['id_to_idx']
        self.dim = metadata['dim']
        
        print(f"✓ Index loaded from {path}")
        print(f"  Documents: {len(self.documents)}")
        print(f"  Dimension: {self.dim}")
    
    def clear(self) -> None:
        """Clear all indexed documents."""
        self.index.reset()
        self.documents = []
        self.id_to_idx = {}
        print("✓ Index cleared")
    
    def get_num_documents(self) -> int:
        """Get number of indexed documents."""
        return len(self.documents)


if __name__ == "__main__":
    # Demo usage
    print("="*80)
    print("FAISS Search Engine Demo")
    print("="*80)
    
    # Sample documents
    demo_docs = [
        {'id': 'd1', 'text': 'def sort_list(lst): return sorted(lst)'},
        {'id': 'd2', 'text': 'def reverse_string(s): return s[::-1]'},
        {'id': 'd3', 'text': 'def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)'},
        {'id': 'd4', 'text': 'def is_prime(n): return all(n % i != 0 for i in range(2, int(n**0.5) + 1))'},
        {'id': 'd5', 'text': 'def merge_sort(arr): pass  # sorting algorithm implementation'},
    ]
    
    # Initialize engine
    engine = FAISSSearchEngine(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Index documents
    engine.ingest(demo_docs)
    
    # Search
    query = "how to sort a list in python"
    print(f"\nQuery: {query}")
    results = engine.search(query, top_k=3)
    
    print("\nTop 3 Results:")
    for r in results:
        print(f"  Rank {r['rank']}: {r['id']} (score: {r['score']:.4f})")
        print(f"    {r['text']}")
    
    # Save and load
    print("\nSaving index...")
    engine.save_index("demo_index")
    
    print("Loading index...")
    engine2 = FAISSSearchEngine(model_name="sentence-transformers/all-MiniLM-L6-v2")
    engine2.load_index("demo_index")
    
    print(f"Loaded {engine2.get_num_documents()} documents")
    
    print("\n✓ Demo completed successfully!")
