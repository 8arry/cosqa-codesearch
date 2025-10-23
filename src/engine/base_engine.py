"""
Abstract base class for search engines.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class BaseSearchEngine(ABC):
    """
    Abstract base class for code search engines.
    
    All search engine implementations should inherit from this class
    and implement the required methods.
    """
    
    @abstractmethod
    def ingest(self, documents: List[Dict[str, str]]) -> None:
        """
        Index a batch of documents.
        
        Args:
            documents: List of dicts with keys 'id' and 'text'
                Example: [{'id': 'd1', 'text': 'def foo(): ...'}]
        """
        pass
    
    @abstractmethod
    def search(
        self, 
        query: str, 
        top_k: int = 10,
        return_scores: bool = True
    ) -> List[Dict]:
        """
        Search for relevant documents given a query.
        
        Args:
            query: Natural language query string
            top_k: Number of top results to return
            return_scores: Whether to include similarity scores
            
        Returns:
            List of dicts with keys:
                - 'id': document ID
                - 'text': document text
                - 'score': similarity score (if return_scores=True)
                - 'rank': 1-based rank in results
        """
        pass
    
    @abstractmethod
    def save_index(self, path: str) -> None:
        """
        Save the index to disk.
        
        Args:
            path: Path to save the index
        """
        pass
    
    @abstractmethod
    def load_index(self, path: str) -> None:
        """
        Load the index from disk.
        
        Args:
            path: Path to load the index from
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all indexed documents."""
        pass
    
    def get_num_documents(self) -> int:
        """
        Get the number of indexed documents.
        
        Returns:
            Number of documents in the index
        """
        raise NotImplementedError
    
    def batch_search(
        self,
        queries: List[str],
        top_k: int = 10,
        return_scores: bool = True
    ) -> List[List[Dict]]:
        """
        Batch search for multiple queries.
        
        Args:
            queries: List of query strings
            top_k: Number of top results per query
            return_scores: Whether to include scores
            
        Returns:
            List of result lists (one per query)
        """
        return [self.search(q, top_k, return_scores) for q in queries]
