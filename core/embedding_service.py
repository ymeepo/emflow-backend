""" Embedding service interface (abstraction). """

from abc import ABC, abstractmethod
from typing import List


class EmbeddingService(ABC):
    """Abstract interface for embedding operations."""
    
    @abstractmethod
    def encode_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        pass
    
    @abstractmethod
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute similarity between two embeddings."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        pass