""" Embedding service using Qwen3-Embedding-0.6B model for semantic search. """

import os
import logging
from typing import List, Optional, Union
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using Qwen3-Embedding-0.6B model."""
    
    def __init__(self):
        self._model: Optional[SentenceTransformer] = None
        self._model_name = "Qwen/Qwen3-Embedding-0.6B"
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._cache_dir = os.getenv('EMBEDDING_CACHE_DIR', './models')
        self._max_sequence_length = int(os.getenv('EMBEDDING_MAX_LENGTH', '512'))
        self._batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE', '32'))
    
    def load_model(self) -> None:
        """Load the Qwen embedding model."""
        try:
            logger.info(f"Loading Qwen embedding model on {self._device}...")
            self._model = SentenceTransformer(
                self._model_name,
                cache_folder=self._cache_dir,
                device=self._device
            )
            
            # Set max sequence length
            if hasattr(self._model, 'max_seq_length'):
                self._model.max_seq_length = self._max_sequence_length
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.get_embedding_dimension()}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def get_model(self) -> SentenceTransformer:
        """Get the loaded model, loading it if necessary."""
        if self._model is None:
            self.load_model()
        return self._model
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        model = self.get_model()
        return model.get_sentence_embedding_dimension()
    
    @lru_cache(maxsize=1000)
    def encode_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        Uses LRU cache for frequently requested texts.
        """
        if not text or not text.strip():
            return np.zeros(self.get_embedding_dimension(), dtype=np.float32)
        
        model = self.get_model()
        embedding = model.encode(
            text.strip(),
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.astype(np.float32)
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return np.array([])
        
        # Filter out empty texts
        valid_texts = [text.strip() for text in texts if text and text.strip()]
        if not valid_texts:
            return np.zeros((len(texts), self.get_embedding_dimension()), dtype=np.float32)
        
        model = self.get_model()
        embeddings = model.encode(
            valid_texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(valid_texts) > 10
        )
        return embeddings.astype(np.float32)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        if embedding1.size == 0 or embedding2.size == 0:
            return 0.0
        
        # Ensure embeddings are normalized
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        normalized1 = embedding1 / norm1
        normalized2 = embedding2 / norm2
        
        similarity = np.dot(normalized1, normalized2)
        return float(similarity)
    
    def find_most_similar(
        self, 
        query_embedding: np.ndarray, 
        candidate_embeddings: List[np.ndarray],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find the most similar embeddings to a query embedding.
        Returns list of (index, similarity_score) tuples.
        """
        if not candidate_embeddings or query_embedding.size == 0:
            return []
        
        similarities = []
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate)
            similarities.append((i, similarity))
        
        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def generate_engineer_embedding(self, engineer_data: dict) -> np.ndarray:
        """Generate embedding for engineer data."""
        # Combine relevant text fields
        text_parts = []
        
        if engineer_data.get('name'):
            text_parts.append(f"Name: {engineer_data['name']}")
        
        if engineer_data.get('position'):
            text_parts.append(f"Position: {engineer_data['position']}")
        
        if engineer_data.get('skills'):
            if isinstance(engineer_data['skills'], list):
                skills_text = ', '.join(engineer_data['skills'])
            else:
                skills_text = str(engineer_data['skills'])
            text_parts.append(f"Skills: {skills_text}")
        
        if engineer_data.get('careerAspirations'):
            text_parts.append(f"Career Goals: {engineer_data['careerAspirations']}")
        
        if engineer_data.get('strengths'):
            if isinstance(engineer_data['strengths'], list):
                strengths_text = ', '.join(engineer_data['strengths'])
            else:
                strengths_text = str(engineer_data['strengths'])
            text_parts.append(f"Strengths: {strengths_text}")
        
        combined_text = ' | '.join(text_parts)
        return self.encode_text(combined_text)
    
    def generate_project_embedding(self, project_data: dict) -> np.ndarray:
        """Generate embedding for project data."""
        text_parts = []
        
        if project_data.get('name'):
            text_parts.append(f"Project: {project_data['name']}")
        
        if project_data.get('description'):
            text_parts.append(f"Description: {project_data['description']}")
        
        if project_data.get('businessProblem'):
            text_parts.append(f"Problem: {project_data['businessProblem']}")
        
        if project_data.get('targetOutput'):
            text_parts.append(f"Target: {project_data['targetOutput']}")
        
        if project_data.get('technologies'):
            if isinstance(project_data['technologies'], list):
                tech_text = ', '.join(project_data['technologies'])
            else:
                tech_text = str(project_data['technologies'])
            text_parts.append(f"Technologies: {tech_text}")
        
        combined_text = ' | '.join(text_parts)
        return self.encode_text(combined_text)


# Global embedding service instance
embedding_service = EmbeddingService()


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance."""
    return embedding_service


def init_embedding_service() -> None:
    """Initialize the embedding service by loading the model."""
    embedding_service.load_model()