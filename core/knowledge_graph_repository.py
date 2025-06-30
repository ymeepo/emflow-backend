""" Knowledge graph repository interface (abstraction). """

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .models import Engineer, Project, SemanticSearchResult, RelationshipSearchResult


class KnowledgeGraphRepository(ABC):
    """Abstract interface for knowledge graph operations."""
    
    @abstractmethod
    async def search_engineers_by_embedding(
        self, 
        query_embedding: List[float], 
        top_k: int = 5, 
        threshold: float = 0.7
    ) -> List[SemanticSearchResult]:
        """Search engineers using semantic similarity."""
        pass
    
    @abstractmethod
    async def search_projects_by_embedding(
        self, 
        query_embedding: List[float], 
        top_k: int = 5, 
        threshold: float = 0.7
    ) -> List[SemanticSearchResult]:
        """Search projects using semantic similarity."""
        pass
    
    @abstractmethod
    async def get_engineer_by_id(self, engineer_id: str) -> Optional[Dict[str, Any]]:
        """Get engineer by ID."""
        pass
    
    @abstractmethod
    async def get_project_by_id(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get project by ID."""
        pass
    
    @abstractmethod
    async def get_engineer_relationships(self, engineer_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for an engineer."""
        pass
    
    @abstractmethod
    async def get_project_relationships(self, project_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for a project."""
        pass
    
    @abstractmethod
    async def get_knowledge_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        pass