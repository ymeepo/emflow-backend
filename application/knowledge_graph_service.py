"""Knowledge graph application service for coordinating domain and infrastructure."""

import logging
from core.knowledge_graph_repository import KnowledgeGraphRepository
from core.embedding_service import EmbeddingService
from core.models import Engineer, Project, SemanticSearchResult

logger = logging.getLogger(__name__)


class KnowledgeGraphApplicationService:
    """Application service for knowledge graph operations."""
    
    def __init__(self, knowledge_repo: KnowledgeGraphRepository, embedding_service: EmbeddingService):
        self._knowledge_repo = knowledge_repo
        self._embedding_service = embedding_service
    
    async def search_engineers_by_skills(self, query: str, limit: int = 5) -> list[SemanticSearchResult]:
        """Search for engineers by skills using semantic search."""
        logger.info(f"KNOWLEDGE SERVICE: Encoding query '{query}'")
        query_embedding = self._embedding_service.encode_text(query)
        logger.info(f"KNOWLEDGE SERVICE: Generated embedding with {len(query_embedding)} dimensions")
        logger.info(f"KNOWLEDGE SERVICE: First 5 embedding values: {query_embedding[:5]}")
        return await self._knowledge_repo.search_engineers_by_embedding(query_embedding, limit)
    
    async def search_projects_by_description(self, query: str, limit: int = 5) -> list[SemanticSearchResult]:
        """Search for projects by description using semantic search."""
        query_embedding = self._embedding_service.encode_text(query)
        return await self._knowledge_repo.search_projects_by_embedding(query_embedding, limit)
    
    async def get_engineer_relationships(self, engineer_id: str) -> list[dict]:
        """Get all relationships for a specific engineer."""
        return await self._knowledge_repo.get_engineer_relationships(engineer_id)
    
    async def get_project_relationships(self, project_id: str) -> list[dict]:
        """Get all relationships for a specific project."""
        return await self._knowledge_repo.get_project_relationships(project_id)