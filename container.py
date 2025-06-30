""" Dependency injection container for wiring up the application. """

from core.knowledge_graph_repository import KnowledgeGraphRepository
from core.embedding_service import EmbeddingService
from infrastructure.neo4j_repository import Neo4jKnowledgeGraphRepository
from infrastructure.qwen_embedding_service import QwenEmbeddingService
from application.knowledge_graph_service import KnowledgeGraphApplicationService
from application.agent_service import AgentApplicationService


class Container:
    """Simple dependency injection container."""
    
    def __init__(self):
        self._instances = {}
    
    @property
    def knowledge_graph_repo(self) -> KnowledgeGraphRepository:
        """Get knowledge graph repository instance."""
        if 'knowledge_graph_repo' not in self._instances:
            self._instances['knowledge_graph_repo'] = Neo4jKnowledgeGraphRepository()
        return self._instances['knowledge_graph_repo']
    
    @property
    def embedding_service(self) -> EmbeddingService:
        """Get embedding service instance."""
        if 'embedding_service' not in self._instances:
            self._instances['embedding_service'] = QwenEmbeddingService()
        return self._instances['embedding_service']
    
    @property
    def knowledge_graph_application_service(self) -> KnowledgeGraphApplicationService:
        """Get knowledge graph application service instance."""
        if 'knowledge_graph_application_service' not in self._instances:
            self._instances['knowledge_graph_application_service'] = KnowledgeGraphApplicationService(
                knowledge_repo=self.knowledge_graph_repo,
                embedding_service=self.embedding_service
            )
        return self._instances['knowledge_graph_application_service']
    
    @property
    def agent_application_service(self) -> AgentApplicationService:
        """Get agent application service instance."""
        if 'agent_application_service' not in self._instances:
            self._instances['agent_application_service'] = AgentApplicationService(
                knowledge_service=self.knowledge_graph_application_service
            )
        return self._instances['agent_application_service']


# Global container instance
container = Container()