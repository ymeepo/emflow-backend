"""Agent application service for coordinating agentic workflows."""

import logging
from .simple_agent import SimpleAgent
from .knowledge_graph_service import KnowledgeGraphApplicationService

logger = logging.getLogger(__name__)


class AgentApplicationService:
    """Application service for agent workflow operations."""
    
    def __init__(self, knowledge_service: KnowledgeGraphApplicationService):
        # Only use the simple LangGraph agent with injected dependencies
        self._simple_agent = SimpleAgent(knowledge_service)
        logger.info("LangGraph simple agent initialized")
    
    async def process_query(self, query: str):
        """Process a user query through the SimpleAgent."""
        agent_response = await self._simple_agent.process_query(query)
        return {
            'type': 'agent_response',
            'response': agent_response,
            'query': query
        }