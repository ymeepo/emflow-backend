"""Agent application service for coordinating agentic workflows."""

from .knowledge_graph_service import KnowledgeGraphApplicationService


class AgentApplicationService:
    """Application service for agent workflow operations."""
    
    def __init__(self, knowledge_service: KnowledgeGraphApplicationService):
        self._knowledge_service = knowledge_service
    
    async def process_query(self, query: str):
        """Process a user query through the agent workflow."""
        # This is where we would implement the agentic workflow
        # For now, we'll delegate to knowledge graph search
        
        # Determine query type and route appropriately
        if any(keyword in query.lower() for keyword in ['engineer', 'person', 'team member', 'who']):
            results = await self._knowledge_service.search_engineers_by_skills(query)
            return {
                'type': 'engineer_search',
                'results': results,
                'query': query
            }
        elif any(keyword in query.lower() for keyword in ['project', 'initiative', 'work']):
            results = await self._knowledge_service.search_projects_by_description(query)
            return {
                'type': 'project_search', 
                'results': results,
                'query': query
            }
        else:
            # Default to general search across both
            engineer_results = await self._knowledge_service.search_engineers_by_skills(query, limit=3)
            project_results = await self._knowledge_service.search_projects_by_description(query, limit=3)
            return {
                'type': 'general_search',
                'engineer_results': engineer_results,
                'project_results': project_results,
                'query': query
            }