"""Simple LangGraph agent with just two tools using create_react_agent."""

import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from .knowledge_graph_service import KnowledgeGraphApplicationService

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class SimpleAgent:
    """Simple LangGraph agent with two tools using create_react_agent."""
    
    def __init__(self, knowledge_service: KnowledgeGraphApplicationService):
        """Initialize the simple agent with injected dependencies."""
        self.knowledge_service = knowledge_service
        
        # Initialize Claude LLM
        anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        
        if anthropic_api_key:
            self.llm = ChatAnthropic(
                model="claude-sonnet-4-20250514",
                temperature=0.1,
                api_key=anthropic_api_key
            )
            logger.info("Initialized Claude Sonnet 4 for simple agent")
        else:
            logger.warning("ANTHROPIC_API_KEY not found. Agent will not work properly.")
            self.llm = None
        
        # Define tools as bound methods
        self.tools = [self.list_all_engineers, self.search_engineers, self.search_projects]
        
        # Create the react agent using LangGraph's built-in helper
        if self.llm:
            self.agent = create_react_agent(self.llm, self.tools)
        else:
            self.agent = None
    
    async def search_engineers(self, query: str) -> str:
        """Search for engineers by skills or expertise using natural language queries.
        
        Args:
            query: Search query for engineers
            
        Returns:
            String describing found engineers
        """
        try:
            logger.info(f"ENGINEER SEARCH INPUT: '{query}'")
        
            # Use async search directly
            results = await self.knowledge_service.search_engineers_by_skills(query, limit=3)
        
            logger.info(f"ENGINEER SEARCH RAW RESULTS: {len(results)} results found")
            for i, result in enumerate(results):
                logger.info(f"Result {i+1}: {result.name} - {result.similarity_score:.3f} - {result.details.get('skills', [])}")
            
            if not results:
                return f"No engineers found matching '{query}'"
            
            formatted_results = []
            for result in results:
                engineer_info = {
                    "name": result.name,
                    "position": result.details.get("position", ""),
                    "skills": result.details.get("skills", []),
                    "similarity": round(result.similarity_score, 3)
                }
                formatted_results.append(engineer_info)
            
            output = f"Found {len(results)} engineers: {formatted_results}"
            logger.info(f"ENGINEER SEARCH OUTPUT: {output}")
            return output
            
        except Exception as e:
            logger.error(f"Engineer search failed: {e}")
            return f"Error searching engineers: {str(e)}"

    async def search_projects(self, query: str) -> str:
        """Search for projects by description or technologies using natural language queries.
        
        Args:
            query: Search query for projects
            
        Returns:
            String describing found projects
        """
        try:
            # Use async search directly
            results = await self.knowledge_service.search_projects_by_description(query, limit=3)
            
            if not results:
                return f"No projects found matching '{query}'"
            
            formatted_results = []
            for result in results:
                project_info = {
                    "name": result.name,
                    "description": result.details.get("description", ""),
                    "technologies": result.details.get("technologies", []),
                    "similarity": round(result.similarity_score, 3)
                }
                formatted_results.append(project_info)
            
            return f"Found {len(results)} projects: {formatted_results}"
            
        except Exception as e:
            logger.error(f"Project search failed: {e}")
            return f"Error searching projects: {str(e)}"

    async def list_all_engineers(self) -> str:
        """Debug function to list all engineers in the database."""
        try:
            from infrastructure.neo4j_connection import get_neo4j_connection
            
            logger.info("DEBUG: Listing all engineers in database")
            
            # Use async database directly since we're in async context
            db = get_neo4j_connection()
            results = await db.execute_query("""
                MATCH (e:Engineer)
                RETURN e.id as id, e.name as name, e.skills as skills, 
                       e.position as position, e.expertise as expertise,
                       size(e.embedding) as embedding_size
                ORDER BY e.name
            """)
            
            logger.info(f"DEBUG: Found {len(results)} engineers in database")
            
            if not results:
                return "No engineers found in database"
            
            engineer_list = []
            for result in results:
                engineer_info = {
                    "name": result['name'],
                    "id": result['id'],
                    "position": result.get('position', ''),
                    "skills": result.get('skills', []),
                    "expertise": result.get('expertise', ''),
                    "embedding_size": result.get('embedding_size', 0)
                }
                engineer_list.append(engineer_info)
                logger.info(f"Engineer: {result['name']} - Skills: {result.get('skills', [])} - Embedding size: {result.get('embedding_size', 0)}")
            
            return f"Found {len(engineer_list)} engineers in database: {engineer_list}"
            
        except Exception as e:
            logger.error(f"Debug query failed: {e}")
            return f"Error listing engineers: {str(e)}"
    
    async def process_query(self, query: str) -> str:
        """Process a user query and return response."""
        if not self.agent:
            return "Agent not initialized - missing ANTHROPIC_API_KEY"
        
        try:
            # Use the react agent to process the query
            response = await self.agent.ainvoke({"messages": [{"role": "user", "content": query}]})
            
            # Extract the final response
            messages = response.get("messages", [])
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, 'content'):
                    return last_message.content
                elif isinstance(last_message, dict) and 'content' in last_message:
                    return last_message['content']
            
            return "No response generated"
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return f"Error processing query: {str(e)}"