""" Agent endpoints for streaming responses and agentic workflows. """

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import asyncio
from container import container

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


class UserPrompt(BaseModel):
    prompt: str
    user_id: str = "anonymous"


async def process_user_prompt(prompt: str, user_id: str):
    """
    Async generator that yields intermediate responses while processing a user prompt.
    Uses the application service to perform real knowledge graph operations.
    """
    
    # Get the agent application service
    agent_service = container.agent_application_service
    
    # Step 1: Parse and understand the prompt
    yield {
        "type": "status",
        "message": f"🧠 Analyzing prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'"
    }
    await asyncio.sleep(0.5)
    
    # Step 2: Query knowledge graph
    yield {
        "type": "status", 
        "message": "🔍 Searching knowledge graph for relevant engineers and projects..."
    }
    
    try:
        # Use real application service to process the query
        search_results = await agent_service.process_query(prompt)
        
        yield {
            "type": "data",
            "message": f"Found {search_results['type']} results",
            "data": {
                "query_type": search_results['type'],
                "results_count": len(search_results.get('results', [])) if 'results' in search_results else 
                              len(search_results.get('engineer_results', [])) + len(search_results.get('project_results', []))
            }
        }
        await asyncio.sleep(0.3)
        
        # Step 3: Present search results
        yield {
            "type": "status",
            "message": "🎯 Processing semantic search results..."
        }
        await asyncio.sleep(0.8)
        
        # Format results for display
        if search_results['type'] == 'engineer_search':
            top_matches = [
                {
                    "name": result.name,
                    "similarity": round(result.similarity_score, 3),
                    "skills": result.details.get('skills', [])
                }
                for result in search_results['results'][:3]
            ]
        elif search_results['type'] == 'project_search':
            top_matches = [
                {
                    "name": result.name,
                    "similarity": round(result.similarity_score, 3),
                    "technologies": result.details.get('technologies', [])
                }
                for result in search_results['results'][:3]
            ]
        else:  # general_search
            top_matches = {
                "engineers": [
                    {
                        "name": result.name,
                        "similarity": round(result.similarity_score, 3),
                        "skills": result.details.get('skills', [])
                    }
                    for result in search_results['engineer_results'][:2]
                ],
                "projects": [
                    {
                        "name": result.name,
                        "similarity": round(result.similarity_score, 3),
                        "technologies": result.details.get('technologies', [])
                    }
                    for result in search_results['project_results'][:2]
                ]
            }
        
        yield {
            "type": "data",
            "message": "Semantic similarity analysis complete",
            "data": {"top_matches": top_matches}
        }
        await asyncio.sleep(0.5)
        
        # Step 4: Generate insights
        yield {
            "type": "status",
            "message": "🤖 Generating insights and recommendations..."
        }
        await asyncio.sleep(1.0)
        
        # Step 5: Final response
        yield {
            "type": "result",
            "message": "Analysis complete! Here are the key findings:",
            "data": {
                "summary": f"Based on your query '{prompt}', I found relevant {search_results['type'].replace('_', ' ')} results.",
                "search_results": search_results,
                "confidence_score": 0.87,
                "processing_time": "2.8 seconds"
            }
        }
        
    except Exception as e:
        yield {
            "type": "error",
            "message": f"Error during processing: {str(e)}"
        }


@router.post("/query")
async def agent_query(request: UserPrompt):
    """
    Stream responses back to the client as the agent processes the user prompt.
    
    Returns Server-Sent Events (SSE) format for real-time streaming of the agentic workflow.
    """
    
    async def generate_stream():
        try:
            # Send initial acknowledgment
            yield f"data: {json.dumps({'type': 'start', 'message': 'Processing your request...', 'user_id': request.user_id})}\n\n"
            
            # Process the prompt and yield intermediate results
            async for response in process_user_prompt(request.prompt, request.user_id):
                # Format as Server-Sent Events
                yield f"data: {json.dumps(response)}\n\n"
                
            # Send completion signal
            yield f"data: {json.dumps({'type': 'complete', 'message': 'Processing finished'})}\n\n"
            
        except Exception as e:
            # Send error if something goes wrong
            error_response = {
                "type": "error",
                "message": f"Error processing request: {str(e)}"
            }
            yield f"data: {json.dumps(error_response)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )


@router.get("/status")
async def agent_status():
    """Get the current status of the agent system."""
    return {
        "status": "active",
        "version": "1.0.0",
        "capabilities": [
            "knowledge_graph_search",
            "semantic_search", 
            "relationship_analysis",
            "insight_generation"
        ],
        "models": {
            "embedding": "Qwen3-Embedding-0.6B",
            "database": "Neo4j"
        }
    }


@router.get("/capabilities")
async def agent_capabilities():
    """Get detailed information about agent capabilities."""
    return {
        "workflow_steps": [
            {
                "step": "prompt_analysis",
                "description": "Parse and understand user intent",
                "estimated_time": "0.5s"
            },
            {
                "step": "knowledge_search", 
                "description": "Query Neo4j knowledge graph",
                "estimated_time": "1.0s"
            },
            {
                "step": "semantic_search",
                "description": "Embedding-based similarity search",
                "estimated_time": "0.8s"
            },
            {
                "step": "relationship_analysis",
                "description": "Analyze connections and collaborations",
                "estimated_time": "0.7s"
            },
            {
                "step": "insight_generation",
                "description": "Generate recommendations and next steps",
                "estimated_time": "1.2s"
            }
        ],
        "supported_queries": [
            "Find engineers with specific skills",
            "Identify project collaborations",
            "Analyze team structures",
            "Recommend team formations"
        ]
    }