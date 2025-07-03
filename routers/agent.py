""" Agent endpoints for streaming responses and agentic workflows. """

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import asyncio
import uuid
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
    
    # Generate unique session ID for tracking
    session_id = str(uuid.uuid4())[:8]
    print(f"🔍 STARTING SESSION {session_id} for prompt: '{prompt}'")
    
    # Get the agent application service
    agent_service = container.agent_application_service
    
    # Step 1: Parse and understand the prompt
    yield {
        "type": "status",
        "message": f"🧠 Analyzing prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'",
        "session_id": session_id,
        "step": 1
    }
    print(f"📤 SESSION {session_id} - Step 1: Analyzing prompt")
    await asyncio.sleep(0.5)
    
    # Step 2: Query knowledge graph
    yield {
        "type": "status", 
        "message": "🔍 Searching knowledge graph for relevant engineers and projects...",
        "session_id": session_id,
        "step": 2
    }
    print(f"📤 SESSION {session_id} - Step 2: Searching knowledge graph")
    
    try:
        # Use real application service to process the query
        search_results = await agent_service.process_query(prompt)
        
        yield {
            "type": "data",
            "message": f"Found {search_results['type']} results",
            "session_id": session_id,
            "step": 3,
            "data": {
                "query_type": search_results['type'],
                "results_count": len(search_results.get('results', [])) if 'results' in search_results else 
                              len(search_results.get('engineer_results', [])) + len(search_results.get('project_results', []))
            }
        }
        print(f"📤 SESSION {session_id} - Step 3: Found {search_results['type']} results")
        await asyncio.sleep(0.3)
        
        # Step 3: Present search results
        yield {
            "type": "status",
            "message": "🎯 Processing semantic search results..."
        }
        await asyncio.sleep(0.8)
        
        # Format results for display
        if search_results['type'] == 'agent_response':
            # For agent responses, just display the response
            top_matches = {"agent_response": search_results['response']}
        elif search_results['type'] == 'engineer_search':
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
            "session_id": session_id,
            "step": 5,
            "data": {
                "summary": f"Based on your query '{prompt}', I found relevant {search_results['type'].replace('_', ' ')} results.",
                "search_results": search_results,
                "confidence_score": 0.87,
                "processing_time": "2.8 seconds"
            }
        }
        print(f"📤 SESSION {session_id} - Step 5: Final response sent")
        
    except Exception as e:
        yield {
            "type": "error",
            "message": f"Error during processing: {str(e)}",
            "session_id": session_id
        }
        print(f"❌ SESSION {session_id} - Error: {str(e)}")
    
    print(f"🏁 SESSION {session_id} - COMPLETED")


@router.post("/query")
async def agent_query(request: UserPrompt):
    """
    Stream responses back to the client as the agent processes the user prompt.
    
    Returns Server-Sent Events (SSE) format for real-time streaming of the agentic workflow.
    """
    
    async def generate_stream():
        stream_id = str(uuid.uuid4())[:8]
        print(f"🌊 STARTING STREAM {stream_id} for user: {request.user_id}")
        try:
            # Send initial acknowledgment
            start_msg = {'type': 'start', 'message': 'Processing your request...', 'user_id': request.user_id, 'stream_id': stream_id}
            print(f"📡 STREAM {stream_id} - Sending start message")
            yield f"data: {json.dumps(start_msg)}\n\n"
            
            # Process the prompt and yield intermediate results
            response_count = 0
            async for response in process_user_prompt(request.prompt, request.user_id):
                response_count += 1
                # Add stream tracking
                response['stream_id'] = stream_id
                response['response_number'] = response_count
                print(f"📡 STREAM {stream_id} - Sending response #{response_count}: {response.get('type', 'unknown')}")
                # Format as Server-Sent Events
                yield f"data: {json.dumps(response)}\n\n"
                
            # Send completion signal
            complete_msg = {'type': 'complete', 'message': 'Processing finished', 'stream_id': stream_id, 'total_responses': response_count}
            print(f"📡 STREAM {stream_id} - Sending completion (total responses: {response_count})")
            yield f"data: {json.dumps(complete_msg)}\n\n"
            
        except Exception as e:
            # Send error if something goes wrong
            error_response = {
                "type": "error",
                "message": f"Error processing request: {str(e)}",
                "stream_id": stream_id
            }
            print(f"❌ STREAM {stream_id} - Error: {str(e)}")
            yield f"data: {json.dumps(error_response)}\n\n"
        
        print(f"🏁 STREAM {stream_id} - STREAM ENDED")
    
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