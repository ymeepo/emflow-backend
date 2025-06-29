from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
import os
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import our new services
from database import init_database, close_database
from embedding_service import init_embedding_service
from schema import initialize_em_tools_schema

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format=os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting EM Tools Backend...")
    try:
        # Initialize database connection
        init_database()
        logger.info("Neo4j database connected")
        
        # Initialize EM Tools schema
        initialize_em_tools_schema()
        logger.info("EM Tools schema initialized")
        
        # Initialize embedding service (this may take a while on first run)
        init_embedding_service()
        logger.info("Embedding service initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down EM Tools Backend...")
    close_database()


app = FastAPI(
    title="EM Tools Backend API",
    description="Backend API for EM Tools enterprise management dashboard with Neo4j knowledge graph and semantic search",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory state storage
state_store: Dict[str, Any] = {}

# In-memory funnel data storage
funnel_data: Dict[str, int] = {}

class StateItem(BaseModel):
    key: str
    value: Any

class StateValue(BaseModel):
    value: Any

class FunnelData(BaseModel):
    applicants: int
    hm_review: int
    hm_interview: int
    tech_screen: int
    panel_1: int
    panel_2: int
    hired: int

@app.get("/")
async def root():
    return {
        "message": "EM Tools Backend API",
        "version": "1.0.0",
        "features": ["State Storage", "Neo4j Knowledge Graph", "Semantic Search"],
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        from database import get_neo4j_connection
        from embedding_service import get_embedding_service
        
        # Test Neo4j connection
        neo4j_status = "connected"
        try:
            db = get_neo4j_connection()
            db.execute_query("RETURN 1")
        except Exception as e:
            neo4j_status = f"error: {str(e)[:100]}"
        
        # Test embedding service
        embedding_status = "loaded"
        try:
            service = get_embedding_service()
            if service._model is None:
                embedding_status = "not_loaded"
        except Exception as e:
            embedding_status = f"error: {str(e)[:100]}"
        
        return {
            "status": "healthy",
            "services": {
                "neo4j": neo4j_status,
                "embedding": embedding_status
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/state")
async def get_all_state():
    return state_store

@app.get("/state/{key}")
async def get_state(key: str):
    if key not in state_store:
        raise HTTPException(status_code=404, detail="Key not found")
    return {"key": key, "value": state_store[key]}

@app.post("/state")
async def set_state(item: StateItem):
    state_store[item.key] = item.value
    return {"message": "State updated", "key": item.key, "value": item.value}

@app.put("/state/{key}")
async def update_state(key: str, value: StateValue):
    state_store[key] = value.value
    return {"message": "State updated", "key": key, "value": value.value}

@app.delete("/state/{key}")
async def delete_state(key: str):
    if key not in state_store:
        raise HTTPException(status_code=404, detail="Key not found")
    deleted_value = state_store.pop(key)
    return {"message": "State deleted", "key": key, "value": deleted_value}

@app.delete("/state")
async def clear_all_state():
    state_store.clear()
    return {"message": "All state cleared"}

# Funnel endpoints
@app.get("/funnel")
async def get_funnel_data():
    if not funnel_data:
        raise HTTPException(status_code=404, detail="No funnel data found")
    return funnel_data

@app.post("/funnel")
async def save_funnel_data(data: FunnelData):
    funnel_data.update({
        "applicants": data.applicants,
        "hm_review": data.hm_review,
        "hm_interview": data.hm_interview,
        "tech_screen": data.tech_screen,
        "panel_1": data.panel_1,
        "panel_2": data.panel_2,
        "hired": data.hired
    })
    return {"message": "Funnel data saved", "data": funnel_data}

@app.delete("/funnel")
async def clear_funnel_data():
    funnel_data.clear()
    return {"message": "Funnel data cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)