from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import infrastructure services
from infrastructure.neo4j_connection import init_database, close_database
from infrastructure.qwen_embedding_service import init_embedding_service
from application.schema import initialize_em_tools_schema
from application.sample_data import create_comprehensive_sample_data

# Import routers
from routers import agent

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
        await init_database()
        logger.info("Neo4j database connected")
        
        # Initialize EM Tools schema
        await initialize_em_tools_schema()
        logger.info("EM Tools schema initialized")
        
        # Initialize embedding service (this may take a while on first run)
        init_embedding_service()
        logger.info("Embedding service initialized")
        
        # Create sample data if it doesn't exist
        await create_comprehensive_sample_data()
        logger.info("Sample data initialization completed")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down EM Tools Backend...")
    await close_database()


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

# Include routers
app.include_router(agent.router)

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        from infrastructure.neo4j_connection import get_neo4j_connection
        from infrastructure.qwen_embedding_service import get_embedding_service
        
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)