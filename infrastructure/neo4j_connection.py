""" Neo4j async database connection and management. """

import os
from typing import Optional, Dict, Any, List
from neo4j import AsyncGraphDatabase, AsyncDriver
import logging

logger = logging.getLogger(__name__)


class AsyncNeo4jConnection:
    """Async Neo4j database connection manager with connection pooling."""
    
    def __init__(self):
        self._driver: Optional[AsyncDriver] = None
        self._uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self._username = os.getenv('NEO4J_USERNAME', 'neo4j')
        self._password = os.getenv('NEO4J_PASSWORD', 'password')
        self._database = os.getenv('NEO4J_DATABASE', 'neo4j')
    
    async def connect(self) -> None:
        """Establish async connection to Neo4j database."""
        try:
            self._driver = AsyncGraphDatabase.driver(
                self._uri,
                auth=(self._username, self._password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )
            # Test connection
            async with self._driver.session(database=self._database) as session:
                await session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {self._uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    async def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            await self._driver.close()
            logger.info("Neo4j connection closed")
    
    def get_session(self):
        """Get a new async session from the driver."""
        if not self._driver:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._driver.session(database=self._database)
    
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a query and return results as a list of dictionaries."""
        async with self.get_session() as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records
    
    async def execute_write_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a write query in a write transaction."""
        async def _run_write_tx(tx):
            result = await tx.run(query, parameters or {})
            return await result.data()
        
        async with self.get_session() as session:
            return await session.execute_write(_run_write_tx)


# Global connection instance
neo4j_db = AsyncNeo4jConnection()


def get_neo4j_connection() -> AsyncNeo4jConnection:
    """Get the global Neo4j connection instance."""
    return neo4j_db


async def init_database() -> None:
    """Initialize the database connection."""
    await neo4j_db.connect()


async def close_database() -> None:
    """Close the database connection."""
    await neo4j_db.close()