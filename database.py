""" Neo4j database connection and management. """

import os
from typing import Optional, Dict, Any
from neo4j import GraphDatabase, Driver
import logging

logger = logging.getLogger(__name__)


class Neo4jConnection:
    """Neo4j database connection manager with connection pooling."""
    
    def __init__(self):
        self._driver: Optional[Driver] = None
        self._uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self._username = os.getenv('NEO4J_USERNAME', 'neo4j')
        self._password = os.getenv('NEO4J_PASSWORD', 'password')
        self._database = os.getenv('NEO4J_DATABASE', 'neo4j')
    
    def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._username, self._password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )
            # Test connection
            with self._driver.session(database=self._database) as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {self._uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            self._driver.close()
            logger.info("Neo4j connection closed")
    
    def get_session(self):
        """Get a new session from the driver."""
        if not self._driver:
            self.connect()
        return self._driver.session(database=self._database)
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> list:
        """Execute a Cypher query and return results."""
        with self.get_session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def execute_write_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> list:
        """Execute a write Cypher query and return results."""
        with self.get_session() as session:
            result = session.execute_write(lambda tx: tx.run(query, parameters or {}))
            return [record.data() for record in result] if result else []
    
# Global connection instance
neo4j_db = Neo4jConnection()


def get_neo4j_connection() -> Neo4jConnection:
    """Get the global Neo4j connection instance."""
    return neo4j_db


def init_database() -> None:
    """Initialize the database connection."""
    neo4j_db.connect()


def close_database() -> None:
    """Close the database connection."""
    neo4j_db.close()