"""
Neo4j schema initialization for EM Tools knowledge graph.
"""
import logging
from infrastructure.neo4j_connection import get_neo4j_connection

logger = logging.getLogger(__name__)


def initialize_em_tools_schema() -> None:
    """Initialize Neo4j schema with constraints and indexes for EM Tools."""
    db = get_neo4j_connection()
    
    schema_queries = [
        # Unique constraints
        "CREATE CONSTRAINT engineer_id IF NOT EXISTS FOR (e:Engineer) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT project_id IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
        
        # Indexes for performance
        "CREATE INDEX engineer_name IF NOT EXISTS FOR (e:Engineer) ON (e.name)",
        "CREATE INDEX project_name IF NOT EXISTS FOR (p:Project) ON (p.name)",
        "CREATE INDEX engineer_skills IF NOT EXISTS FOR (e:Engineer) ON (e.skills)",
        "CREATE INDEX engineer_level IF NOT EXISTS FOR (e:Engineer) ON (e.level)",
        "CREATE INDEX engineer_team IF NOT EXISTS FOR (e:Engineer) ON (e.team)",
        "CREATE INDEX project_stage IF NOT EXISTS FOR (p:Project) ON (p.stage)",
        "CREATE INDEX project_status IF NOT EXISTS FOR (p:Project) ON (p.status)",
        
        # Vector indexes for semantic search (requires Neo4j 5.0+)
        """
        CREATE VECTOR INDEX engineer_embeddings IF NOT EXISTS
        FOR (e:Engineer) ON (e.embedding)
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 512,
                `vector.similarity_function`: 'cosine'
            }
        }
        """,
        """
        CREATE VECTOR INDEX project_embeddings IF NOT EXISTS
        FOR (p:Project) ON (p.embedding)
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 512,
                `vector.similarity_function`: 'cosine'
            }
        }
        """
    ]
    
    for query in schema_queries:
        try:
            db.execute_write_query(query)
            logger.info(f"Schema query executed: {query[:50]}...")
        except Exception as e:
            logger.warning(f"Schema query failed (may already exist): {e}")
    
    logger.info("EM Tools schema initialization completed")


def create_sample_data() -> None:
    """Create sample engineers and projects for development."""
    db = get_neo4j_connection()
    
    # Sample engineers
    engineers = [
        {
            "id": "sarah-chen",
            "name": "Sarah Chen",
            "position": "Senior Frontend Engineer",
            "level": "Senior Engineer",
            "team": "Frontend Team",
            "skills": ["React", "TypeScript", "CSS", "Next.js"],
            "tenure": "2.5 years"
        },
        {
            "id": "alex-chen", 
            "name": "Alex Chen",
            "position": "Staff Backend Engineer",
            "level": "Staff Engineer",
            "team": "Backend Team",
            "skills": ["Python", "Node.js", "PostgreSQL", "AWS"],
            "tenure": "3.8 years"
        }
    ]
    
    # Sample projects
    projects = [
        {
            "id": "ai-customer-support",
            "name": "AI Customer Support Dashboard",
            "stage": "Prototype",
            "status": "ongoing",
            "description": "AI-powered customer support dashboard with automated responses"
        },
        {
            "id": "react-migration",
            "name": "React 18 Migration", 
            "stage": "Validation",
            "status": "completed",
            "description": "Migration of legacy frontend to React 18 with modern patterns"
        }
    ]
    
    # Create engineers
    for engineer in engineers:
        query = """
        MERGE (e:Engineer {id: $id})
        SET e.name = $name,
            e.position = $position,
            e.level = $level,
            e.team = $team,
            e.skills = $skills,
            e.tenure = $tenure,
            e.created_at = datetime(),
            e.updated_at = datetime()
        """
        db.execute_write_query(query, engineer)
    
    # Create projects
    for project in projects:
        query = """
        MERGE (p:Project {id: $id})
        SET p.name = $name,
            p.stage = $stage,
            p.status = $status,
            p.description = $description,
            p.created_at = datetime(),
            p.updated_at = datetime()
        """
        db.execute_write_query(query, project)
    
    # Create relationships
    relationships = [
        ("sarah-chen", "ai-customer-support", "LEADS"),
        ("sarah-chen", "react-migration", "WORKED_ON"),
        ("alex-chen", "ai-customer-support", "WORKED_ON")
    ]
    
    for engineer_id, project_id, relationship in relationships:
        query = """
        MATCH (e:Engineer {id: $engineer_id})
        MATCH (p:Project {id: $project_id})
        MERGE (e)-[r:""" + relationship + """]->(p)
        SET r.created_at = datetime()
        """
        db.execute_write_query(query, {"engineer_id": engineer_id, "project_id": project_id})
    
    logger.info("Sample data created successfully")


def clear_all_data() -> None:
    """Clear all data from the knowledge graph (for development/testing)."""
    db = get_neo4j_connection()
    
    # Delete all nodes and relationships
    query = "MATCH (n) DETACH DELETE n"
    db.execute_write_query(query)
    
    logger.info("All data cleared from knowledge graph")