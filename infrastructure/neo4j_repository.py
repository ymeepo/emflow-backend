""" Neo4j implementation of the knowledge graph repository. """

from typing import List, Dict, Any, Optional
import numpy as np
import logging

from core.knowledge_graph_repository import KnowledgeGraphRepository
from core.models import SemanticSearchResult, EntityType
from .neo4j_connection import get_neo4j_connection
from .qwen_embedding_service import get_embedding_service as get_concrete_embedding_service

logger = logging.getLogger(__name__)


class Neo4jKnowledgeGraphRepository(KnowledgeGraphRepository):
    """Neo4j implementation of the knowledge graph repository."""
    
    def __init__(self):
        self.db = get_neo4j_connection()
    
    def search_engineers_by_embedding(
        self, 
        query_embedding: List[float], 
        top_k: int = 5, 
        threshold: float = 0.7
    ) -> List[SemanticSearchResult]:
        """Search engineers using semantic similarity."""
        cypher_query = """
        MATCH (e:Engineer)
        WHERE e.embedding IS NOT NULL
        WITH e, 
             gds.similarity.cosine(e.embedding, $query_embedding) AS similarity
        WHERE similarity >= $threshold
        RETURN e.id as id,
               e.name as name,
               similarity,
               e.position as position,
               e.level as level,
               e.team as team,
               e.skills as skills,
               e.expertise as expertise,
               e.tenure as tenure,
               e.status as status,
               e.email as email
        ORDER BY similarity DESC
        LIMIT $top_k
        """
        
        try:
            raw_results = self.db.execute_query(cypher_query, {
                "query_embedding": query_embedding,
                "threshold": threshold,
                "top_k": top_k
            })
            return self._convert_to_search_results(raw_results, EntityType.ENGINEER)
        except Exception:
            # Fallback: Get all entities and compute similarity in Python
            logger.warning("GDS not available, using Python similarity computation")
            return self._fallback_engineer_search(query_embedding, top_k, threshold)
    
    def search_projects_by_embedding(
        self, 
        query_embedding: List[float], 
        top_k: int = 5, 
        threshold: float = 0.7
    ) -> List[SemanticSearchResult]:
        """Search projects using semantic similarity."""
        cypher_query = """
        MATCH (p:Project)
        WHERE p.embedding IS NOT NULL
        WITH p,
             gds.similarity.cosine(p.embedding, $query_embedding) AS similarity
        WHERE similarity >= $threshold
        RETURN p.id as id,
               p.name as name,
               similarity,
               p.description as description,
               p.stage as stage,
               p.status as status,
               p.technologies as technologies,
               p.business_value as business_value,
               p.priority as priority
        ORDER BY similarity DESC
        LIMIT $top_k
        """
        
        try:
            raw_results = self.db.execute_query(cypher_query, {
                "query_embedding": query_embedding,
                "threshold": threshold,
                "top_k": top_k
            })
            return self._convert_to_search_results(raw_results, EntityType.PROJECT)
        except Exception:
            # Fallback: Get all entities and compute similarity in Python
            logger.warning("GDS not available, using Python similarity computation")
            return self._fallback_project_search(query_embedding, top_k, threshold)
    
    def get_engineer_by_id(self, engineer_id: str) -> Optional[Dict[str, Any]]:
        """Get engineer by ID."""
        query = """
        MATCH (e:Engineer {id: $engineer_id})
        RETURN e.id as id,
               e.name as name,
               e.position as position,
               e.level as level,
               e.team as team,
               e.skills as skills,
               e.expertise as expertise,
               e.tenure as tenure,
               e.status as status,
               e.email as email
        """
        results = self.db.execute_query(query, {"engineer_id": engineer_id})
        return results[0] if results else None
    
    def get_project_by_id(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get project by ID."""
        query = """
        MATCH (p:Project {id: $project_id})
        RETURN p.id as id,
               p.name as name,
               p.description as description,
               p.stage as stage,
               p.status as status,
               p.technologies as technologies,
               p.business_value as business_value,
               p.priority as priority
        """
        results = self.db.execute_query(query, {"project_id": project_id})
        return results[0] if results else None
    
    def get_engineer_relationships(self, engineer_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for an engineer."""
        relationship_queries = {
            "projects": """
            MATCH (e:Engineer {id: $entity_id})-[r]->(p:Project)
            RETURN 'project' as related_type,
                   type(r) as relationship_type,
                   p.id as related_id,
                   p.name as related_name,
                   p.stage as stage,
                   p.status as status,
                   r.role as role,
                   r.start_date as start_date
            """,
            "manager": """
            MATCH (m:Engineer)-[r:MANAGES]->(e:Engineer {id: $entity_id})
            RETURN 'manager' as related_type,
                   'reports_to' as relationship_type,
                   m.id as related_id,
                   m.name as related_name,
                   m.position as position,
                   m.level as level,
                   null as role,
                   null as start_date
            """,
            "collaborators": """
            MATCH (e:Engineer {id: $entity_id})-[r:COLLABORATES_WITH|MENTORED_BY|ADVISES]-(colleague:Engineer)
            RETURN 'collaborator' as related_type,
                   type(r) as relationship_type,
                   colleague.id as related_id,
                   colleague.name as related_name,
                   colleague.position as position,
                   colleague.level as level,
                   r.context as context,
                   null as start_date
            """
        }
        
        all_relationships = []
        for rel_type, query in relationship_queries.items():
            try:
                results = self.db.execute_query(query, {"entity_id": engineer_id})
                all_relationships.extend(results)
            except Exception as e:
                logger.warning(f"Relationship query '{rel_type}' failed: {e}")
        
        return all_relationships
    
    def get_project_relationships(self, project_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for a project."""
        relationship_queries = {
            "engineers": """
            MATCH (e:Engineer)-[r]->(p:Project {id: $entity_id})
            RETURN 'engineer' as related_type,
                   type(r) as relationship_type,
                   e.id as related_id,
                   e.name as related_name,
                   e.position as position,
                   e.level as level,
                   r.role as role,
                   r.start_date as start_date
            """
        }
        
        all_relationships = []
        for rel_type, query in relationship_queries.items():
            try:
                results = self.db.execute_query(query, {"entity_id": project_id})
                all_relationships.extend(results)
            except Exception as e:
                logger.warning(f"Relationship query '{rel_type}' failed: {e}")
        
        return all_relationships
    
    def get_knowledge_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        # Node counts
        node_stats = self.db.execute_query("""
        MATCH (n)
        RETURN labels(n)[0] as node_type, count(n) as count
        ORDER BY count DESC
        """)
        
        # Relationship counts
        rel_stats = self.db.execute_query("""
        MATCH ()-[r]->()
        RETURN type(r) as relationship_type, count(r) as count
        ORDER BY count DESC
        """)
        
        # Skills distribution
        skills_stats = self.db.execute_query("""
        MATCH (e:Engineer)
        UNWIND e.skills as skill
        RETURN skill, count(*) as engineer_count
        ORDER BY engineer_count DESC
        LIMIT 10
        """)
        
        # Active projects
        active_projects = self.db.execute_query("""
        MATCH (p:Project)
        WHERE p.status = 'ongoing'
        RETURN count(p) as active_count
        """)
        
        return {
            "nodes": {record["node_type"]: record["count"] for record in node_stats},
            "relationships": {record["relationship_type"]: record["count"] for record in rel_stats},
            "top_skills": [{"skill": record["skill"], "count": record["engineer_count"]} 
                          for record in skills_stats],
            "active_projects": active_projects[0]["active_count"] if active_projects else 0
        }
    
    def _fallback_engineer_search(self, query_embedding: List[float], top_k: int, threshold: float) -> List[SemanticSearchResult]:
        """Fallback method for engineer search when GDS is not available."""
        fallback_query = """
        MATCH (e:Engineer)
        WHERE e.embedding IS NOT NULL
        RETURN e.id as id, e.name as name, e.embedding as embedding,
               e.position as position, e.level as level, e.team as team,
               e.skills as skills, e.expertise as expertise, e.tenure as tenure,
               e.status as status, e.email as email
        """
        all_entities = self.db.execute_query(fallback_query)
        return self._compute_similarity_in_python(all_entities, query_embedding, top_k, threshold)
    
    def _fallback_project_search(self, query_embedding: List[float], top_k: int, threshold: float) -> List[SemanticSearchResult]:
        """Fallback method for project search when GDS is not available."""
        fallback_query = """
        MATCH (p:Project)
        WHERE p.embedding IS NOT NULL
        RETURN p.id as id, p.name as name, p.embedding as embedding,
               p.description as description, p.stage as stage, p.status as status,
               p.technologies as technologies, p.business_value as business_value, p.priority as priority
        """
        all_entities = self.db.execute_query(fallback_query)
        return self._compute_similarity_in_python(all_entities, query_embedding, top_k, threshold)
    
    def _compute_similarity_in_python(self, entities: List[Dict], query_embedding: List[float], top_k: int, threshold: float) -> List[SemanticSearchResult]:
        """Compute similarity in Python when GDS is not available."""
        embedding_service = get_concrete_embedding_service()
        query_emb = np.array(query_embedding, dtype=np.float32)
        
        results = []
        for entity in entities:
            entity_embedding = np.array(entity['embedding'], dtype=np.float32)
            similarity = embedding_service.compute_similarity(query_emb, entity_embedding)
            
            if similarity >= threshold:
                entity_data = entity.copy()
                entity_data['similarity'] = similarity
                del entity_data['embedding']  # Remove embedding from result
                results.append(entity_data)
        
        # Sort by similarity and limit
        results.sort(key=lambda x: x['similarity'], reverse=True)
        limited_results = results[:top_k]
        
        # Determine entity type from the first result if available
        if limited_results:
            entity_type = EntityType.ENGINEER if 'position' in limited_results[0] else EntityType.PROJECT
        else:
            # Default fallback - could be either, but we'll use engineer as default
            entity_type = EntityType.ENGINEER
            
        return self._convert_to_search_results(limited_results, entity_type)
    
    def _convert_to_search_results(self, raw_results: List[Dict[str, Any]], entity_type: EntityType) -> List[SemanticSearchResult]:
        """Convert raw database results to SemanticSearchResult domain models."""
        search_results = []
        for record in raw_results:
            # Extract common fields
            entity_id = record.get('id', '')
            name = record.get('name', '')
            similarity = record.get('similarity', 0.0)
            
            # Extract details (everything except id, name, similarity)
            details = {k: v for k, v in record.items() if k not in ['id', 'name', 'similarity']}
            
            search_result = SemanticSearchResult(
                id=entity_id,
                name=name,
                similarity_score=float(similarity),
                entity_type=entity_type,
                details=details
            )
            search_results.append(search_result)
        
        return search_results