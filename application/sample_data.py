""" Sample data for demonstrating semantic search and relationship traversal capabilities. """

import logging
from infrastructure.neo4j_connection import get_neo4j_connection
from infrastructure.qwen_embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


def check_sample_data_exists() -> bool:
    """Check if sample data already exists in the database."""
    try:
        db = get_neo4j_connection()
        
        # Check for both engineers AND projects to ensure complete sample data
        engineer_result = db.execute_query("""
        MATCH (e:Engineer {id: 'sarah-chen'})
        RETURN count(e) as count
        """)
        
        project_result = db.execute_query("""
        MATCH (p:Project {id: 'chatbot-platform'})
        RETURN count(p) as count
        """)
        
        has_engineers = engineer_result[0]['count'] > 0 if engineer_result else False
        has_projects = project_result[0]['count'] > 0 if project_result else False
        
        # Only return True if we have BOTH engineers and projects
        sample_data_complete = has_engineers and has_projects
        
        if has_engineers and not has_projects:
            logger.warning("Found engineers but no projects - sample data incomplete")
        elif not has_engineers and has_projects:
            logger.warning("Found projects but no engineers - sample data incomplete")
        
        return sample_data_complete
        
    except Exception as e:
        logger.warning(f"Could not check for existing sample data: {e}")
        return False


def create_comprehensive_sample_data() -> None:
    """Create comprehensive sample data with engineers, projects, and relationships."""
    db = get_neo4j_connection()
    embedding_service = get_embedding_service()
    
    # Check if sample data already exists
    if check_sample_data_exists():
        logger.info("Sample data already exists, skipping creation")
        return
    
    logger.info("Creating comprehensive sample data...")
    
    # Sample engineers with diverse skills for semantic search testing
    engineers = [
        {
            "id": "sarah-chen",
            "name": "Sarah Chen",
            "position": "Staff AI Engineer",
            "level": "Staff Engineer",
            "team": "AI/ML Team",
            "skills": ["Gen AI", "LangChain", "Python", "TensorFlow", "RAG", "Vector Databases"],
            "expertise": "Generative AI systems, RAG implementations, LLM fine-tuning",
            "tenure": "3.2 years",
            "status": "active",
            "email": "sarah.chen@company.com"
        },
        {
            "id": "alex-rodriguez",
            "name": "Alex Rodriguez", 
            "position": "Senior Machine Learning Engineer",
            "level": "Senior Engineer",
            "team": "AI/ML Team",
            "skills": ["Machine Learning", "Deep Learning", "PyTorch", "MLOps", "Computer Vision"],
            "expertise": "Computer vision models, ML pipeline automation, model deployment",
            "tenure": "2.8 years",
            "status": "active",
            "email": "alex.rodriguez@company.com"
        },
        {
            "id": "maria-garcia",
            "name": "Maria Garcia",
            "position": "Principal AI Researcher",
            "level": "Principal Engineer",
            "team": "Research Team",
            "skills": ["NLP", "Transformers", "BERT", "GPT", "Research", "Publications"],
            "expertise": "Natural language processing, transformer architectures, AI research",
            "tenure": "5.1 years", 
            "status": "active",
            "email": "maria.garcia@company.com"
        },
        {
            "id": "james-kim",
            "name": "James Kim",
            "position": "Senior Backend Engineer", 
            "level": "Senior Engineer",
            "team": "Platform Team",
            "skills": ["Node.js", "PostgreSQL", "Microservices", "Docker", "Kubernetes", "API Design"],
            "expertise": "Scalable backend systems, database optimization, cloud architecture",
            "tenure": "4.3 years",
            "status": "active",
            "email": "james.kim@company.com"
        },
        {
            "id": "emily-davis",
            "name": "Emily Davis",
            "position": "Frontend Tech Lead",
            "level": "Staff Engineer", 
            "team": "Frontend Team",
            "skills": ["React", "TypeScript", "Next.js", "GraphQL", "Design Systems"],
            "expertise": "Frontend architecture, component libraries, user experience",
            "tenure": "3.7 years",
            "status": "active",
            "email": "emily.davis@company.com"
        },
        {
            "id": "david-wilson",
            "name": "David Wilson",
            "position": "AI Product Manager",
            "level": "Senior Manager",
            "team": "Product Team",
            "skills": ["Product Strategy", "AI Ethics", "Stakeholder Management", "Roadmapping"],
            "expertise": "AI product development, ethical AI frameworks, cross-functional leadership",
            "tenure": "2.1 years",
            "status": "active",
            "email": "david.wilson@company.com"
        },
        {
            "id": "lisa-thompson",
            "name": "Lisa Thompson",
            "position": "Engineering Manager",
            "level": "Manager",
            "team": "AI/ML Team",
            "skills": ["Team Leadership", "Performance Management", "Technical Strategy", "Mentoring"],
            "expertise": "Engineering team management, talent development, technical direction",
            "tenure": "6.2 years",
            "status": "active",
            "email": "lisa.thompson@company.com"
        }
    ]
    
    # Sample projects demonstrating various AI/ML initiatives
    projects = [
        {
            "id": "chatbot-platform",
            "name": "Enterprise Chatbot Platform",
            "description": "AI-powered chatbot platform using LangChain and GPT models for customer support automation",
            "stage": "Production",
            "status": "ongoing",
            "technologies": ["LangChain", "OpenAI GPT", "Vector DB", "Python", "FastAPI"],
            "business_value": "40% reduction in support ticket volume, 24/7 customer assistance",
            "priority": "high"
        },
        {
            "id": "document-rag",
            "name": "Document RAG System", 
            "description": "Retrieval-Augmented Generation system for internal knowledge management and document search",
            "stage": "Beta",
            "status": "ongoing",
            "technologies": ["RAG", "Embeddings", "ChromaDB", "Streamlit", "Azure OpenAI"],
            "business_value": "60% faster information retrieval, improved knowledge sharing",
            "priority": "high"
        },
        {
            "id": "ml-recommendation",
            "name": "ML Recommendation Engine",
            "description": "Machine learning recommendation system for product suggestions and personalization",
            "stage": "Production",
            "status": "completed",
            "technologies": ["PyTorch", "Collaborative Filtering", "Feature Engineering", "A/B Testing"],
            "business_value": "25% increase in user engagement, 15% revenue uplift",
            "priority": "medium"
        },
        {
            "id": "computer-vision-qc",
            "name": "Computer Vision Quality Control",
            "description": "Computer vision system for automated quality control in manufacturing processes",
            "stage": "Prototype",
            "status": "ongoing", 
            "technologies": ["OpenCV", "YOLO", "PyTorch", "Edge Computing", "MLOps"],
            "business_value": "95% defect detection accuracy, reduced manual inspection time",
            "priority": "medium"
        },
        {
            "id": "nlp-sentiment",
            "name": "NLP Sentiment Analysis Platform",
            "description": "Natural language processing platform for social media sentiment analysis and brand monitoring",
            "stage": "Production",
            "status": "completed",
            "technologies": ["BERT", "Transformers", "Social Media APIs", "Real-time Processing"],
            "business_value": "Real-time brand sentiment tracking, proactive crisis management",
            "priority": "low"
        },
        {
            "id": "ai-ethics-framework",
            "name": "AI Ethics & Governance Framework",
            "description": "Comprehensive framework for responsible AI development and deployment across the organization",
            "stage": "Implementation",
            "status": "ongoing",
            "technologies": ["Policy Framework", "Bias Detection", "Explainable AI", "Governance Tools"],
            "business_value": "Risk mitigation, regulatory compliance, ethical AI practices",
            "priority": "high"
        }
    ]
    
    # Create engineers with embeddings
    logger.info(f"Creating {len(engineers)} engineers...")
    for i, engineer in enumerate(engineers):
        try:
            # Generate embedding for engineer
            engineer_text = f"Name: {engineer['name']} | Position: {engineer['position']} | Skills: {', '.join(engineer['skills'])} | Expertise: {engineer['expertise']}"
            embedding = embedding_service.encode_text(engineer_text)
            
            query = """
            MERGE (e:Engineer {id: $id})
            ON CREATE SET e.created_at = datetime()
            SET e.name = $name,
                e.position = $position,
                e.level = $level,
                e.team = $team,
                e.skills = $skills,
                e.expertise = $expertise,
                e.tenure = $tenure,
                e.status = $status,
                e.email = $email,
                e.embedding = $embedding,
                e.updated_at = datetime()
            """
            params = engineer.copy()
            params['embedding'] = embedding.tolist()
            db.execute_write_query(query, params)
            logger.info(f"Created engineer {i+1}/{len(engineers)}: {engineer['name']}")
        except Exception as e:
            logger.error(f"Failed to create engineer {engineer['name']}: {e}")
            raise
    
    # Create projects with embeddings
    logger.info(f"Creating {len(projects)} projects...")
    for i, project in enumerate(projects):
        try:
            # Generate embedding for project
            project_text = f"Project: {project['name']} | Description: {project['description']} | Technologies: {', '.join(project['technologies'])} | Business Value: {project['business_value']}"
            embedding = embedding_service.encode_text(project_text)
            
            query = """
            MERGE (p:Project {id: $id})
            ON CREATE SET p.created_at = datetime()
            SET p.name = $name,
                p.description = $description,
                p.stage = $stage,
                p.status = $status,
                p.technologies = $technologies,
                p.business_value = $business_value,
                p.priority = $priority,
                p.embedding = $embedding,
                p.updated_at = datetime()
            """
            params = project.copy()
            params['embedding'] = embedding.tolist()
            db.execute_write_query(query, params)
            logger.info(f"Created project {i+1}/{len(projects)}: {project['name']}")
        except Exception as e:
            logger.error(f"Failed to create project {project['name']}: {e}")
            raise
    
    # Create management relationships
    management_relationships = [
        ("lisa-thompson", "sarah-chen", "MANAGES"),
        ("lisa-thompson", "alex-rodriguez", "MANAGES"),
        ("lisa-thompson", "maria-garcia", "MANAGES"),
        ("david-wilson", "lisa-thompson", "COLLABORATES_WITH")
    ]
    
    for manager_id, engineer_id, relationship in management_relationships:
        query = f"""
        MATCH (m:Engineer {{id: $manager_id}})
        MATCH (e:Engineer {{id: $engineer_id}})
        CREATE (m)-[r:{relationship}]->(e)
        SET r.created_at = datetime()
        """
        db.execute_write_query(query, {"manager_id": manager_id, "engineer_id": engineer_id})
    
    # Create project work relationships with roles
    project_assignments = [
        ("sarah-chen", "chatbot-platform", "LEADS", "Technical Lead"),
        ("sarah-chen", "document-rag", "WORKS_ON", "AI Engineer"),
        ("alex-rodriguez", "ml-recommendation", "LEADS", "ML Lead"),
        ("alex-rodriguez", "computer-vision-qc", "WORKS_ON", "ML Engineer"),
        ("maria-garcia", "nlp-sentiment", "LEADS", "Research Lead"),
        ("maria-garcia", "ai-ethics-framework", "CONTRIBUTES_TO", "Ethics Advisor"),
        ("james-kim", "chatbot-platform", "WORKS_ON", "Backend Engineer"),
        ("james-kim", "document-rag", "WORKS_ON", "Platform Engineer"),
        ("emily-davis", "chatbot-platform", "WORKS_ON", "Frontend Lead"),
        ("david-wilson", "ai-ethics-framework", "LEADS", "Product Owner"),
        ("david-wilson", "chatbot-platform", "MANAGES", "Product Manager")
    ]
    
    for engineer_id, project_id, relationship, role in project_assignments:
        query = f"""
        MATCH (e:Engineer {{id: $engineer_id}})
        MATCH (p:Project {{id: $project_id}})
        CREATE (e)-[r:{relationship}]->(p)
        SET r.role = $role,
            r.start_date = date(),
            r.created_at = datetime()
        """
        db.execute_write_query(query, {
            "engineer_id": engineer_id, 
            "project_id": project_id, 
            "role": role
        })
    
    # Create peer collaboration relationships
    collaborations = [
        ("sarah-chen", "alex-rodriguez", "COLLABORATES_WITH", "AI/ML projects"),
        ("sarah-chen", "maria-garcia", "MENTORED_BY", "Research guidance"),
        ("alex-rodriguez", "james-kim", "COLLABORATES_WITH", "MLOps infrastructure"),
        ("emily-davis", "sarah-chen", "COLLABORATES_WITH", "Frontend AI integration"),
        ("maria-garcia", "david-wilson", "ADVISES", "AI ethics consultation")
    ]
    
    for person1_id, person2_id, relationship, context in collaborations:
        query = f"""
        MATCH (p1:Engineer {{id: $person1_id}})
        MATCH (p2:Engineer {{id: $person2_id}})
        CREATE (p1)-[r:{relationship}]->(p2)
        SET r.context = $context,
            r.created_at = datetime()
        """
        db.execute_write_query(query, {
            "person1_id": person1_id,
            "person2_id": person2_id, 
            "context": context
        })
    
    logger.info("Comprehensive sample data created successfully")
    logger.info("Data includes:")
    logger.info("- 7 engineers with diverse AI/ML and engineering skills")
    logger.info("- 6 projects covering various AI/ML domains")
    logger.info("- Management, collaboration, and project relationships")
    logger.info("- Embeddings for semantic search capabilities")


def get_data_summary() -> dict:
    """Get a summary of the data in the knowledge graph."""
    db = get_neo4j_connection()
    
    summary = {}
    
    # Count nodes
    node_counts = db.execute_query("""
        MATCH (n)
        RETURN labels(n)[0] as label, count(n) as count
        ORDER BY count DESC
    """)
    summary['nodes'] = {record['label']: record['count'] for record in node_counts}
    
    # Count relationships
    rel_counts = db.execute_query("""
        MATCH ()-[r]->()
        RETURN type(r) as relationship, count(r) as count
        ORDER BY count DESC
    """)
    summary['relationships'] = {record['relationship']: record['count'] for record in rel_counts}
    
    # Get sample skills for semantic search testing
    skills = db.execute_query("""
        MATCH (e:Engineer)
        UNWIND e.skills as skill
        RETURN DISTINCT skill
        ORDER BY skill
        LIMIT 20
    """)
    summary['sample_skills'] = [record['skill'] for record in skills]
    
    return summary