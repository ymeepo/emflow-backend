""" Core domain models for the EM Tools knowledge graph. """

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class EntityType(Enum):
    ENGINEER = "Engineer"
    PROJECT = "Project"


class ProjectStatus(Enum):
    ONGOING = "ongoing"
    COMPLETED = "completed"
    PAUSED = "paused"


class ProjectStage(Enum):
    DISCOVERY = "Discovery"
    PROTOTYPE = "Prototype"
    VALIDATION = "Validation"
    PRODUCTION = "Production"
    BETA = "Beta"
    IMPLEMENTATION = "Implementation"


class EngineeerLevel(Enum):
    ENGINEER_I = "Engineer I"
    ENGINEER_II = "Engineer II"
    SENIOR_ENGINEER = "Senior Engineer"
    STAFF_ENGINEER = "Staff Engineer"
    PRINCIPAL_ENGINEER = "Principal Engineer"
    MANAGER = "Manager"
    SENIOR_MANAGER = "Senior Manager"


@dataclass
class Engineer:
    id: str
    name: str
    position: str
    level: str
    team: str
    skills: List[str]
    expertise: str
    tenure: str
    status: str
    email: str
    embedding: Optional[List[float]] = None


@dataclass
class Project:
    id: str
    name: str
    description: str
    stage: str
    status: str
    technologies: List[str]
    business_value: str
    priority: str
    embedding: Optional[List[float]] = None


@dataclass
class Relationship:
    from_entity_id: str
    to_entity_id: str
    relationship_type: str
    details: Dict[str, Any]


@dataclass
class SemanticSearchRequest:
    query: str
    entity_type: EntityType = EntityType.ENGINEER
    top_k: int = 5
    similarity_threshold: float = 0.7


@dataclass
class SemanticSearchResult:
    id: str
    name: str
    similarity_score: float
    entity_type: EntityType
    details: Dict[str, Any]


@dataclass
class RelationshipSearchRequest:
    entity_id: str
    entity_type: EntityType = EntityType.ENGINEER
    relationship_types: Optional[List[str]] = None
    max_depth: int = 2


@dataclass
class RelationshipSearchResult:
    entity: Dict[str, Any]
    relationships: List[Dict[str, Any]]


@dataclass
class AgentRequest:
    prompt: str
    user_id: str = "anonymous"


@dataclass
class AgentResponse:
    type: str  # status, data, result, error, complete
    message: str
    data: Optional[Dict[str, Any]] = None