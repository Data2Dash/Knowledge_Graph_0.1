from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
from pydantic import BaseModel, Field

# --- Runtime Data Classes (Graph Pipeline Usage) ---

@dataclass(frozen=True)
class Entity:
    name: str
    type: str = "Concept"

@dataclass(frozen=True)
class Relation:
    head: str
    head_type: str
    relation: str
    tail: str
    tail_type: str
    evidence: Optional[str] = None

# --- Pydantic Validation Models (LLM Extraction) ---

class ExtractedEntity(BaseModel):
    id: str = Field(..., description="Unique slug for the entity")
    type: str = Field(..., description="Ontology class of the entity")
    desc: Optional[str] = Field(None, description="Brief definition context")

class ExtractedRelation(BaseModel):
    source: str = Field(..., description="ID of the source entity")
    target: str = Field(..., description="ID of the target entity")
    predicate: str = Field(..., description="Relationship type")
    evidence: str = Field(..., description="Exact quote supporting the relation")

class JointExtractionResult(BaseModel):
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relations: List[ExtractedRelation] = Field(default_factory=list)
