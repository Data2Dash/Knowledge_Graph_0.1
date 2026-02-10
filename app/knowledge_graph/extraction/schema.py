from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

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
