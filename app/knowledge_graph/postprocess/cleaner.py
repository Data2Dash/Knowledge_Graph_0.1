from __future__ import annotations
from typing import List, Tuple
from app.knowledge_graph.extraction.schema import Entity, Relation
from app.knowledge_graph.extraction.validator import dedupe_entities, dedupe_relations

def clean_entities_relations(entities: List[Entity], relations: List[Relation]) -> Tuple[List[Entity], List[Relation]]:
    entities = dedupe_entities(entities)
    relations = dedupe_relations(relations)
    return entities, relations
