from __future__ import annotations
import re
from typing import List, Tuple
from app.knowledge_graph.extraction.schema import Entity, Relation


_only_number = re.compile(r"^\d+(\.\d+)?$")


def normalize_rel_type(r: str) -> str:
    return (r or "RELATED_TO").strip().upper().replace(" ", "_")

def dedupe_entities(entities: List[Entity]) -> List[Entity]:
    seen = set()
    out = []
    for e in entities:
        name = (e.name or "").strip()
        typ = (e.type or "Concept").strip()

        if not name:
            continue

        # 🚫 drop numeric-only entities like "28.4"
        if _only_number.match(name):
            continue

        key = (name.lower(), typ)
        if key in seen:
            continue
        seen.add(key)
        out.append(Entity(name=name, type=typ or "Concept"))
    return out

def dedupe_relations(relations: List[Relation]) -> List[Relation]:
    seen = set()
    out = []
    for r in relations:
        h = r.head.strip()
        t = r.tail.strip()
        rt = normalize_rel_type(r.relation)
        if not h or not t:
            continue
        key = (h.lower(), t.lower(), rt)
        if key in seen:
            continue
        seen.add(key)
        out.append(Relation(h, r.head_type or "Concept", rt, t, r.tail_type or "Concept", r.evidence))
    return out
