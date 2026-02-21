# app/knowledge_graph/postprocess/cleaner.py
from __future__ import annotations

from typing import Dict, List, Set, Tuple, Optional

from app.knowledge_graph.extraction.schema import (
    Entity,
    Relation,
    normalize_predicate,
)


def _canon(s: str) -> str:
    return (s or "").strip().lower()


def _entity_key(e: Entity) -> str:
    nn = getattr(e, "normalized_name", None)
    if nn:
        return _canon(nn)
    return _canon(getattr(e, "name", ""))


def clean_entities_relations(
    entities: List[Entity],
    relations: List[Relation],
    *,
    min_entity_confidence: float = 0.0,
    min_relation_confidence: float = 0.0,
    max_relations_per_entity: Optional[int] = None,
    drop_weak_related_to: bool = True,
    remove_isolated_entities: bool = True,
) -> Tuple[List[Entity], List[Relation]]:

    # ==========================================================
    # 1) Confidence Filtering
    # ==========================================================

    entities = [
        e for e in entities
        if float(getattr(e, "confidence", 0.0)) >= min_entity_confidence
    ]

    relations = [
        r for r in relations
        if float(getattr(r, "confidence", 0.0)) >= min_relation_confidence
    ]

    if not entities:
        return [], []

    # ==========================================================
    # 2) Entity Lookup Map
    # ==========================================================

    key_to_entity: Dict[str, Entity] = {}
    for e in entities:
        k = _entity_key(e)
        if k:
            key_to_entity[k] = e
        for a in getattr(e, "aliases", []) or []:
            ak = _canon(a)
            if ak:
                key_to_entity.setdefault(ak, e)

    valid_keys: Set[str] = set(key_to_entity.keys())

    # ==========================================================
    # 3) Relation Canonicalization + Pruning
    # ==========================================================

    cleaned: List[Relation] = []
    seen_edges: Set[Tuple[str, str, str]] = set()

    for r in relations:

        hk = _canon(r.head)
        tk = _canon(r.tail)

        if not hk or not tk:
            continue

        if hk == tk:
            continue  # remove self loops

        if hk not in valid_keys or tk not in valid_keys:
            continue

        head_ent = key_to_entity[hk]
        tail_ent = key_to_entity[tk]

        pred = normalize_predicate(
            getattr(r, "predicate", None) or getattr(r, "relation", "RELATED_TO")
        )

        if drop_weak_related_to and pred == "RELATED_TO":
            if float(getattr(r, "confidence", 0.0)) < 0.8:
                continue

        edge_key = (head_ent.name.lower(), pred, tail_ent.name.lower())
        if edge_key in seen_edges:
            continue

        seen_edges.add(edge_key)

        cleaned.append(
            Relation(
                head=head_ent.name,
                head_type=head_ent.type,
                relation=pred,
                tail=tail_ent.name,
                tail_type=tail_ent.type,
                evidence=r.evidence,
                confidence=r.confidence,
                chunk_id=r.chunk_id,
                page=r.page,
                source=r.source,
                meta=r.meta or {},
            )
        )

    if not cleaned:
        return entities, []

    # ==========================================================
    # 4) Cap relations per entity (degree control)
    # ==========================================================

    if max_relations_per_entity is not None:

        buckets: Dict[str, List[Relation]] = {}

        for r in cleaned:
            buckets.setdefault(r.head, []).append(r)
            buckets.setdefault(r.tail, []).append(r)

        keep_ids: Set[str] = set()

        for ent, rels in buckets.items():
            rels_sorted = sorted(
                rels,
                key=lambda x: float(getattr(x, "confidence", 0.0)),
                reverse=True,
            )[:max_relations_per_entity]

            for rr in rels_sorted:
                keep_ids.add(rr.id)

        cleaned = [r for r in cleaned if r.id in keep_ids]

    # ==========================================================
    # 5) Remove isolated entities
    # ==========================================================

    if remove_isolated_entities:

        connected: Set[str] = set()

        for r in cleaned:
            connected.add(r.head.lower())
            connected.add(r.tail.lower())

        entities = [
            e for e in entities
            if e.name.lower() in connected
        ]

    # ==========================================================
    # 6) Final Deterministic Sort
    # ==========================================================

    entities = sorted(
        entities,
        key=lambda e: (-float(e.confidence), e.normalized_name.lower()),
    )

    cleaned = sorted(
        cleaned,
        key=lambda r: (-float(r.confidence), r.head.lower(), r.tail.lower()),
    )

    return entities, cleaned