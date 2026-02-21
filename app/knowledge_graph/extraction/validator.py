# app/knowledge_graph/extraction/validator.py
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from app.knowledge_graph.extraction.schema import (
    ALLOWED_PREDICATES,
    Entity,
    Relation,
    build_entity_name_index,
    canon,
    norm_key,
    normalize_entity_type,
    normalize_predicate,
)

_only_number = re.compile(r"^\d+(\.\d+)?$")

# Generic junk entities that LLMs often produce
_JUNK_ENTITIES = {
    "this paper", "the paper", "paper",
    "our method", "our approach", "our model",
    "method", "approach", "model", "system", "framework",
    "algorithm", "pipeline",
    "task", "dataset", "benchmark", "metric",
    "result", "results",
    "introduction", "conclusion", "abstract",
}

# If these appear with low confidence and no evidence, drop (too weak)
_WEAK_PREDICATES = {"RELATED_TO", "MENTIONS"}

# confidence thresholds
_MIN_ENTITY_CONF = 0.25
_MIN_REL_CONF = 0.25
_MIN_REL_CONF_WEAK = 0.55  # require higher confidence for RELATED_TO/MENTIONS


def _canon_text(s: str) -> str:
    # Use schema.canon for consistency
    return canon(s)


def _is_junk_entity(name: str) -> bool:
    n = _canon_text(name).lower()
    if not n:
        return True
    if _only_number.match(n):
        return True
    if n in _JUNK_ENTITIES:
        return True
    if len(n) <= 2:
        return True
    return False


def dedupe_entities(entities: List[Entity]) -> List[Entity]:
    """
    Dedupe using normalized name + type.
    Keeps best (highest confidence) and merges aliases/meta conservatively.
    """
    best: Dict[Tuple[str, str], Entity] = {}

    for e in entities or []:
        try:
            name = _canon_text(getattr(e, "name", "") or "")
            if _is_junk_entity(name):
                continue

            typ = normalize_entity_type(getattr(e, "type", "") or "Concept")
            conf = float(getattr(e, "confidence", 0.75) or 0.75)
            if conf < _MIN_ENTITY_CONF:
                continue

            norm = (getattr(e, "normalized_name", None) or name).strip().lower()
            key = (norm, typ)

            if key not in best:
                best[key] = Entity(
                    name=name,
                    type=typ,
                    id=getattr(e, "id", "") or "",
                    confidence=conf,
                    aliases=list(getattr(e, "aliases", []) or []),
                    evidence=getattr(e, "evidence", None),
                    evidence_obj=getattr(e, "evidence_obj", None),
                    meta=getattr(e, "meta", {}) or {},
                )
                continue

            cur = best[key]
            cur_conf = float(getattr(cur, "confidence", 0.75) or 0.75)

            # choose winner by confidence
            if conf > cur_conf:
                winner, loser = e, cur
            else:
                winner, loser = cur, e

            merged_aliases = list(
                dict.fromkeys(
                    (getattr(winner, "aliases", []) or []) + (getattr(loser, "aliases", []) or [])
                )
            )

            merged_meta = dict(getattr(loser, "meta", {}) or {})
            merged_meta.update(getattr(winner, "meta", {}) or {})

            ev = getattr(winner, "evidence", None) or getattr(loser, "evidence", None)
            ev_obj = getattr(winner, "evidence_obj", None) or getattr(loser, "evidence_obj", None)

            best[key] = Entity(
                name=_canon_text(getattr(winner, "name", "") or ""),
                type=normalize_entity_type(getattr(winner, "type", "") or typ),
                id=getattr(winner, "id", "") or "",
                confidence=float(getattr(winner, "confidence", 0.75) or 0.75),
                aliases=merged_aliases,
                evidence=ev,
                evidence_obj=ev_obj,
                meta=merged_meta,
            )
        except Exception:
            continue

    out = list(best.values())
    out.sort(key=lambda x: (-float(getattr(x, "confidence", 0.0)), getattr(x, "normalized_name", x.name).lower()))
    return out


def _entity_type_maps(entities: List[Entity]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns:
    - name_index: normalized_key -> canonical entity name
    - type_by_norm: normalized_key -> entity type
    """
    name_index = build_entity_name_index(entities)
    type_by_norm: Dict[str, str] = {}
    for e in entities:
        type_by_norm[norm_key(e.name)] = e.type
        for a in e.aliases or []:
            nk = norm_key(a)
            if nk and nk not in type_by_norm:
                type_by_norm[nk] = e.type
    return name_index, type_by_norm


def dedupe_relations(relations: List[Relation], entities: Optional[List[Entity]] = None) -> List[Relation]:
    """
    Validate + dedupe relations:
    - normalize predicate using schema.normalize_predicate
    - drop self-loops
    - canonicalize head/tail to entity list when available
    - infer head_type/tail_type from entity list when available
    - require evidence for weak predicates unless confidence is high
    - dedupe by (head, predicate, tail, chunk_id)
    """
    entities = entities or []
    name_index, type_by_norm = _entity_type_maps(entities) if entities else ({}, {})

    seen: Set[Tuple[str, str, str, Optional[str]]] = set()
    out: List[Relation] = []

    for r in relations or []:
        try:
            raw_h = getattr(r, "head", "") or ""
            raw_t = getattr(r, "tail", "") or ""
            h0 = _canon_text(raw_h)
            t0 = _canon_text(raw_t)
            if not h0 or not t0:
                continue

            # remove self-loop
            if h0.lower() == t0.lower():
                continue

            # canonicalize to known entities (best effort)
            h_key = norm_key(h0)
            t_key = norm_key(t0)
            h = name_index.get(h_key, h0)
            t = name_index.get(t_key, t0)

            # infer types from entities if possible, else fall back to provided/Concept
            ht = type_by_norm.get(h_key) or normalize_entity_type(getattr(r, "head_type", "") or "Concept")
            tt = type_by_norm.get(t_key) or normalize_entity_type(getattr(r, "tail_type", "") or "Concept")

            # normalize predicate (prefer r.predicate if present)
            raw_rel = getattr(r, "predicate", None) or getattr(r, "relation", None) or "RELATED_TO"
            pred = normalize_predicate(str(raw_rel))
            if pred not in ALLOWED_PREDICATES:
                pred = "RELATED_TO"

            # confidence
            try:
                conf_f = float(getattr(r, "confidence", 0.7) if getattr(r, "confidence", None) is not None else 0.7)
            except Exception:
                conf_f = 0.7
            if conf_f < _MIN_REL_CONF:
                continue

            evidence = getattr(r, "evidence", None)
            if isinstance(evidence, str):
                evidence = evidence.strip() or None
            else:
                evidence = None

            # weak predicate rule
            if pred in _WEAK_PREDICATES and conf_f < _MIN_REL_CONF_WEAK and evidence is None:
                continue

            chunk_id = getattr(r, "chunk_id", None)
            key = (h.lower(), pred, t.lower(), str(chunk_id) if chunk_id is not None else None)
            if key in seen:
                continue
            seen.add(key)

            meta = getattr(r, "meta", {}) or {}
            if not isinstance(meta, dict):
                meta = {}

            out.append(
                Relation(
                    head=h,
                    head_type=ht,
                    relation=pred,  # keep backwards field aligned
                    tail=t,
                    tail_type=tt,
                    evidence=evidence,
                    id=getattr(r, "id", "") or "",
                    confidence=conf_f,
                    chunk_id=str(chunk_id) if chunk_id is not None else None,
                    page=getattr(r, "page", None),
                    source=getattr(r, "source", None),
                    meta=meta,
                )
            )
        except Exception:
            continue

    out.sort(key=lambda x: -float(getattr(x, "confidence", 0.0)))
    return out