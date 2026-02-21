# app/knowledge_graph/extraction/relation_extractor.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from app.core.config import PipelineConfig
from app.core.logging import setup_logging
from app.core.settings import get_settings
from app.knowledge_graph.extraction.schema import (
    ALLOWED_PREDICATES,
    Entity,
    Relation,
    canon,
    norm_key,
    normalize_entity_type,
    normalize_predicate,
)
from app.knowledge_graph.llm.prompts import RESEARCH_PAPER_RELATION_PROMPT

LOGGER = setup_logging("knowledge_graph.relation_extractor")

_CODEBLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_JSON_ARR_RE = re.compile(r"\[[\s\S]*\]")
_MULTI_SPACE_RE = re.compile(r"\s+")


# ==========================================================
# Utilities
# ==========================================================

def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else (s[:n].rstrip() + "…")


def _norm_ws(s: str) -> str:
    return _MULTI_SPACE_RE.sub(" ", (s or "").strip())


def _evidence_supported(evidence: Optional[str], text: str) -> bool:
    """
    PDF text is messy; compare with whitespace-normalized, case-insensitive containment.
    """
    if not evidence:
        return False
    ev = _norm_ws(evidence).lower()
    tx = _norm_ws(text).lower()
    # Quick containment; avoids expensive fuzzy matching
    return ev in tx


def _extract_json_array(raw: str) -> List[Dict[str, Any]]:
    if not raw:
        return []

    raw = raw.strip()

    m = _CODEBLOCK_RE.search(raw)
    if m:
        raw = m.group(1).strip()

    if raw.startswith("[") and raw.endswith("]"):
        try:
            arr = json.loads(raw)
            return arr if isinstance(arr, list) else []
        except Exception:
            return []

    m2 = _JSON_ARR_RE.search(raw)
    if m2:
        try:
            arr = json.loads(m2.group(0))
            return arr if isinstance(arr, list) else []
        except Exception:
            return []

    return []


# ==========================================================
# Entity Grounding
# ==========================================================

def _entity_choices(entities: List[Entity], max_n: int) -> List[Dict[str, str]]:
    seen = set()
    out: List[Dict[str, str]] = []

    for e in entities:
        name = canon(e.name)
        typ = canon(e.type) or "Concept"
        if not name:
            continue

        key = (name.lower(), typ)
        if key in seen:
            continue

        seen.add(key)
        out.append({"name": name, "type": typ})

        if len(out) >= max_n:
            break

    return out


def _build_entity_index(entities: List[Entity]) -> Dict[str, Entity]:
    """
    Map normalized keys -> canonical entity
    (includes aliases to improve match).
    """
    idx: Dict[str, Entity] = {}
    for e in entities:
        idx[norm_key(e.name)] = e
        for a in getattr(e, "aliases", []) or []:
            nk = norm_key(a)
            if nk and nk not in idx:
                idx[nk] = e
    return idx


def _ground_entity(name: str, idx: Dict[str, Entity]) -> Optional[Entity]:
    nk = norm_key(name)
    if not nk:
        return None
    return idx.get(nk)


# ==========================================================
# Relation Builder
# ==========================================================

def _to_relations(
    items: List[Dict[str, Any]],
    idx: Dict[str, Entity],
    chunk_text: str,
    context: Dict[str, Any],
    cfg: PipelineConfig,
) -> List[Relation]:
    settings = get_settings()

    confidence_threshold = float(getattr(cfg, "relation_confidence_threshold", 0.5))
    max_relations = int(getattr(cfg, "max_relations_per_chunk", 50))

    rels: List[Relation] = []
    seen = set()

    chunk_id = context.get("chunk_id")
    page = context.get("page")
    source = context.get("source")
    section = context.get("section")

    for it in items or []:
        try:
            head_raw = it.get("head") or it.get("source") or ""
            tail_raw = it.get("tail") or it.get("target") or ""
            rel_raw = it.get("relation") or it.get("predicate") or "RELATED_TO"

            head_ent = _ground_entity(str(head_raw), idx)
            tail_ent = _ground_entity(str(tail_raw), idx)

            if head_ent is None or tail_ent is None:
                continue  # hallucination guard

            if head_ent.name.lower() == tail_ent.name.lower():
                continue  # no self loops

            pred = normalize_predicate(str(rel_raw))
            if pred not in ALLOWED_PREDICATES:
                pred = "RELATED_TO"

            evidence = it.get("evidence")
            if isinstance(evidence, str):
                evidence = evidence.strip()[:400]
            else:
                evidence = None

            # Only enforce evidence presence for weak predicates, or when settings wants strict JSON-only
            if evidence and not _evidence_supported(evidence, chunk_text):
                evidence = None

            conf_raw = it.get("confidence", 0.7)
            try:
                conf = float(conf_raw)
            except Exception:
                conf = 0.7

            if conf < confidence_threshold:
                continue

            key = (head_ent.name.lower(), pred, tail_ent.name.lower())
            if key in seen:
                continue
            seen.add(key)

            rels.append(
                Relation(
                    head=head_ent.name,
                    head_type=normalize_entity_type(head_ent.type),
                    relation=pred,
                    tail=tail_ent.name,
                    tail_type=normalize_entity_type(tail_ent.type),
                    evidence=evidence,
                    confidence=conf,
                    chunk_id=str(chunk_id) if chunk_id else None,
                    page=int(page) if isinstance(page, int) else None,
                    source=str(source) if source else None,
                    meta={"section": section} if section else {},
                )
            )

            if len(rels) >= max_relations:
                break

        except Exception:
            continue

    rels.sort(key=lambda r: -float(r.confidence))
    return rels


# ==========================================================
# Main Extraction
# ==========================================================

def extract_relations(
    llm: ChatGroq,
    chunk_text: str,
    entities: List[Entity],
    cfg: PipelineConfig,
    *,
    context: Optional[dict] = None,
) -> List[Relation]:
    context = context or {}

    if not entities:
        return []

    chunk = _truncate(chunk_text, cfg.max_chunk_chars_for_llm)

    max_entities = int(getattr(cfg, "relation_max_entities_in_prompt", 30))
    choices = _entity_choices(entities, max_n=max_entities)

    prompt = (
        RESEARCH_PAPER_RELATION_PROMPT
        + "\n\nEntity list (JSON):\n"
        + json.dumps(choices, ensure_ascii=False)
        + "\n\nText:\n"
        + chunk
    )

    msg = llm.invoke([HumanMessage(content=prompt)])
    raw = msg.content if hasattr(msg, "content") else str(msg)

    items = _extract_json_array(raw)

    idx = _build_entity_index(entities)
    rels = _to_relations(items, idx, chunk, context, cfg)

    LOGGER.info(
        "Relations extracted",
        extra={
            "count": len(rels),
            "raw_items": len(items),
            **{k: v for k, v in context.items() if k in ("chunk_id", "page", "source")},
        },
    )

    return rels