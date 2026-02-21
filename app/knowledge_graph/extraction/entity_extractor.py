# app/knowledge_graph/extraction/entity_extractor.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from app.core.config import PipelineConfig
from app.core.logging import setup_logging
from app.knowledge_graph.llm.prompts import RESEARCH_PAPER_ENTITY_PROMPT
from app.knowledge_graph.extraction.schema import (
    ALLOWED_ENTITY_TYPES,
    Entity,
    canon,
    norm_key,
    normalize_entity_type,
)

LOGGER = setup_logging("knowledge_graph.entity_extractor")

_CODEBLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")
_JSON_ARR_RE = re.compile(r"\[[\s\S]*\]")
_MULTI_SPACE_RE = re.compile(r"\s+")
_BAD_NAME_RE = re.compile(r"^(page\s*\d+|\d+)$", re.IGNORECASE)
_HASH_LIKE_RE = re.compile(r"^[a-f0-9]{12,}$", re.IGNORECASE)
_PAGE_MARK_RE = re.compile(r"^\[PAGE:\d+\]$", re.IGNORECASE)

# ==========================================================
# Cleaning / guards
# ==========================================================

def _clean_name(name: str) -> str:
    n = canon(name)
    if len(n) < 2:
        return ""
    if _BAD_NAME_RE.match(n):
        return ""
    return n


def _norm_ws(s: str) -> str:
    return _MULTI_SPACE_RE.sub(" ", (s or "").strip())


def _appears_in_text(candidate: str, text: str) -> bool:
    """
    Hallucination guard:
    Use whitespace-normalized, case-insensitive containment.
    """
    if not candidate:
        return False
    c = _norm_ws(candidate).lower()
    t = _norm_ws(text).lower()
    return c in t


# ==========================================================
# JSON extraction
# ==========================================================

def _extract_json_candidates(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []

    m = _CODEBLOCK_RE.search(raw)
    if m:
        raw = m.group(1).strip()

    candidates: List[str] = []

    ma = _JSON_ARR_RE.search(raw)
    if ma:
        candidates.append(ma.group(0))

    mo = _JSON_OBJ_RE.search(raw)
    if mo:
        candidates.append(mo.group(0))

    candidates.append(raw)

    seen = set()
    out: List[str] = []
    for c in candidates:
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _parse_entities_payload(payload: Any) -> List[Dict[str, Any]]:
    if payload is None:
        return []

    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        for key in ("entities", "items", "data", "result"):
            v = payload.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        if "name" in payload or "type" in payload:
            return [payload]

    return []


def _extract_entities_from_raw(raw: str) -> List[Dict[str, Any]]:
    for cand in _extract_json_candidates(raw):
        try:
            payload = json.loads(cand)
            arr = _parse_entities_payload(payload)
            if arr:
                return arr
        except Exception:
            continue
    return []


# ==========================================================
# Dedupe
# ==========================================================

def _dedupe_entities(entities: List[Entity]) -> List[Entity]:
    best: Dict[Tuple[str, str], Entity] = {}

    for ent in entities:
        key = (norm_key(ent.name), ent.type)

        if key not in best:
            best[key] = ent
            continue

        cur = best[key]
        winner = ent if ent.confidence > cur.confidence else cur
        loser = cur if winner is ent else ent

        merged_aliases = list(dict.fromkeys((winner.aliases or []) + (loser.aliases or [])))
        merged_meta = dict(loser.meta or {})
        merged_meta.update(winner.meta or {})

        best[key] = Entity(
            name=winner.name,
            type=winner.type,
            confidence=winner.confidence,
            aliases=merged_aliases,
            evidence=winner.evidence or loser.evidence,
            evidence_obj=winner.evidence_obj or loser.evidence_obj,
            meta=merged_meta,
        )

    out = list(best.values())
    out.sort(key=lambda x: (-float(x.confidence), norm_key(x.name)))
    return out


# ==========================================================
# Main extraction
# ==========================================================

def extract_entities(
    llm: ChatGroq,
    text: str,
    cfg: PipelineConfig,
    *,
    context: Optional[dict] = None,
) -> List[Entity]:
    context = context or {}

    t = (text or "").strip()
    if not t:
        return []

    # Budget the chunk we send to the model
    hard_cap = int(getattr(cfg, "max_chunk_chars_for_llm", 6000))
    t = t[: max(200, hard_cap)]

    prompt = (
        "Return ONLY valid JSON under key `entities`.\n"
        "Each entity must have: name, type.\n"
        "Optional: aliases, confidence (0..1), evidence.\n"
        f"Allowed types: {sorted(ALLOWED_ENTITY_TYPES)}\n\n"
        + RESEARCH_PAPER_ENTITY_PROMPT
        + "\n\nText:\n"
        + t
    )

    msg = llm.invoke([HumanMessage(content=prompt)])
    raw = msg.content if hasattr(msg, "content") else str(msg)

    arr = _extract_entities_from_raw(raw)

    extracted: List[Entity] = []

    confidence_threshold = float(getattr(cfg, "entity_confidence_threshold", 0.5))
    max_entities = int(getattr(cfg, "max_entities_per_chunk", 60))

    for it in arr:
        try:
            name = _clean_name(str(it.get("name") or ""))
            if not name:
                continue

            # hallucination guard: name OR one alias must appear
            aliases_raw = it.get("aliases") or []
            aliases: List[str] = []
            if isinstance(aliases_raw, list):
                for a in aliases_raw:
                    ca = _clean_name(str(a))
                    if ca and ca.lower() != name.lower():
                        aliases.append(ca)

            if not _appears_in_text(name, t):
                if not any(_appears_in_text(a, t) for a in aliases):
                    continue

            typ = normalize_entity_type(str(it.get("type") or "Concept"))
            if typ not in ALLOWED_ENTITY_TYPES:
                typ = "Concept"

            try:
                confidence = float(it.get("confidence", 0.75))
            except Exception:
                confidence = 0.75
            if confidence < confidence_threshold:
                continue

            ev_text = None
            if isinstance(it.get("evidence"), str):
                ev_text = it["evidence"].strip()[:400]

            meta = it.get("meta") if isinstance(it.get("meta"), dict) else {}
            if _HASH_LIKE_RE.match(name) or _PAGE_MARK_RE.match(name):
                continue
            extracted.append(
                Entity(
                    name=name,
                    type=typ,
                    confidence=confidence,
                    aliases=aliases,
                    evidence=ev_text,
                    evidence_obj=None,
                    meta=meta,
                )
            )

            if len(extracted) >= max_entities:
                break

        except Exception:
            continue

    deduped = _dedupe_entities(extracted)

    LOGGER.info(
        "Entities extracted",
        extra={
            "count": len(deduped),
            "raw_count": len(arr),
            **{k: v for k, v in context.items() if k in ("chunk_id", "page", "source")},
        },
    )

    return deduped