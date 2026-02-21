from __future__ import annotations
import json, re
from typing import List
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from app.knowledge_graph.llm.prompts import RESEARCH_PAPER_RELATION_PROMPT
from app.knowledge_graph.extraction.schema import Relation, Entity
from app.knowledge_graph.extraction.validator import normalize_rel_type

def _extract_json(raw: str):
    raw = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if m:
        raw = m.group(1).strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start < 0 or end < 0 or end <= start:
        return []
    try:
        out = json.loads(raw[start:end+1])
        return out if isinstance(out, list) else []
    except Exception:
        return []

def extract_relations(llm: ChatGroq, text: str, entities: List[Entity], max_chars: int) -> List[Relation]:
    ent_list = [{"name": e.name, "type": e.type} for e in entities][:120]
    prompt = (
        RESEARCH_PAPER_RELATION_PROMPT
        + "\n\nEntity List (use these):\n"
        + json.dumps(ent_list, ensure_ascii=False)
        + "\n\nText:\n"
        + text[:max_chars]
    )
    msg = llm.invoke([HumanMessage(content=prompt)])
    raw = msg.content if hasattr(msg, "content") else str(msg)
    arr = _extract_json(raw)

    out: List[Relation] = []
    for it in arr:
        if not isinstance(it, dict):
            continue
        h = (it.get("head") or "").strip()
        t = (it.get("tail") or "").strip()
        r = normalize_rel_type(it.get("relation") or "RELATED_TO")
        ht = (it.get("head_type") or "Concept").strip()
        tt = (it.get("tail_type") or "Concept").strip()
        ev = (it.get("evidence") or "").strip() or None
        if h and t and r:
            out.append(Relation(h, ht, r, t, tt, ev))
    return out
