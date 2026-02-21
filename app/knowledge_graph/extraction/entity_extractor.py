from __future__ import annotations
import json, re
from typing import List
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from app.knowledge_graph.llm.prompts import RESEARCH_PAPER_ENTITY_PROMPT
from app.knowledge_graph.extraction.schema import Entity

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

def extract_entities(llm: ChatGroq, text: str, max_chars: int) -> List[Entity]:
    msg = llm.invoke([HumanMessage(content=RESEARCH_PAPER_ENTITY_PROMPT + "\n\nText:\n" + text[:max_chars])])
    raw = msg.content if hasattr(msg, "content") else str(msg)
    arr = _extract_json(raw)
    out: List[Entity] = []
    for it in arr:
        if isinstance(it, dict):
            name = (it.get("name") or "").strip()
            typ = (it.get("type") or "Concept").strip()
            if name:
                out.append(Entity(name=name, type=typ))
    return out
