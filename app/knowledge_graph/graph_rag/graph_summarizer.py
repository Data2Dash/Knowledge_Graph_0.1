# app/knowledge_graph/graph_rag/graph_summarizer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import hashlib

import networkx as nx

from app.knowledge_graph.extraction.schema import Entity, Relation


# ==========================================================
# Types
# ==========================================================

@dataclass(frozen=True, slots=True)
class CommunitySummary:
    community_id: str
    title: str
    summary: str
    nodes: List[str]
    edges: List[str]
    meta: Dict[str, str]


# ==========================================================
# Prompt (Grounded)
# ==========================================================

_SUMMARY_PROMPT = """\
You are summarizing a knowledge graph community extracted from a research paper.

STRICT RULES:
- Use ONLY the provided triples.
- Do NOT invent facts.
- If triples are sparse, say that clearly.
- Be precise and factual.

Output format:
Title: short name (<= 8 words)
Summary:
- bullet 1
- bullet 2
- bullet 3

Triples:
{triples}
"""


# ==========================================================
# Graph Building (Directed + Weighted)
# ==========================================================

def _build_graph(relations: List[Relation]) -> nx.DiGraph:
    g = nx.DiGraph()

    for r in relations or []:
        h = getattr(r, "head", None)
        t = getattr(r, "tail", None)
        rel = getattr(r, "predicate", None) or getattr(r, "relation", "RELATED_TO")
        conf_raw = getattr(r, "confidence", 0.7)

        if not h or not t:
            continue

        try:
            conf = float(conf_raw) if conf_raw is not None else 0.7
        except Exception:
            conf = 0.7

        g.add_edge(str(h), str(t), relation=str(rel), weight=max(0.0, min(1.0, conf)))

    return g


# ==========================================================
# Community Detection
# ==========================================================

def _communities(g: nx.DiGraph) -> List[List[str]]:
    undirected = g.to_undirected()
    comps = [list(c) for c in nx.connected_components(undirected)]
    comps.sort(key=len, reverse=True)
    return comps


# ==========================================================
# Triple Extraction (Signal-Aware)
# ==========================================================

_HIGH_SIGNAL = {
    "PROPOSES", "INTRODUCES", "IMPROVES_OVER",
    "ACHIEVES", "EVALUATES_ON", "USES",
    "OUTPERFORMS", "TRAINED_ON", "FINE_TUNED_ON",
    "WRITTEN_BY", "AFFILIATED_WITH",
}

def _extract_triples(g: nx.DiGraph, nodes: List[str], max_edges: int = 60) -> List[str]:
    sub = g.subgraph(nodes)

    edges = []
    for u, v, data in sub.edges(data=True):
        rel = str(data.get("relation", "RELATED_TO"))
        weight = float(data.get("weight", 0.0))

        bonus = 1.0 if rel in _HIGH_SIGNAL else 0.0
        score = weight + bonus

        edges.append((score, str(u), rel, str(v)))

    edges.sort(key=lambda x: (-x[0], x[1], x[3]))

    triples: List[str] = []
    seen = set()

    for _, u, rel, v in edges:
        tri = f"{u} {rel} {v}"
        if tri in seen:
            continue
        seen.add(tri)
        triples.append(tri)
        if len(triples) >= max_edges:
            break

    return triples


# ==========================================================
# Summarization
# ==========================================================

def _llm_summarize(llm, triples: List[str]) -> Tuple[str, str]:
    """
    llm must support: llm.invoke([HumanMessage(content=...)])
    We avoid importing LangChain message classes here to keep module lightweight.
    """
    if not triples:
        return ("Empty Community", "- No grounded facts available.")

    # Lazy import to avoid hard dependency if user doesn't use LLM summaries
    from langchain_core.messages import HumanMessage

    prompt = _SUMMARY_PROMPT.format(triples="\n".join(f"- {t}" for t in triples[:80]))
    msg = llm.invoke([HumanMessage(content=prompt)])
    raw = msg.content if hasattr(msg, "content") else str(msg)

    title = "Community"
    for line in raw.splitlines():
        if line.lower().startswith("title:"):
            title = (line.split(":", 1)[1].strip() or title)
            break

    # Clamp sizes
    title = title[:60]
    summary = raw.strip()[:2000]
    return title, summary


def _rule_summarize(triples: List[str]) -> Tuple[str, str]:
    if not triples:
        return ("Empty Community", "- No grounded facts available.")
    title = "Key Relations"
    bullets = "\n".join(f"- {t}" for t in triples[:8])
    return title, bullets


def _stable_community_id(nodes: List[str]) -> str:
    key = "|".join(sorted(nodes))[:5000]
    return hashlib.sha1(key.encode("utf-8", "ignore")).hexdigest()[:10]


# ==========================================================
# Main Entry
# ==========================================================

def summarize_communities(
    entities: List[Entity],
    relations: List[Relation],
    *,
    llm: Optional[object] = None,
    max_communities: int = 10,
    max_edges_per_community: int = 60,
) -> List[CommunitySummary]:
    g = _build_graph(relations)
    if g.number_of_nodes() == 0:
        return []

    comps = _communities(g)[: max(1, int(max_communities))]
    summaries: List[CommunitySummary] = []

    for nodes in comps:
        triples = _extract_triples(g, nodes, max_edges=max_edges_per_community)

        if llm is not None:
            try:
                title, summary = _llm_summarize(llm, triples)
            except Exception:
                title, summary = _rule_summarize(triples)
        else:
            title, summary = _rule_summarize(triples)

        summaries.append(
            CommunitySummary(
                community_id=_stable_community_id(nodes),
                title=title,
                summary=summary,
                nodes=nodes,
                edges=triples,
                meta={"nodes": str(len(nodes)), "edges": str(len(triples))},
            )
        )

    return summaries