# app/knowledge_graph/graph_rag/context_builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union, Any


@dataclass(frozen=True, slots=True)
class ContextConfig:
    max_chunk_chars_each: int = 1200
    max_total_context_chars: int = 14000
    include_scores: bool = True
    heading: str = "Retrieved Evidence Chunks"


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rstrip() + "…"


def _as_triplets(
    retrieved_chunks: Sequence[Union[Tuple[str, str, float], Any]]
) -> List[Tuple[str, str, float]]:
    """
    Normalize input into List[(chunk_id, text, score)].

    Accepts:
    - (chunk_id, text, score)
    - RetrievedChunk-like objects with .chunk_id .text .score
    """
    out: List[Tuple[str, str, float]] = []
    for item in retrieved_chunks or []:
        if isinstance(item, tuple) and len(item) == 3:
            cid, txt, score = item
            out.append((str(cid), str(txt), float(score)))
            continue

        # RetrievedChunk dataclass (or similar)
        cid = getattr(item, "chunk_id", None)
        txt = getattr(item, "text", None)
        score = getattr(item, "score", None)
        if cid is not None and txt is not None and score is not None:
            try:
                out.append((str(cid), str(txt), float(score)))
            except Exception:
                continue

    return out


def build_context(
    retrieved_chunks: Sequence[Union[Tuple[str, str, float], Any]],
    graph_facts_text: str = "",
    *,
    cc: Optional[ContextConfig] = None,
) -> str:
    """
    Build LLM context from retrieved chunks + optional graph facts.

    Input supports:
    - List[(chunk_id, text, score)]
    - List[RetrievedChunk]
    Output: Markdown context (bounded by max_total_context_chars)
    """
    cc = cc or ContextConfig()
    triplets = _as_triplets(retrieved_chunks)

    parts: List[str] = [f"# {cc.heading}"]

    for cid, txt, score in triplets:
        text = _truncate(txt, cc.max_chunk_chars_each)
        if cc.include_scores:
            parts.append(f"\n## [Chunk {cid}] score={score:.3f}\n{text}\n")
        else:
            parts.append(f"\n## [Chunk {cid}]\n{text}\n")

    if graph_facts_text:
        rel = graph_facts_text.strip()
        if rel:
            parts.append("\n# Graph Facts\n" + rel)

    ctx = "\n".join(parts).strip()

    if len(ctx) > cc.max_total_context_chars:
        ctx = ctx[: cc.max_total_context_chars].rstrip() + "\n…(truncated)"

    return ctx