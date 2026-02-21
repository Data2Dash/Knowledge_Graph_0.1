# app/knowledge_graph/graph_rag/retriever.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import re
import math

from app.knowledge_graph.store.vector_store import InMemoryVectorStore

from app.knowledge_graph.embeddings.embedder import embed_texts, cosine


# ==========================================================
# Types
# ==========================================================

@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class RetrieverConfig:
    top_k: int = 6
    max_chunk_chars_each: int = 1200

    rerank_semantic: bool = True
    semantic_weight: float = 0.5
    store_weight: float = 0.5

    min_score_threshold: float = 0.05
    lexical_fallback: bool = True

    # how much of each chunk to embed for reranking (keep higher than display truncation)
    rerank_embed_chars: int = 2400


# ==========================================================
# Utils
# ==========================================================

_WORD_RE = re.compile(r"[A-Za-z0-9_\-]+")


def _clean_query(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip())


def _truncate(text: str, n: int) -> str:
    t = (text or "").strip()
    return t if len(t) <= n else t[:n].rstrip() + "…"


def _simple_lexical_score(query: str, text: str) -> float:
    q_tokens = set(_WORD_RE.findall(query.lower()))
    t_tokens = set(_WORD_RE.findall(text.lower()))
    if not q_tokens or not t_tokens:
        return 0.0
    overlap = len(q_tokens & t_tokens)
    return overlap / max(len(q_tokens), 1)


# ==========================================================
# Retrieval
# ==========================================================

def retrieve_chunks(
    vstore: InMemoryVectorStore,
    query: str,
    rc: Optional[RetrieverConfig] = None,
    *,
    cfg=None,  # kept for compatibility; ignored
) -> List[RetrievedChunk]:
    rc = rc or RetrieverConfig()
    q = _clean_query(query)
    if not q:
        return []

    # 1) First-pass vector retrieval
    raw = vstore.search(q, top_k=max(rc.top_k * 2, rc.top_k), cfg=cfg)
    if not raw:
        return []

    # Keep full text for rerank, truncated for display later
    full_texts = [text for _, text, _ in raw]
    chunk_ids = [str(cid) for cid, _, _ in raw]
    base_scores = [float(score) for _, _, score in raw]

    # 2) Optional semantic rerank
    final_scores = base_scores[:]
    if rc.rerank_semantic and full_texts:
        try:
            q_vec = embed_texts([q])[0]
            # embed a longer cap than UI display truncation
            to_embed = [_truncate(t, rc.rerank_embed_chars) for t in full_texts]
            c_vecs = embed_texts(to_embed)

            semantic_scores = []
            for i in range(len(to_embed)):
                try:
                    semantic_scores.append(float(cosine(q_vec, c_vecs[i])))
                except Exception:
                    semantic_scores.append(0.0)

            final_scores = [
                rc.store_weight * base_scores[i] + rc.semantic_weight * semantic_scores[i]
                for i in range(len(base_scores))
            ]
        except Exception:
            # Fallback: lexical rerank if enabled
            if rc.lexical_fallback:
                lex = [_simple_lexical_score(q, t) for t in full_texts]
                final_scores = [
                    rc.store_weight * base_scores[i] + rc.semantic_weight * lex[i]
                    for i in range(len(base_scores))
                ]
            else:
                final_scores = base_scores

    # 3) Build results with display truncation
    chunks = [
        RetrievedChunk(
            chunk_id=chunk_ids[i],
            text=_truncate(full_texts[i], rc.max_chunk_chars_each),
            score=float(final_scores[i]),
        )
        for i in range(len(chunk_ids))
    ]

    # stable deterministic sort
    chunks.sort(key=lambda x: (-x.score, x.chunk_id))

    # 4) Filter weak matches
    filtered = [c for c in chunks if not math.isnan(c.score) and c.score >= rc.min_score_threshold]

    return filtered[: rc.top_k]