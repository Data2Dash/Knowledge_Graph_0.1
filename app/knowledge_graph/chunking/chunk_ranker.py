# app/knowledge_graph/chunking/chunk_ranker.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import math
import re

from app.core.config import PipelineConfig
from app.knowledge_graph.chunking.semantic_chunker import Chunk

# Optional semantic reranking (works if embedder is available)
try:
    from app.knowledge_graph.embeddings.embedder import embed_texts, cosine
except Exception:
    embed_texts = None
    cosine = None


_SECTION_BOOSTS = {
    "abstract": 6.0,
    "introduction": 2.0,
    "background": 1.5,
    "related work": 1.5,
    "method": 5.0,
    "methods": 5.0,
    "approach": 4.5,
    "model": 3.0,
    "architecture": 3.0,
    "experiments": 5.0,
    "experimental setup": 4.0,
    "results": 5.0,
    "evaluation": 5.0,
    "discussion": 2.0,
    "analysis": 3.0,
    "ablation": 4.0,
    "limitations": 2.5,
    "conclusion": 3.0,
}

_GENERIC_TERMS = {
    "algorithm": 2.0,
    "framework": 1.5,
    "pipeline": 1.5,
    "objective": 1.5,
    "loss": 2.0,
    "optimization": 1.5,
    "regularization": 1.5,
    "hyperparameter": 1.5,
    "inference": 1.5,
    "training": 1.5,
    "dataset": 2.0,
    "benchmark": 2.5,
    "protocol": 1.0,
    "evaluation": 2.5,
    "metric": 2.0,
    "baseline": 2.5,
    "comparison": 1.5,
    "we propose": 3.0,
    "we introduce": 3.0,
    "our method": 2.5,
    "novel": 1.5,
    "state-of-the-art": 3.0,
    "sota": 3.0,
    "generalization": 1.5,
    "robust": 1.5,
    "efficiency": 1.5,
    "scalable": 1.5,
}

_NUMERIC_RE = re.compile(r"\b\d+(\.\d+)?\b")
_PERCENT_RE = re.compile(r"\b\d+(\.\d+)?\s*%")
_TABLE_FIG_RE = re.compile(r"\b(table|figure|fig\.)\s*\d+\b", re.IGNORECASE)
_CITATION_RE = re.compile(r"\[[0-9,\s]+\]|\([A-Z][A-Za-z]+,\s*\d{4}\)")
_EQUATION_RE = re.compile(r"(=|\bmin\b|\bmax\b|\bargmin\b|\bargmax\b|\blambda\b|∑|∫|→)")
_CODE_RE = re.compile(r"```|def\s+\w+\(|class\s+\w+\(|import\s+\w+")
_HEADING_HASH_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)


@dataclass(frozen=True)
class RankedChunk:
    score: float
    chunk: Chunk


def _normalize_len(text_len: int) -> float:
    if text_len <= 0:
        return 0.0
    x = text_len
    return max(0.6, min(1.15, 0.85 + 0.08 * math.log(max(x, 10), 10)))


def _section_boost(text: str) -> float:
    t = (text or "").lower()
    boost = 0.0
    head = t[:500]
    for sec, w in _SECTION_BOOSTS.items():
        if sec in head:
            boost += w
    if _HEADING_HASH_RE.search(text or ""):
        boost += 0.8
    return boost


def _term_score(text: str) -> float:
    t = (text or "").lower()
    score = 0.0
    for term, w in _GENERIC_TERMS.items():
        if term in t:
            score += w
    return score


def _signal_score(text: str) -> float:
    t = text or ""
    score = 0.0

    n_nums = len(_NUMERIC_RE.findall(t))
    n_pct = len(_PERCENT_RE.findall(t))
    n_tbl = len(_TABLE_FIG_RE.findall(t))
    n_cit = len(_CITATION_RE.findall(t))
    n_eq = len(_EQUATION_RE.findall(t))
    n_code = len(_CODE_RE.findall(t))

    score += 1.2 * math.log1p(n_nums)
    score += 1.5 * math.log1p(n_pct)
    score += 1.0 * math.log1p(n_tbl)
    score += 0.6 * math.log1p(n_eq)
    score += 0.4 * math.log1p(n_code)
    score += min(2.0, 0.25 * n_cit)

    return score


def _lexical_query_score(text: str, query: str) -> float:
    if not query:
        return 0.0
    t = (text or "").lower()
    q = query.lower()
    q_words = {w for w in re.findall(r"[a-zA-Z0-9_]+", q) if len(w) > 3}
    if not q_words:
        return 0.0
    overlap = sum(1 for w in q_words if w in t)
    return 1.8 * overlap


def _embed_texts_safe(texts: List[str], cfg: Optional[PipelineConfig]):
    """
    Supports both:
      embed_texts(texts, cfg)
      embed_texts(texts)
    """
    if embed_texts is None:
        return None
    try:
        return embed_texts(texts, cfg)  # type: ignore[misc]
    except TypeError:
        return embed_texts(texts)  # type: ignore[call-arg]


def _semantic_query_score(chunks: List[Chunk], query: str, cfg: Optional[PipelineConfig]) -> List[float]:
    if not query or embed_texts is None or cosine is None:
        return [0.0] * len(chunks)

    try:
        q_emb = _embed_texts_safe([query], cfg)[0]
        c_embs = _embed_texts_safe([c.text for c in chunks], cfg)
        if not c_embs:
            return [0.0] * len(chunks)
    except Exception:
        return [0.0] * len(chunks)

    qv = getattr(q_emb, "values", None)
    if not isinstance(qv, list) or not qv:
        return [0.0] * len(chunks)

    scores: List[float] = []
    for e in c_embs:
        ev = getattr(e, "values", None)
        try:
            scores.append(4.0 * float(cosine(qv, ev)))
        except Exception:
            scores.append(0.0)
    return scores


def rank_chunks(
    chunks: List[Chunk],
    cfg: Optional[PipelineConfig] = None,
    query: Optional[str] = None,
    use_semantic_rerank: bool = True,
) -> List[Chunk]:
    """
    Works with your flat PipelineConfig:
      cfg.prioritize_top_k
      cfg.max_total_chunks
    """
    if not chunks:
        return []

    cfg = cfg or PipelineConfig()

    sem_scores = (
        _semantic_query_score(chunks, query or "", cfg)
        if (query and use_semantic_rerank)
        else [0.0] * len(chunks)
    )

    ranked: List[RankedChunk] = []
    for i, ch in enumerate(chunks):
        text = ch.text or ""

        base = _section_boost(text) + _term_score(text) + _signal_score(text)
        qlex = _lexical_query_score(text, query or "")
        qsem = sem_scores[i] if i < len(sem_scores) else 0.0
        length_norm = _normalize_len(len(text))

        score = (base + qlex + qsem) * length_norm
        ranked.append(RankedChunk(score=score, chunk=ch))

    ranked.sort(key=lambda x: x.score, reverse=True)

    # ✅ FIX: use flat config fields, NOT cfg.chunking.*
    top_k = int(getattr(cfg, "prioritize_top_k", 28))
    if top_k <= 0:
        top_k = min(28, len(ranked))

    top_k = min(top_k, len(ranked))
    return [rc.chunk for rc in ranked[:top_k]]