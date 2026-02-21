# app/knowledge_graph/chunking/semantic_chunker.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from app.core.config import PipelineConfig
from app.knowledge_graph.embeddings.embedder import embed_texts, cosine, Embedding
from app.knowledge_graph.chunking.structure_parser import (
    to_paragraphs,
    to_sections,
    Paragraph,
    Section,
)


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    text: str


def semantic_chunk(text: str, cfg: PipelineConfig) -> List[Chunk]:
    """
    Hybrid chunking for KG extraction (recommended):
    1) Section-aware split (headings) -> paragraphs inside each section
    2) Semantic greedy packing inside each section using embeddings:
       - hard cap: semantic_max_chunk_chars
       - target: semantic_target_chunk_chars
       - split on semantic drift: sim < semantic_sim_threshold
    3) Optional paragraph overlap for relation continuity

    Notes:
    - We apply overlap ONLY ONCE (global) to avoid duplication explosion.
    - If cfg has no `section_aware` flag or it's False, falls back to paragraph-only mode.
    """
    text = (text or "").strip()
    if not text:
        return []

    section_aware = bool(getattr(cfg, "section_aware", True))
    drop_refs = bool(getattr(cfg, "drop_references_section", True))

    if section_aware:
        sections = to_sections(
            text,
            min_chars=int(getattr(cfg, "semantic_min_paragraph_chars", 160)),
            drop_references_section=drop_refs,
        )
        if not sections:
            return []
        chunks = _chunk_sections(sections, cfg)
    else:
        paras = to_paragraphs(
            text,
            min_chars=int(getattr(cfg, "semantic_min_paragraph_chars", 160)),
            drop_references_section=drop_refs,
        )
        if not paras:
            return []
        chunks = _chunk_paragraphs(paras, cfg, id_prefix=None, section_title=None)

    ov = int(getattr(cfg, "semantic_overlap_paragraphs", 0) or 0)
    if ov > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, ov)

    return chunks


# =========================
# Section-aware chunking
# =========================

def _chunk_sections(sections: List[Section], cfg: PipelineConfig) -> List[Chunk]:
    out: List[Chunk] = []
    for si, sec in enumerate(sections, start=1):
        if not sec.paragraphs:
            continue
        id_prefix = f"S{si}"
        out.extend(
            _chunk_paragraphs(
                sec.paragraphs,
                cfg,
                id_prefix=id_prefix,
                section_title=sec.title,
            )
        )
    return out


# =========================
# Embedding utilities
# =========================

def _as_vec(e: Embedding) -> Optional[np.ndarray]:
    v = getattr(e, "values", None)
    if isinstance(v, np.ndarray) and v.size > 0:
        return v.astype(np.float32, copy=False)
    # backward fallback if some other embedding wrapper sneaks in
    if isinstance(v, list) and v:
        return np.asarray(v, dtype=np.float32)
    return None


# =========================
# Core semantic packing
# =========================

def _chunk_paragraphs(
    paras: List[Paragraph],
    cfg: PipelineConfig,
    id_prefix: Optional[str],
    section_title: Optional[str] = None,
) -> List[Chunk]:
    clean_paras: List[Paragraph] = []
    for p in paras:
        t = (p.text or "").strip()
        if t:
            clean_paras.append(Paragraph(text=t))
    if not clean_paras:
        return []

    para_texts = [p.text for p in clean_paras]

    embs = embed_texts(para_texts) or []
    if len(embs) != len(clean_paras):
        return _fallback_size_chunks(clean_paras, cfg, id_prefix=id_prefix, section_title=section_title)

    out: List[Chunk] = []
    buf: List[Paragraph] = []
    buf_len = 0
    last_emb: Optional[np.ndarray] = None

    def _make_id(local_idx: int) -> str:
        return f"{id_prefix}-C{local_idx}" if id_prefix else str(local_idx)

    def flush() -> None:
        nonlocal buf, buf_len, last_emb
        if not buf:
            return

        chunk_body = "\n\n".join(p.text for p in buf).strip()
        chunk_text = f"{section_title}\n{chunk_body}".strip() if section_title else chunk_body

        min_chunk_chars = int(getattr(cfg, "semantic_min_chunk_chars", 200) or 200)
        if chunk_text and len(chunk_text) >= min_chunk_chars:
            out.append(Chunk(id=_make_id(len(out) + 1), text=chunk_text))
        else:
            if _looks_valuable_small_chunk(chunk_text):
                out.append(Chunk(id=_make_id(len(out) + 1), text=chunk_text))

        buf = []
        buf_len = 0
        last_emb = None

    for p, e in zip(clean_paras, embs):
        p_text = p.text
        p_len = len(p_text)

        p_emb = _as_vec(e)

        if not buf:
            buf = [Paragraph(text=p_text)]
            buf_len = p_len
            last_emb = p_emb
            continue

        projected = buf_len + 2 + p_len
        sim = float(cosine(last_emb, p_emb)) if (last_emb is not None and p_emb is not None) else 0.0

        should_split = False
        if projected > int(cfg.semantic_max_chunk_chars):
            should_split = True
        elif projected > int(cfg.semantic_target_chunk_chars) and sim < float(cfg.semantic_sim_threshold):
            should_split = True

        if should_split:
            flush()
            buf = [Paragraph(text=p_text)]
            buf_len = p_len
            last_emb = p_emb
        else:
            buf.append(Paragraph(text=p_text))
            buf_len = projected

            # Update topic embedding cheaply: mean of two vectors
            if last_emb is not None and p_emb is not None and last_emb.shape == p_emb.shape:
                last_emb = (last_emb + p_emb) * 0.5
            elif p_emb is not None:
                last_emb = p_emb

    flush()
    return out


def _looks_valuable_small_chunk(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    s_low = s.lower()
    if s_low.startswith("figure ") or s_low.startswith("table "):
        return True
    signals = [
        "accuracy", "f1", "bleu", "rouge", "map", "auc",
        "precision", "recall", "outperform", "achieve", "%", "latency"
    ]
    return any(tok in s_low for tok in signals)


# =========================
# Overlap utilities (global)
# =========================

def _apply_overlap(chunks: List[Chunk], overlap_paragraphs: int) -> List[Chunk]:
    overlapped: List[Chunk] = []
    prev_tail = ""

    for i, ch in enumerate(chunks):
        if i == 0:
            overlapped.append(ch)
            prev_tail = _tail_paragraphs(_strip_section_header(ch.text), overlap_paragraphs)
            continue

        merged = (prev_tail + "\n\n" + ch.text).strip() if prev_tail else ch.text
        overlapped.append(Chunk(id=ch.id, text=merged))
        prev_tail = _tail_paragraphs(_strip_section_header(ch.text), overlap_paragraphs)

    return overlapped


def _tail_paragraphs(text: str, k: int) -> str:
    ps = [p.strip() for p in (text or "").split("\n\n") if p.strip()]
    if not ps or k <= 0:
        return ""
    return "\n\n".join(ps[-k:])


def _strip_section_header(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    lines = [ln.rstrip() for ln in s.splitlines() if ln.strip()]
    if len(lines) <= 2:
        return s

    first = lines[0].strip()
    second = lines[1].strip()

    if len(first) <= 80 and len(second) >= 40 and not first.endswith((".", "?", "!", ",")):
        if not first.lower().startswith(("we ", "this ", "in ", "our ")):
            return "\n".join(lines[1:]).strip()

    return s


# =========================
# Fallback size-only chunking
# =========================

def _fallback_size_chunks(
    paras: List[Paragraph],
    cfg: PipelineConfig,
    id_prefix: Optional[str],
    section_title: Optional[str] = None,
) -> List[Chunk]:
    out: List[Chunk] = []
    buf: List[str] = []
    buf_len = 0

    def _make_id(local_idx: int) -> str:
        return f"{id_prefix}-C{local_idx}" if id_prefix else str(local_idx)

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return

        chunk_body = "\n\n".join(buf).strip()
        chunk_text = f"{section_title}\n{chunk_body}".strip() if section_title else chunk_body

        min_chunk_chars = int(getattr(cfg, "semantic_min_chunk_chars", 200) or 200)
        if chunk_text and len(chunk_text) >= min_chunk_chars:
            out.append(Chunk(id=_make_id(len(out) + 1), text=chunk_text))
        else:
            if _looks_valuable_small_chunk(chunk_text):
                out.append(Chunk(id=_make_id(len(out) + 1), text=chunk_text))

        buf = []
        buf_len = 0

    for p in paras:
        t = (p.text or "").strip()
        if not t:
            continue
        projected = buf_len + 2 + len(t)
        if projected > int(cfg.semantic_max_chunk_chars) and buf:
            flush()
        buf.append(t)
        buf_len = projected

        if buf_len > int(cfg.semantic_target_chunk_chars) + 800:
            flush()

    flush()
    return out