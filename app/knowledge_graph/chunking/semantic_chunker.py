from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from app.core.config import PipelineConfig
from app.knowledge_graph.embeddings.embedder import embed_texts, cosine
from app.knowledge_graph.chunking.structure_parser import to_paragraphs, Paragraph


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str


def semantic_chunk(text: str, cfg: PipelineConfig) -> List[Chunk]:
    """
    Semantic chunking:
    - Split into paragraphs (structure_parser)
    - Embed each paragraph
    - Greedily pack paragraphs into chunks using:
        * hard max size
        * target size + semantic drift threshold
    - Optional paragraph overlap between chunks for relation continuity
    """
    text = (text or "").strip()
    if not text:
        return []

    paras = to_paragraphs(text, min_chars=cfg.semantic_min_paragraph_chars)
    if not paras:
        return []

    # Embeddings (must align with paragraphs)
    embs = embed_texts([p.text for p in paras]) or []
    if len(embs) != len(paras):
        # Fail safe: if embedding step breaks alignment, fall back to size-based chunking
        return _fallback_size_chunks(paras, cfg)

    out: List[Chunk] = []

    buf: List[Paragraph] = []
    buf_len = 0
    last_emb: Optional[List[float]] = None

    def flush():
        nonlocal buf, buf_len, last_emb
        if not buf:
            return
        chunk_text = "\n\n".join(p.text for p in buf).strip()

        # Filter tiny chunks (avoid junk)
        if chunk_text and len(chunk_text) >= 200:
            out.append(Chunk(id=str(len(out) + 1), text=chunk_text))

        buf = []
        buf_len = 0
        last_emb = None

    for p, e in zip(paras, embs):
        p_text = (p.text or "").strip()
        if not p_text:
            continue

        p_len = len(p_text)
        p_emb = getattr(e, "values", None)

        # If embedding missing, treat as unrelated paragraph
        if not isinstance(p_emb, list) or not p_emb:
            p_emb = None

        if not buf:
            buf = [Paragraph(text=p_text)]
            buf_len = p_len
            last_emb = p_emb
            continue

        projected = buf_len + 2 + p_len  # +2 for "\n\n"
        sim = cosine(last_emb, p_emb) if (last_emb is not None and p_emb is not None) else 0.0

        should_split = False
        if projected > cfg.semantic_max_chunk_chars:
            should_split = True
        elif projected > cfg.semantic_target_chunk_chars and sim < cfg.semantic_sim_threshold:
            should_split = True

        if should_split:
            flush()
            buf = [Paragraph(text=p_text)]
            buf_len = p_len
            last_emb = p_emb
        else:
            buf.append(Paragraph(text=p_text))
            buf_len = projected

            # Update "topic" embedding cheaply: average of previous and current
            if last_emb is not None and p_emb is not None and len(last_emb) == len(p_emb):
                last_emb = [(a + b) / 2 for a, b in zip(last_emb, p_emb)]
            else:
                last_emb = p_emb if p_emb is not None else last_emb

    flush()

    # Paragraph overlap (keeps relation continuity across chunk boundaries)
    if cfg.semantic_overlap_paragraphs > 0 and len(out) > 1:
        out = _apply_overlap(out, cfg.semantic_overlap_paragraphs)

    return out


def _apply_overlap(chunks: List[Chunk], overlap_paragraphs: int) -> List[Chunk]:
    overlapped: List[Chunk] = []
    prev_tail = ""

    for i, ch in enumerate(chunks):
        if i == 0:
            overlapped.append(ch)
            prev_tail = _tail_paragraphs(ch.text, overlap_paragraphs)
            continue

        merged = (prev_tail + "\n\n" + ch.text).strip() if prev_tail else ch.text
        overlapped.append(Chunk(id=ch.id, text=merged))
        prev_tail = _tail_paragraphs(ch.text, overlap_paragraphs)

    return overlapped


def _tail_paragraphs(text: str, k: int) -> str:
    ps = [p.strip() for p in (text or "").split("\n\n") if p.strip()]
    if not ps or k <= 0:
        return ""
    return "\n\n".join(ps[-k:])


def _fallback_size_chunks(paras: List[Paragraph], cfg: PipelineConfig) -> List[Chunk]:
    """
    Safety fallback if embeddings fail/mismatch.
    Packs paragraphs by size only.
    """
    out: List[Chunk] = []
    buf: List[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if not buf:
            return
        chunk_text = "\n\n".join(buf).strip()
        if chunk_text and len(chunk_text) >= 200:
            out.append(Chunk(id=str(len(out) + 1), text=chunk_text))
        buf = []
        buf_len = 0

    for p in paras:
        t = (p.text or "").strip()
        if not t:
            continue
        projected = buf_len + 2 + len(t)
        if projected > cfg.semantic_max_chunk_chars and buf:
            flush()
        buf.append(t)
        buf_len = buf_len + 2 + len(t)

        # optional: if we exceed target by a lot, flush
        if buf_len > cfg.semantic_target_chunk_chars + 800:
            flush()

    flush()
    return out
