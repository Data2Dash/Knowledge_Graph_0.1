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
    paras = to_paragraphs(text, min_chars=cfg.semantic_min_paragraph_chars)
    if not paras:
        return []

    embs = embed_texts([p.text for p in paras])
    out: List[Chunk] = []

    buf: List[Paragraph] = []
    buf_len = 0
    last_emb: Optional[List[float]] = None

    def flush():
        nonlocal buf, buf_len, last_emb
        if not buf:
            return
        chunk_text = "\n\n".join(p.text for p in buf).strip()
        if chunk_text:
            out.append(Chunk(id=str(len(out) + 1), text=chunk_text))
        buf = []
        buf_len = 0
        last_emb = None

    for p, e in zip(paras, embs):
        p_len = len(p.text)
        if not buf:
            buf = [p]
            buf_len = p_len
            last_emb = e.values
            continue

        sim = cosine(last_emb, e.values) if last_emb is not None else 0.0
        projected = buf_len + 2 + p_len

        should_split = False
        if projected > cfg.semantic_max_chunk_chars:
            should_split = True
        elif projected > cfg.semantic_target_chunk_chars and sim < cfg.semantic_sim_threshold:
            should_split = True

        if should_split:
            flush()
            buf = [p]
            buf_len = p_len
            last_emb = e.values
        else:
            buf.append(p)
            buf_len = projected
            # update last embedding to “topic drift aware” average (cheap)
            last_emb = [(a + b) / 2 for a, b in zip(last_emb, e.values)]

    flush()

    # Semantic overlap by paragraphs (keeps relations continuity)
    if cfg.semantic_overlap_paragraphs > 0 and len(out) > 1:
        overlapped: List[Chunk] = []
        prev_tail = ""
        for i, ch in enumerate(out):
            if i == 0:
                overlapped.append(ch)
                prev_tail = _tail_paragraphs(ch.text, cfg.semantic_overlap_paragraphs)
                continue
            merged = (prev_tail + "\n\n" + ch.text).strip() if prev_tail else ch.text
            overlapped.append(Chunk(id=ch.id, text=merged))
            prev_tail = _tail_paragraphs(ch.text, cfg.semantic_overlap_paragraphs)
        out = overlapped

    return out

def _tail_paragraphs(text: str, k: int) -> str:
    ps = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not ps:
        return ""
    return "\n\n".join(ps[-k:])
