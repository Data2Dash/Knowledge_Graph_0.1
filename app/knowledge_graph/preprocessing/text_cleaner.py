from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal
import re
import unicodedata

from app.core.config import PipelineConfig
from app.core.logging import setup_logging
from app.knowledge_graph.chunking.structure_parser import to_sections
from app.knowledge_graph.chunking.semantic_chunker import semantic_chunk, Chunk

LOGGER = setup_logging("knowledge_graph.text_cleaner")

ChunkStrategy = Literal["semantic", "sections", "sliding", "pages"]


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    unicode_normalize: bool = True
    strip_null_bytes: bool = True
    normalize_newlines: bool = True
    fix_hyphenated_linebreaks: bool = True

    remove_numeric_citations: bool = True         # [1], [2,3]
    remove_year_only_parens: bool = True          # (2020)
    remove_inline_latex: bool = False             # $...$
    remove_display_latex: bool = False            # $$...$$

    # Used for sliding/pages/oversized sections
    max_chunk_soft_chars: int = 2800
    max_chunk_hard_chars: int = 3500
    overlap_chars: int = 400

    # Safety caps
    min_chunk_chars: int = 300
    max_chunks: int = 120


def _derive_preprocess_cfg(
    preprocess_cfg: Optional[PreprocessConfig],
    pipeline_cfg: Optional[PipelineConfig],
) -> PreprocessConfig:
    """
    If preprocess_cfg isn't provided, derive reasonable defaults from PipelineConfig
    so chunk sizes stay consistent across the project.
    """
    if preprocess_cfg is not None:
        return preprocess_cfg

    if pipeline_cfg is None:
        return PreprocessConfig()

    soft = int(getattr(pipeline_cfg, "semantic_target_chunk_chars", 2800))
    hard = int(getattr(pipeline_cfg, "semantic_max_chunk_chars", 3500))
    hard = max(hard, soft)

    # overlap heuristic: ~15% of soft cap, bounded
    overlap = max(200, min(800, soft // 6))

    return PreprocessConfig(
        max_chunk_soft_chars=soft,
        max_chunk_hard_chars=hard,
        overlap_chars=overlap,
    )


# -----------------------------
# Cleaning
# -----------------------------

def clean_text(text: str, cfg: Optional[PreprocessConfig] = None) -> str:
    cfg = cfg or PreprocessConfig()
    if not text:
        return ""

    if cfg.strip_null_bytes:
        text = text.replace("\x00", "")

    if cfg.normalize_newlines:
        text = text.replace("\r\n", "\n").replace("\r", "\n")

    if cfg.unicode_normalize:
        text = unicodedata.normalize("NFKC", text)

    if cfg.fix_hyphenated_linebreaks:
        # join "hyphen-\nated" -> "hyphenated"
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    if cfg.remove_numeric_citations:
        text = re.sub(r"\[(\d+(?:,\s*\d+)*)\]", "", text)

    if cfg.remove_year_only_parens:
        text = re.sub(r"\(\d{4}\)", "", text)

    if cfg.remove_display_latex:
        text = re.sub(r"\$\$(.*?)\$\$", " ", text, flags=re.DOTALL)

    if cfg.remove_inline_latex:
        text = re.sub(r"\$(.*?)\$", " ", text)

    # whitespace normalization
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# -----------------------------
# Sliding window (char-aware)
# -----------------------------

def sliding_window_chunks(text: str, cfg: PreprocessConfig) -> List[str]:
    if not text:
        return []

    soft = max(200, int(cfg.max_chunk_soft_chars))
    hard = max(soft, int(cfg.max_chunk_hard_chars))
    overlap = max(0, int(cfg.overlap_chars))

    step = max(1, soft - overlap)  # prevents infinite loop

    chunks: List[str] = []
    start = 0

    while start < len(text) and len(chunks) < int(cfg.max_chunks):
        end = min(start + hard, len(text))
        chunk = text[start:end].strip()

        if len(chunk) >= int(cfg.min_chunk_chars):
            chunks.append(chunk)

        start += step

    return chunks


# -----------------------------
# Section-based
# -----------------------------

def sections_as_chunks(text: str, cfg: PreprocessConfig) -> List[str]:
    sections = to_sections(text)
    chunks: List[str] = []

    for sec in sections:
        body = "\n".join(p.text for p in sec.paragraphs).strip()
        if not body:
            continue

        full = f"{sec.title}\n{body}".strip() if sec.title else body
        if len(full) < int(cfg.min_chunk_chars):
            continue

        # If section too large, split it
        if len(full) > int(cfg.max_chunk_hard_chars):
            chunks.extend(sliding_window_chunks(full, cfg))
        else:
            chunks.append(full)

        if len(chunks) >= int(cfg.max_chunks):
            break

    return chunks[: int(cfg.max_chunks)]


# -----------------------------
# Page-based
# -----------------------------

_PAGE_SPLIT_PATTERNS = [
    # [[PAGE 3]]
    re.compile(r"(?:^|\n)\s*\[\[\s*PAGE\s*\d+\s*\]\]\s*(?:\n|$)", re.IGNORECASE),
    # [PAGE:3]  (backward-compatible marker)
    re.compile(r"(?:^|\n)\s*\[\s*PAGE\s*:\s*\d+\s*\]\s*(?:\n|$)", re.IGNORECASE),

    # === Page 3 === / ---- Page 3 ----
    re.compile(r"(?:^|\n)\s*(?:=+|-+)\s*page\s*\d+\s*(?:=+|-+)\s*(?:\n|$)", re.IGNORECASE),
    # Page 3:
    re.compile(r"(?:^|\n)\s*page\s*\d+\s*:\s*(?:\n|$)", re.IGNORECASE),
]

def _split_pages(text: str) -> List[str]:
    """
    Best-effort split based on common page markers.
    If none found, returns [text].
    """
    for pat in _PAGE_SPLIT_PATTERNS:
        parts = pat.split(text)
        # if split actually happened
        if len(parts) > 1:
            parts = [p.strip() for p in parts if p.strip()]
            return parts if parts else [text.strip()]
    return [text.strip()] if text.strip() else []


def pages_as_chunks(text: str, cfg: PreprocessConfig) -> List[str]:
    pages = _split_pages(text)
    out: List[str] = []

    for p in pages:
        if not p:
            continue
        if len(p) < int(cfg.min_chunk_chars):
            continue

        if len(p) > int(cfg.max_chunk_hard_chars):
            out.extend(sliding_window_chunks(p, cfg))
        else:
            out.append(p)

        if len(out) >= int(cfg.max_chunks):
            break

    return out[: int(cfg.max_chunks)]


# -----------------------------
# Semantic (delegation)
# -----------------------------

def _semantic_chunks_as_list(text: str, pipeline_cfg: PipelineConfig) -> List[Chunk]:
    kg_chunks = semantic_chunk(text, pipeline_cfg)
    return [ch for ch in kg_chunks if (ch.text or "").strip()]


# -----------------------------
# Public API
# -----------------------------

def make_chunks(
    text: str,
    strategy: ChunkStrategy,
    *,
    pipeline_cfg: Optional[PipelineConfig] = None,
    preprocess_cfg: Optional[PreprocessConfig] = None,
) -> List[Chunk]:
    """
    Return list of Chunk(id, text) for pipeline and rank_chunks.
    """
    preprocess_cfg = _derive_preprocess_cfg(preprocess_cfg, pipeline_cfg)
    cleaned = clean_text(text, preprocess_cfg)

    if not cleaned:
        return []

    if strategy == "semantic":
        if not pipeline_cfg:
            raise ValueError("pipeline_cfg required for semantic chunking")
        return _semantic_chunks_as_list(cleaned, pipeline_cfg)

    if strategy == "sections":
        raw = sections_as_chunks(cleaned, preprocess_cfg)
        return [Chunk(id=f"sec:{i+1}", text=s) for i, s in enumerate(raw)]

    if strategy == "sliding":
        raw = sliding_window_chunks(cleaned, preprocess_cfg)
        return [Chunk(id=f"win:{i+1}", text=s) for i, s in enumerate(raw)]

    if strategy == "pages":
        raw = pages_as_chunks(cleaned, preprocess_cfg)
        return [Chunk(id=f"page:{i+1}", text=s) for i, s in enumerate(raw)]

    raise ValueError(f"Unknown chunk strategy: {strategy}")