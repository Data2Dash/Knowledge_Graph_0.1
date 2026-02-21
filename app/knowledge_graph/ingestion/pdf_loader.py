# app/knowledge_graph/ingestion/pdf_loader.py
from __future__ import annotations

from typing import Optional, Tuple, List, Literal
import re
import unicodedata


# ==========================================================
# Text Cleaning Helpers
# ==========================================================

_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_HEADER_FOOTER_LIKE = re.compile(r"^\s*(page\s*\d+|\d+)\s*$", re.IGNORECASE)


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _clean_text(text: str) -> str:
    if not text:
        return ""

    text = _normalize_unicode(text)
    text = text.replace("\x00", "")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)

    return text.strip()


def _is_noise_line(line: str) -> bool:
    return bool(_HEADER_FOOTER_LIKE.match(line.strip()))


def _remove_page_noise(content: str) -> str:
    lines = content.splitlines()
    cleaned = []
    for line in lines:
        if _is_noise_line(line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ==========================================================
# Marker formatting
# ==========================================================

MarkerFormat = Literal["double_bracket", "square_colon"]


def _page_marker(page_num: int, fmt: MarkerFormat) -> str:
    # Preferred: easy to split reliably
    if fmt == "double_bracket":
        return f"[[PAGE {page_num}]]"
    # Backward-compatible with your original
    return f"[PAGE:{page_num}]"


# ==========================================================
# Main Loader
# ==========================================================

def load_pdf_text(
    pdf_path: str,
    *,
    with_page_markers: bool = True,
    marker_format: MarkerFormat = "double_bracket",
    page_range: Optional[Tuple[int, int]] = None,
    max_total_chars: int = 2_000_000,
) -> str:
    """
    Production-safe PDF loader with:
    - Unicode normalization
    - Hyphen repair
    - Header/footer removal
    - Page filtering (1-based inclusive)
    - Hard size cap
    - Optional page markers (for page-based chunking)
    """

    try:
        from langchain_community.document_loaders import PyPDFLoader
    except Exception as e:
        raise RuntimeError(
            "PyPDFLoader not available. Install dependencies:\n"
            "pip install langchain-community pypdf"
        ) from e

    loader = PyPDFLoader(pdf_path)

    try:
        docs = loader.load()
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {e}") from e

    if not docs:
        return ""

    start_page: Optional[int] = None
    end_page: Optional[int] = None
    if page_range is not None:
        start_page, end_page = page_range
        if start_page is None or end_page is None:
            raise ValueError("page_range must be a (start, end) tuple")
        if start_page < 1 or end_page < 1 or end_page < start_page:
            raise ValueError("page_range must be 1-based inclusive (start>=1, end>=start).")

    parts: List[str] = []
    total_chars = 0

    for i, d in enumerate(docs):
        page_meta = (getattr(d, "metadata", {}) or {}).get("page", i)
        # LangChain usually uses 0-based page in metadata
        try:
            page_num = int(page_meta) + 1
        except Exception:
            page_num = i + 1

        if start_page is not None and (page_num < start_page or page_num > end_page):
            continue

        raw = (d.page_content or "").strip()
        if not raw:
            continue

        cleaned = _clean_text(raw)
        cleaned = _remove_page_noise(cleaned)

        # Skip near-empty pages
        if len(cleaned) < 50:
            continue

        if with_page_markers:
            marker = _page_marker(page_num, marker_format)
            block = f"\n\n{marker}\n{cleaned}"
        else:
            block = cleaned

        total_chars += len(block)
        if total_chars > max_total_chars:
            break

        parts.append(block)

    combined = "\n\n".join(parts).strip()
    return _clean_text(combined)