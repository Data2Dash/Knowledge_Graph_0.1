from __future__ import annotations

from dataclasses import dataclass
from typing import List
import re


@dataclass(frozen=True)
class Paragraph:
    text: str


# --- helpers ---
_SPLIT_RE = re.compile(r"\n\s*\n")            # split on blank lines (flexible)
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_DECOR_LINE_RE = re.compile(r"^[-=_]{3,}$")   # decorative lines like ---- or ====
_PAGE_NUM_RE = re.compile(r"^\s*\d+\s*$")     # lines that are only a number
_FIG_TABLE_RE = re.compile(r"^(figure|fig\.|table)\s*\d+[:.\s]*$", re.IGNORECASE)


def _clean_block(block: str) -> str:
    # normalize whitespace but keep line breaks for now
    block = block.replace("\r\n", "\n").replace("\r", "\n")
    # remove repeated spaces
    block = _MULTI_SPACE_RE.sub(" ", block)
    return block.strip()


def _join_broken_lines(block: str) -> str:
    """
    PDFs often break sentences into multiple lines.
    We join lines inside a block when it looks like a paragraph.
    """
    lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
    if not lines:
        return ""

    # if it's a list, keep line breaks (better readability)
    bullet_like = sum(1 for ln in lines if re.match(r"^(\-|\*|•|\d+\.)\s+", ln)) >= 2
    if bullet_like:
        return "\n".join(lines)

    # otherwise, join into a single paragraph
    return " ".join(lines)


def to_paragraphs(text: str, min_chars: int = 160) -> List[Paragraph]:
    """
    Improved paragraph parsing:
    - split on blank lines robustly
    - clean blocks
    - join broken PDF lines within blocks
    - merge small blocks instead of dropping them
    - light noise filtering
    """
    text = (text or "").strip()
    if not text:
        return []

    # 1) Split into rough blocks
    raw_blocks = [b for b in _SPLIT_RE.split(text) if b and b.strip()]

    # 2) Clean + join broken lines
    blocks: List[str] = []
    for b in raw_blocks:
        b = _clean_block(b)
        if not b:
            continue

        # filter obvious noise blocks
        if _DECOR_LINE_RE.match(b):
            continue

        # If block is single-line noise (page number / figure tag), skip
        if "\n" not in b and (_PAGE_NUM_RE.match(b) or _FIG_TABLE_RE.match(b)):
            continue

        b = _join_broken_lines(b)
        if b:
            blocks.append(b)

    if not blocks:
        # absolute fallback
        return [Paragraph(text=text)]

    # 3) Merge small blocks (instead of dropping)
    merged: List[str] = []
    buf = ""

    def flush_buf():
        nonlocal buf
        if buf.strip():
            merged.append(buf.strip())
        buf = ""

    for b in blocks:
        if len(b) >= min_chars:
            # if we have buffered small text, attach it before the big paragraph
            if buf:
                b = (buf.strip() + "\n\n" + b).strip()
                buf = ""
            merged.append(b)
        else:
            # accumulate small blocks
            if not buf:
                buf = b
            else:
                # keep a separator so it doesn't become unreadable
                buf = (buf.strip() + "\n\n" + b).strip()

            # if buffer becomes reasonably sized, flush it
            if len(buf) >= min_chars:
                flush_buf()

    flush_buf()

    # 4) Final wrap as Paragraph objects
    paras = [Paragraph(text=m) for m in merged if m and m.strip()]

    # 5) Safety fallback
    if not paras:
        return [Paragraph(text=text)]

    return paras
