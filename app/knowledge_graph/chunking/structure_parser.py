# app/knowledge_graph/chunking/structure_parser.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import re
from collections import Counter

from app.core.logging import setup_logging

LOGGER = setup_logging("knowledge_graph.structure_parser")


@dataclass(frozen=True, slots=True)
class Paragraph:
    text: str


@dataclass(frozen=True, slots=True)
class Section:
    title: str
    level: int
    paragraphs: List[Paragraph]


# Split on one-or-more blank lines (robust to whitespace)
_SPLIT_RE = re.compile(r"\n\s*\n+")

# Whitespace normalization
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# Noise patterns
_DECOR_LINE_RE = re.compile(r"^[-=_]{3,}$")
_PAGE_NUM_RE = re.compile(r"^\s*\d+\s*$")

# Caption detection (KEEP, not noise)
_CAPTION_RE = re.compile(r"^\s*(figure|fig\.|table)\s*(\d+)\s*[:.\-]?\s*(.*)$", re.IGNORECASE)

# Hyphenation join at line breaks: "atten-\n tion" -> "attention"
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")

# Bullet/list detection
_BULLET_RE = re.compile(r"^(\-|\*|•|▪|■|►|◆|▶|●|◦|\d+\.)\s+")

# Numbered headings like:
# "1. Introduction" / "2.3 Methods" / "1 Introduction"
_NUM_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+(.+)$")

# Roman headings:
# "IV. Experiments" / "II Related Work"
_ROMAN_HEADING_RE = re.compile(r"^\s*([IVX]+)\.?\s+(.+)$", re.IGNORECASE)

# Heading heuristics
_ALL_CAPS_RE = re.compile(r"^[A-Z0-9 \-–—:]{6,}$")
_ENDS_COLON_RE = re.compile(r".+:\s*$")

# References section markers
_REFS_START_RE = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE)


def _only_numbers(s: str) -> bool:
    ss = (s or "").strip()
    if not ss:
        return True
    return bool(re.fullmatch(r"[\d\.\-\–—]+", ss))


def _clean_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def _clean_block(block: str) -> str:
    block = _MULTI_SPACE_RE.sub(" ", (block or ""))
    return block.strip()


def _looks_like_noise_single_line(line: str) -> bool:
    """
    Single-line patterns that are very likely NOT useful content.
    IMPORTANT: captions are NOT noise.
    """
    s = (line or "").strip()
    if not s:
        return True
    if _DECOR_LINE_RE.match(s):
        return True
    if _PAGE_NUM_RE.match(s):
        return True
    if _CAPTION_RE.match(s):
        return False
    if len(s) <= 2 and all(ch in ".-–—_•" for ch in s):
        return True
    return False


def _is_heading_line(line: str) -> bool:
    """
    Conservative heading detection.
    Avoid treating captions as headings.
    """
    s = (line or "").strip()
    if not s:
        return False
    if len(s) > 120:
        return False
    if _CAPTION_RE.match(s):
        return False

    m = _NUM_HEADING_RE.match(s)
    if m:
        title = (m.group(2) or "").strip()
        if title and not _only_numbers(title):
            return True

    m2 = _ROMAN_HEADING_RE.match(s)
    if m2:
        title = (m2.group(2) or "").strip()
        if title and not _only_numbers(title):
            return True

    if _ENDS_COLON_RE.match(s) and len(s) <= 80:
        return True

    # ALL CAPS headings (common in PDFs). Allow single-word headings too.
    if _ALL_CAPS_RE.match(s):
        if not s.endswith((".", "?", "!")):
            return True

    # Title-case-ish short line (few words) can be heading
    words = s.split()
    if 2 <= len(words) <= 8 and sum(w[:1].isupper() for w in words if w) >= max(2, len(words) // 2):
        if not s.endswith((".", "?", "!")):
            return True

    return False


def _join_broken_lines(block: str) -> str:
    """
    Join broken PDF lines into a paragraph unless it looks like a list/table-ish text.
    """
    lines = [ln.strip() for ln in (block or "").split("\n") if ln.strip()]
    if not lines:
        return ""

    if _CAPTION_RE.match(lines[0]):
        return "\n".join(lines)

    bullet_like = sum(1 for ln in lines if _BULLET_RE.match(ln)) >= 2
    if bullet_like:
        return "\n".join(lines)

    short_lines = sum(1 for ln in lines if len(ln) <= 30)
    if len(lines) >= 5 and short_lines >= (len(lines) * 0.6):
        return "\n".join(lines)

    return " ".join(lines)


def _strip_repeating_headers_footers(blocks: List[str], min_repeats: int = 3) -> List[str]:
    candidates: List[str] = []
    for b in blocks:
        lines = [ln.strip() for ln in b.split("\n") if ln.strip()]
        if not lines:
            continue
        if len(lines[0]) <= 80:
            candidates.append(lines[0])
        if len(lines[-1]) <= 80:
            candidates.append(lines[-1])

    freq = Counter(candidates)
    repeated = {k for k, v in freq.items() if v >= min_repeats}
    if not repeated:
        return blocks

    cleaned: List[str] = []
    for b in blocks:
        lines = [ln for ln in b.split("\n")]
        if lines and lines[0].strip() in repeated and not _REFS_START_RE.match(lines[0].strip() or ""):
            if not _CAPTION_RE.match(lines[0].strip() or ""):
                lines = lines[1:]
        if lines and lines[-1].strip() in repeated and not _REFS_START_RE.match(lines[-1].strip() or ""):
            if not _CAPTION_RE.match(lines[-1].strip() or ""):
                lines = lines[:-1]
        nb = "\n".join(lines).strip()
        if nb:
            cleaned.append(nb)
    return cleaned


def to_paragraphs(text: str, min_chars: int = 160, drop_references_section: bool = False) -> List[Paragraph]:
    t = _clean_text(text)
    if not t:
        return []

    raw_blocks = [b for b in _SPLIT_RE.split(t) if b and b.strip()]

    blocks: List[str] = []
    for b in raw_blocks:
        b = _clean_block(b)
        if not b:
            continue
        if "\n" not in b and _looks_like_noise_single_line(b):
            continue
        blocks.append(b)

    if not blocks:
        return [Paragraph(text=t)]

    blocks = _strip_repeating_headers_footers(blocks, min_repeats=3)

    if drop_references_section:
        trimmed: List[str] = []
        for b in blocks:
            first_line = (b.split("\n", 1)[0] if b else "").strip()
            if _REFS_START_RE.match(first_line):
                break
            trimmed.append(b)
        if trimmed:
            blocks = trimmed

    joined_blocks: List[str] = []
    for b in blocks:
        jb = _join_broken_lines(b)
        if jb:
            joined_blocks.append(jb)

    if not joined_blocks:
        return [Paragraph(text=t)]

    merged: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        merged_text = "\n\n".join(buf).strip()
        if merged_text:
            merged.append(merged_text)
        buf = []
        buf_len = 0

    pending_heading: str = ""

    for b in joined_blocks:
        b = b.strip()
        if not b:
            continue

        if "\n" not in b and _is_heading_line(b):
            flush()
            pending_heading = b
            continue

        if pending_heading:
            b = f"{pending_heading}\n{b}"
            pending_heading = ""

        if len(b) >= min_chars:
            if buf:
                buf.append(b)
                flush()
            else:
                merged.append(b)
        else:
            buf.append(b)
            buf_len += len(b) + 2
            if buf_len >= min_chars:
                flush()

    if pending_heading:
        if buf:
            buf.append(pending_heading)
            flush()
        else:
            merged.append(pending_heading)

    flush()

    paras = [Paragraph(text=p) for p in merged if p and p.strip()]
    return paras or [Paragraph(text=t)]


def to_sections(text: str, min_chars: int = 120, drop_references_section: bool = True) -> List[Section]:
    t = _clean_text(text)
    if not t:
        return []

    raw_blocks = [b for b in _SPLIT_RE.split(t) if b and b.strip()]

    blocks: List[str] = []
    for b in raw_blocks:
        b = _clean_block(b)
        if not b:
            continue
        if "\n" not in b and _looks_like_noise_single_line(b):
            continue
        blocks.append(b)

    if not blocks:
        return [Section(title="Document", level=0, paragraphs=[Paragraph(text=t)])]

    blocks = _strip_repeating_headers_footers(blocks, min_repeats=3)

    joined_blocks: List[str] = []
    for b in blocks:
        jb = _join_broken_lines(b)
        if jb:
            joined_blocks.append(jb)

    sections: List[Section] = []
    cur_title = "Document"
    cur_level = 0
    cur_paras: List[Paragraph] = []

    def flush_section() -> None:
        nonlocal cur_paras
        if cur_paras:
            sections.append(Section(title=cur_title, level=cur_level, paragraphs=cur_paras))
            cur_paras = []

    def push_paragraph(p: str) -> None:
        p = (p or "").strip()
        if not p:
            return
        if cur_paras and len(p) < min_chars:
            prev = cur_paras[-1].text
            cur_paras[-1] = Paragraph(text=(prev + "\n\n" + p).strip())
        else:
            cur_paras.append(Paragraph(text=p))

    def heading_level(h: str) -> int:
        h = (h or "").strip()
        m = _NUM_HEADING_RE.match(h)
        if m:
            num = (m.group(1) or "").strip()
            return max(1, num.count(".") + 1)
        if _ROMAN_HEADING_RE.match(h):
            return 1
        return 1

    for b in joined_blocks:
        s = b.strip()
        if not s:
            continue

        first_line = (s.split("\n", 1)[0] if s else "").strip()

        if drop_references_section and _REFS_START_RE.match(first_line):
            break

        if "\n" not in s and _is_heading_line(s):
            flush_section()
            cur_title = s
            cur_level = heading_level(s)
            continue

        cap = _CAPTION_RE.match(first_line)
        if cap:
            kind_raw = cap.group(1).lower()
            kind = "figure" if kind_raw.startswith("fig") else "table"
            num = cap.group(2)
            rest = (cap.group(3) or "").strip()

            caption_text = f"{kind.title()} {num}"
            if rest:
                caption_text = f"{caption_text}: {rest}"

            push_paragraph(caption_text)

            if "\n" in s:
                tail = s.split("\n", 1)[1].strip()
                if tail:
                    push_paragraph(tail)
            continue

        push_paragraph(s)

    flush_section()
    return sections or [Section(title="Document", level=0, paragraphs=[Paragraph(text=t)])]