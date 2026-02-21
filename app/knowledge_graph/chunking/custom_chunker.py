import re
from typing import List, Tuple

from app.knowledge_graph.chunking.semantic_chunker import Chunk

PAGE_RE = re.compile(r"\n\s*---\s*Page\s*(\d+)\s*---\s*\n", re.IGNORECASE)

SECTION_PATTERNS = [
    r"^\s*abstract\s*$",
    r"^\s*introduction\s*$",
    r"^\s*background\s*$",
    r"^\s*related work\s*$",
    r"^\s*method(ology)?\s*$",
    r"^\s*experiments?\s*$",
    r"^\s*results?\s*$",
    r"^\s*discussion\s*$",
    r"^\s*conclusion(s)?\s*$",
    r"^\s*references\s*$",
    r"^\s*\d+(\.\d+)*\s+.+$",  # numbered headings
]
SECTION_RE = re.compile("|".join(f"(?:{p})" for p in SECTION_PATTERNS),
                        re.IGNORECASE | re.MULTILINE)

def _split_pages(text: str) -> List[Tuple[int, str]]:
    matches = list(PAGE_RE.finditer(text))
    if not matches:
        return [(1, text.strip())] if text.strip() else []
    out = []
    for i, m in enumerate(matches):
        p = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            out.append((p, body))
    return out

def _split_sections(page_text: str) -> List[Tuple[str, str]]:
    matches = list(SECTION_RE.finditer(page_text))
    if not matches:
        return [("page", page_text.strip())]
    out = []
    for i, m in enumerate(matches):
        title = re.sub(r"\s+", " ", m.group(0).strip())
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(page_text)
        body = page_text[start:end].strip()
        if body:
            out.append((title, body))
    return out or [("page", page_text.strip())]

def _paras(s: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n+", s) if p.strip()]

def _window(paras: List[str], target_words: int, overlap_words: int) -> List[str]:
    chunks = []
    cur, cur_words = [], 0

    def flush():
        nonlocal cur, cur_words
        if not cur:
            return
        chunks.append("\n\n".join(cur).strip())
        if overlap_words <= 0:
            cur, cur_words = [], 0
            return
        kept, kept_words = [], 0
        for p in reversed(cur):
            w = len(p.split())
            if kept_words + w > overlap_words and kept:
                break
            kept.append(p)
            kept_words += w
        kept.reverse()
        cur, cur_words = kept, kept_words

    for p in paras:
        w = len(p.split())
        if w > target_words * 1.5:
            words = p.split()
            step = max(1, target_words - overlap_words)
            for i in range(0, len(words), step):
                chunks.append(" ".join(words[i:i + target_words]))
            continue

        if cur and (cur_words + w > target_words):
            flush()
        cur.append(p)
        cur_words += w

    flush()
    return chunks

def custom_chunk(text_with_page_markers: str,
                 target_words: int = 900,
                 overlap_words: int = 150,
                 drop_references: bool = True) -> List[Chunk]:
    out: List[Chunk] = []
    cid = 1

    for page_num, page_text in _split_pages(text_with_page_markers):
        for sec_title, sec_text in _split_sections(page_text):
            if drop_references and sec_title.strip().lower() == "references":
                continue

            for chunk_text in _window(_paras(sec_text), target_words, overlap_words):
                # Encode metadata into id (no schema change needed)
                chunk_id = f"{cid:05d}|p{page_num}|{sec_title[:30].strip()}"
                out.append(Chunk(chunk_id, chunk_text))
                cid += 1

    return out