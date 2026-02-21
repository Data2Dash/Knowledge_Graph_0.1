from __future__ import annotations

# Reuse your exact implementation (copied) but expose:
# - PreprocessConfig
# - preprocess_text(text, cfg)
# - make_chunks(text, strategy, cfg)
# For brevity: import from your existing module if you keep it.

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import os, re, unicodedata, logging

LOGGER = logging.getLogger(__name__)

DEFAULT_ENTITY_MAP: Dict[str, str] = {
    "llms": "Large Language Models",
    "llm": "Large Language Model",
    "kg": "Knowledge Graph",
    "kgs": "Knowledge Graphs",
    "nlp": "Natural Language Processing",
    "gnn": "Graph Neural Network",
    "transformer": "Transformer Architecture",
    "cnn": "Convolutional Neural Network",
    "rnn": "Recurrent Neural Network",
}

DEFAULT_STOP_HEADINGS = (
    "references","bibliography","acknowledgment","acknowledgements","appendix",
    "supplementary material","author contributions","ethics statement","data availability",
    "conflict of interest","funding",
)

DEFAULT_SECTION_HEADINGS = (
    "Abstract","Introduction","Background","Related Work","Preliminaries","Methodology","Methods",
    "Approach","Model Architecture","Model","Architecture","Implementation","Training","Datasets",
    "Benchmarks","Experimental Setup","Setup","Experiments","Results","Evaluation","Analysis",
    "Ablation","Discussion","Limitations","Future Work","Conclusion","Contributions","Key Findings",
    "Findings","Proposed Method",
)

@dataclass(frozen=True)
class PreprocessConfig:
    unicode_normalize: bool = True
    strip_null_bytes: bool = True
    normalize_newlines: bool = True
    keep_double_newlines: bool = True

    remove_numeric_citations: bool = True
    remove_year_only_parens: bool = True
    remove_bullets: bool = True
    remove_decorative_lines: bool = True
    fix_hyphenated_linebreaks: bool = True
    remove_inline_latex: bool = False
    remove_display_latex: bool = False

    stop_headings: Tuple[str, ...] = DEFAULT_STOP_HEADINGS
    entity_map: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ENTITY_MAP))

    max_chunk_size: int = 3500
    overlap: int = 450

    sliding_window_size: int = 2800
    sliding_step: int = 900
    sliding_max_chunks: int = 30

    min_page_chars: int = 150
    section_headings: Tuple[str, ...] = DEFAULT_SECTION_HEADINGS


def _compile_patterns(cfg: PreprocessConfig):
    numeric_citations = re.compile(
        r"\[(?:\s*\d+\s*(?:[-–]\s*\d+\s*)?)(?:\s*,\s*\d+\s*(?:[-–]\s*\d+\s*)?)*\]"
    )
    year_only_parens = re.compile(r"\(\s*\d{4}\s*\)")
    bullets = re.compile(r"[•▪■►◆▶●◦]")
    decorative_lines = re.compile(r"(?:-{3,}|={3,}|_{3,}|\*{3,})")
    spaces_tabs = re.compile(r"[ \t]+")
    many_newlines = re.compile(r"\n{3,}")
    hyphen_linebreak = re.compile(r"(\w)-\n(\w)")
    inline_latex = re.compile(r"\$(?!\s)(.*?)(?<!\s)\$", flags=re.DOTALL)
    display_latex_1 = re.compile(r"\\\[(.*?)\\\]", flags=re.DOTALL)
    display_latex_2 = re.compile(r"\$\$(.*?)\$\$", flags=re.DOTALL)

    stop_heading_union = "|".join(re.escape(h) for h in cfg.stop_headings)
    stop_heading = re.compile(rf"^(\d+\.?\s*)?({stop_heading_union})$", flags=re.IGNORECASE)

    section_union = "|".join(cfg.section_headings).replace(" ", r"\s+")
    section_regex = re.compile(
        rf"(?m)^(?:\d+\.|[IVX]+\.|[IVX]+\b|\d+\b)?\s*({section_union})\s*$",
        flags=re.IGNORECASE,
    )

    return dict(
        numeric_citations=numeric_citations,
        year_only_parens=year_only_parens,
        bullets=bullets,
        decorative_lines=decorative_lines,
        spaces_tabs=spaces_tabs,
        many_newlines=many_newlines,
        hyphen_linebreak=hyphen_linebreak,
        inline_latex=inline_latex,
        display_latex_1=display_latex_1,
        display_latex_2=display_latex_2,
        stop_heading=stop_heading,
        section_regex=section_regex,
    )


def clean_text(text: str, cfg: Optional[PreprocessConfig] = None) -> str:
    cfg = cfg or PreprocessConfig()
    pat = _compile_patterns(cfg)
    if text is None:
        return ""
    if cfg.strip_null_bytes:
        text = text.replace("\x00", "")
    if cfg.normalize_newlines:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    if cfg.unicode_normalize:
        text = unicodedata.normalize("NFKC", text)
    if cfg.fix_hyphenated_linebreaks:
        text = pat["hyphen_linebreak"].sub(r"\1\2", text)

    if cfg.remove_numeric_citations:
        text = pat["numeric_citations"].sub("", text)
    if cfg.remove_year_only_parens:
        text = pat["year_only_parens"].sub("", text)
    if cfg.remove_bullets:
        text = pat["bullets"].sub("", text)
    if cfg.remove_decorative_lines:
        text = pat["decorative_lines"].sub(" ", text)

    if cfg.remove_display_latex:
        text = pat["display_latex_1"].sub(" ", text)
        text = pat["display_latex_2"].sub(" ", text)
    if cfg.remove_inline_latex:
        text = pat["inline_latex"].sub(" ", text)

    text = pat["spaces_tabs"].sub(" ", text)
    if cfg.keep_double_newlines:
        text = pat["many_newlines"].sub("\n\n", text)
    return text.strip()


def remove_irrelevant_sections(text: str, cfg: Optional[PreprocessConfig] = None) -> str:
    cfg = cfg or PreprocessConfig()
    pat = _compile_patterns(cfg)
    if not text.strip():
        return ""
    out = []
    for line in text.split("\n"):
        cl = line.strip().lower()
        if 0 < len(cl) < 60 and pat["stop_heading"].match(cl):
            break
        out.append(line)
    return "\n".join(out).strip()


def normalize_entities(text: str, cfg: Optional[PreprocessConfig] = None) -> str:
    cfg = cfg or PreprocessConfig()
    if not text.strip():
        return ""
    items = sorted(cfg.entity_map.items(), key=lambda kv: len(kv[0]), reverse=True)
    for short, full in items:
        text = re.sub(rf"\b{re.escape(short)}\b", full, text, flags=re.IGNORECASE)
    return text


def preprocess_text(text: str, cfg: Optional[PreprocessConfig] = None) -> str:
    cfg = cfg or PreprocessConfig()
    t = clean_text(text, cfg=cfg)
    t = remove_irrelevant_sections(t, cfg=cfg)
    t = normalize_entities(t, cfg=cfg)
    return t


# Keep your existing chunk strategies (sections/sliding/pages) for fallback.
def split_by_sections(text: str, max_chunk_size: int = 3500, overlap: int = 450, cfg: Optional[PreprocessConfig] = None):
    cfg = cfg or PreprocessConfig()
    pat = _compile_patterns(cfg)
    t = text.strip()
    if not t:
        return []
    parts = re.split(pat["section_regex"], t)
    final_chunks = []
    current = []
    heading_prefix = re.compile(r"^(abstract|introduction|background|related|prelim|method|approach|model|architect|implement|train|dataset|benchmark|experimental|setup|experiment|result|evaluat|analysis|ablation|discuss|limit|future|conclus|contribution|finding|proposed)", flags=re.I)
    for part in parts:
        if not part:
            continue
        part = part.strip()
        if len(part) < 60 and heading_prefix.match(part):
            current.append(f"\n\n## {part}\n\n")
            continue
        current.append(part)
        full = " ".join(current).strip()
        if len(full) >= max_chunk_size:
            final_chunks.append(full)
            tail = full[-overlap:].strip() if overlap > 0 else ""
            current = [tail] if tail else []
    if current:
        final_chunks.append(" ".join(current).strip())
    return [c for c in final_chunks if len(c.strip()) >= 200]


def sliding_window_chunks(text: str, window_size: int = 2800, step: int = 900, max_chunks: int = 30):
    t = text.strip()
    if not t:
        return []
    if window_size <= 0 or step <= 0:
        return [t]
    chunks = []
    start = 0
    while start < len(t) and len(chunks) < max_chunks:
        c = t[start:start+window_size].strip()
        if len(c) >= 200:
            chunks.append(c)
        start += step
    return chunks


def page_based_chunks(text: str, min_page_chars: int = 150):
    t = text.strip()
    if not t:
        return []
    pages = re.split(r"\n*--- Page \d+ ---\n*", t)
    pages = [p.strip() for p in pages if p.strip()]
    out = []
    buf = ""
    for p in pages:
        if buf and (len(buf) + len(p) < 4000):
            buf = buf + "\n\n" + p
        elif buf:
            if len(buf.strip()) >= min_page_chars:
                out.append(buf.strip())
            buf = p
        else:
            buf = p
    if buf and len(buf.strip()) >= min_page_chars:
        out.append(buf.strip())
    return out
