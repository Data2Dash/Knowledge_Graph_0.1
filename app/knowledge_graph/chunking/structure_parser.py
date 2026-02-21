from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class Paragraph:
    text: str

def to_paragraphs(text: str, min_chars: int = 160) -> List[Paragraph]:
    raw = [p.strip() for p in text.split("\n\n") if p.strip()]
    paras = []
    for p in raw:
        if len(p) >= min_chars:
            paras.append(Paragraph(text=p))
    # If everything is short (messy PDFs), fallback to line-merge
    if not paras and text.strip():
        paras = [Paragraph(text=text.strip())]
    return paras
