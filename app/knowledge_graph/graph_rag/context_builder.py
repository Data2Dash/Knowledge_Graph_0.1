from __future__ import annotations
from typing import List, Tuple

def build_context(retrieved_chunks: List[Tuple[str, str, float]], relations_text: str = "") -> str:
    parts = ["# Retrieved Evidence Chunks\n"]
    for cid, txt, score in retrieved_chunks:
        parts.append(f"\n## Chunk {cid} (score={score:.3f})\n{txt}\n")
    if relations_text:
        parts.append("\n# Graph Facts\n" + relations_text)
    return "\n".join(parts).strip()
