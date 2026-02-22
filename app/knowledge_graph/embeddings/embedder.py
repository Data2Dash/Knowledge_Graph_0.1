from __future__ import annotations
import math
import hashlib
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class Embedding:
    values: List[float]

def _hash_to_vec(text: str, dim: int = 64) -> List[float]:
    # Deterministic “good enough” embedding placeholder (no extra deps).
    # Replace later with SentenceTransformers/OpenAI embeddings without changing interfaces.
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    vals = []
    for i in range(dim):
        b = h[i % len(h)]
        vals.append((b / 255.0) * 2 - 1)
    # normalize
    norm = math.sqrt(sum(v*v for v in vals)) or 1.0
    return [v / norm for v in vals]

def embed_texts(texts: List[str], dim: int = 64) -> List[Embedding]:
    return [Embedding(_hash_to_vec(t, dim=dim)) for t in texts]

def cosine(a: List[float], b: List[float]) -> float:
    s = sum(x*y for x, y in zip(a, b))
    return s
