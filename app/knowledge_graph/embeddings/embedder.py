from sentence_transformers import SentenceTransformer
import math
from typing import List

_model = SentenceTransformer("all-MiniLM-L6-v2")


class Embedding:
    def __init__(self, values: List[float]):
        self.values = values


def embed_texts(texts: List[str]) -> List[Embedding]:
    """
    Convert list of texts into semantic embeddings
    """
    vectors = _model.encode(
        texts,
        normalize_embeddings=True,  # مهم جدًا
        show_progress_bar=False
    )
    return [Embedding(values=v.tolist()) for v in vectors]


def cosine(a: List[float], b: List[float]) -> float:
    """
    Cosine similarity between two vectors
    """
    return sum(x * y for x, y in zip(a, b))
