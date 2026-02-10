from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from app.knowledge_graph.embeddings.embedder import embed_texts, cosine

@dataclass
class VectorItem:
    id: str
    text: str
    emb: List[float]

class InMemoryVectorStore:
    def __init__(self):
        self.items: List[VectorItem] = []

    def add_texts(self, ids: List[str], texts: List[str]) -> None:
        embs = embed_texts(texts)
        for i, t, e in zip(ids, texts, embs):
            self.items.append(VectorItem(id=i, text=t, emb=e.values))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        q = embed_texts([query])[0].values
        scored = []
        for it in self.items:
            scored.append((it.id, it.text, cosine(q, it.emb)))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]
