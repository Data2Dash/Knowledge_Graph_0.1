from __future__ import annotations
from typing import List, Tuple
from app.knowledge_graph.store.vector_store import InMemoryVectorStore

def retrieve_chunks(vstore: InMemoryVectorStore, query: str, top_k: int = 6) -> List[Tuple[str, str, float]]:
    return vstore.search(query, top_k=top_k)
