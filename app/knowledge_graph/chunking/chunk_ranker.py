from __future__ import annotations
from typing import List
from app.knowledge_graph.chunking.semantic_chunker import Chunk

PRIORITY_KEYWORDS = [
    "method", "architecture", "model", "transformer", "attention",
    "experiment", "results", "evaluation", "dataset", "benchmark",
    "training", "loss", "approach", "encoder", "decoder",
    "proposed", "compared", "achieves", "outperforms",
    "baseline", "metric", "accuracy", "f1", "bleu",
    "ablation", "limitation", "hyperparameter", "contribution"
]

def rank_chunks(chunks: List[Chunk]) -> List[Chunk]:
    scored = []
    for ch in chunks:
        t = ch.text.lower()
        score = sum(k in t for k in PRIORITY_KEYWORDS)
        score += 1 if "##" in ch.text else 0
        scored.append((score, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]
