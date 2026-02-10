from __future__ import annotations
from typing import List
from app.knowledge_graph.chunking.semantic_chunker import Chunk

PRIORITY_KEYWORDS = [

    # Core research signals
    "method", "approach", "architecture", "model", "framework",
    "system", "pipeline", "algorithm", "technique",

    # Training / learning
    "trained", "training", "fine-tuned", "pretrained",
    "optimization", "objective", "loss", "inference",

    # Data
    "dataset", "corpus", "benchmark", "annotations", "samples",

    # Evaluation
    "evaluation", "metric", "accuracy", "precision", "recall",
    "f1", "bleu", "roc", "auc", "performance",

    # Comparison
    "baseline", "comparison", "compared", "outperforms",
    "improves", "improvement", "gain", "state-of-the-art", "sota",

    # Model internals
    "encoder", "decoder", "attention", "layer",
    "embedding", "representation", "feature",

    # Contribution signals
    "proposed", "introduces", "contribution",
    "extends", "based on", "leverages", "incorporates",

    # Research diagnostics
    "ablation", "limitation", "challenge",
    "hyperparameter", "constraint"
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
