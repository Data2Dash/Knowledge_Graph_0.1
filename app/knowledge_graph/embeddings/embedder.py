# app/knowledge_graph/embeddings/embedder.py
from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.settings import get_settings
from app.core.logging import setup_logging

LOGGER = setup_logging("knowledge_graph.embedder")


# -----------------------------
# Public type (compatibility)
# -----------------------------
@dataclass(frozen=True, slots=True)
class Embedding:
    values: np.ndarray  # store numpy directly

    def __iter__(self):
        return iter(self.values.tolist())

    def __len__(self):
        return int(self.values.shape[0])

    def __getitem__(self, idx):
        return float(self.values[idx])


VectorLike = Union[Embedding, Sequence[float], np.ndarray]


# -----------------------------
# Model registry (singleton per process)
# -----------------------------
_MODELS: Dict[str, SentenceTransformer] = {}
_MODELS_LOCK = threading.Lock()


def _get_model(model_name: str) -> SentenceTransformer:
    m = _MODELS.get(model_name)
    if m is None:
        with _MODELS_LOCK:
            m = _MODELS.get(model_name)
            if m is None:
                LOGGER.info("Loading embedding model", extra={"model": model_name})
                m = SentenceTransformer(model_name)
                _MODELS[model_name] = m
    return m


# -----------------------------
# Cache helpers
# -----------------------------
def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()


def _cache_path(base: Path, key: str) -> Path:
    return base / f"{key}.npy"


def _ensure_cache_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Core API
# -----------------------------
def embed_texts(texts: List[str]) -> List[Embedding]:
    """
    Production embedding pipeline.

    - Uses Settings (not PipelineConfig)
    - Singleton model
    - Disk cache
    - Batching
    - Optional normalization
    """

    if not texts:
        return []

    settings = get_settings()

    model_name = settings.EMBEDDINGS_MODEL
    normalize = settings.NORMALIZE_EMBEDDINGS
    use_cache = settings.ENABLE_EMBEDDING_CACHE

    cache_dir = Path(settings.EMBEDDING_CACHE_DIR)
    if use_cache:
        _ensure_cache_dir(cache_dir)

    keys: List[str] = []
    cached: List[Optional[np.ndarray]] = [None] * len(texts)
    missing_idx: List[int] = []

    salt = f"{model_name}|norm={int(normalize)}|v2"

    for i, t in enumerate(texts):
        t_clean = (t or "").strip() or " "
        key = _sha256(salt + "|" + t_clean)
        keys.append(key)

        if use_cache:
            p = _cache_path(cache_dir, key)
            if p.exists():
                try:
                    cached[i] = np.load(p)
                    continue
                except Exception:
                    pass

        missing_idx.append(i)

    if missing_idx:
        model = _get_model(model_name)

        max_per_call = max(1, settings.EMBEDDING_MAX_TEXTS_PER_CALL)
        bs = max(1, settings.EMBEDDING_BATCH_SIZE)

        start = 0
        while start < len(missing_idx):
            sub = missing_idx[start : start + max_per_call]
            batch_texts = [(texts[i] or "").strip() or " " for i in sub]

            vecs = model.encode(
                batch_texts,
                batch_size=bs,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )

            vecs = np.asarray(vecs, dtype=np.float32)

            for local_j, global_i in enumerate(sub):
                v = vecs[local_j]
                cached[global_i] = v
                if use_cache:
                    try:
                        np.save(_cache_path(cache_dir, keys[global_i]), v)
                    except Exception:
                        pass

            start += max_per_call

    out: List[Embedding] = []
    for v in cached:
        if v is None:
            out.append(Embedding(values=np.zeros(1, dtype=np.float32)))
        else:
            out.append(Embedding(values=v.astype(np.float32, copy=False)))

    return out


# -----------------------------
# Cosine similarity
# -----------------------------
def cosine(a: VectorLike, b: VectorLike) -> float:
    av = np.asarray(a.values if isinstance(a, Embedding) else a, dtype=np.float32)
    bv = np.asarray(b.values if isinstance(b, Embedding) else b, dtype=np.float32)

    if av.size == 0 or bv.size == 0:
        return 0.0

    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na == 0.0 or nb == 0.0:
        return 0.0

    if abs(na - 1.0) < 1e-2 and abs(nb - 1.0) < 1e-2:
        return float(np.dot(av, bv))

    return float(np.dot(av, bv) / (na * nb))