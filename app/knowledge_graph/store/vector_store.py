from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import threading

import numpy as np

from app.core.config import PipelineConfig  # kept for backward-compatible signatures
from app.core.settings import get_settings
from app.knowledge_graph.embeddings.embedder import embed_texts, Embedding


@dataclass(slots=True)
class VectorItem:
    id: str
    text: str
    emb: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)


def _to_vec(e: Embedding | np.ndarray | List[float]) -> np.ndarray:
    if isinstance(e, np.ndarray):
        v = e
    elif isinstance(e, Embedding):
        v = np.asarray(e.values, dtype=np.float32)
    else:
        v = np.asarray(e, dtype=np.float32)
    return v.astype(np.float32, copy=False)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32, copy=False)
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return v
    return v / n


class InMemoryVectorStore:
    """
    Production-ready in-memory vector store.

    - Thread-safe
    - Upsert support
    - Lazy matrix rebuild
    - Fast vectorized cosine search (normalized dot product)
    - Metadata filtering
    - Capacity control (FIFO eviction)
    - Embedding dimension & model consistency checks
    """

    def __init__(self, max_items: Optional[int] = None):
        self._items: Dict[str, VectorItem] = {}
        self._id_order: List[str] = []
        self._mat: Optional[np.ndarray] = None
        self._dirty: bool = True
        self._lock = threading.RLock()
        self._max_items = int(max_items) if max_items is not None else None

        self._dim: Optional[int] = None
        self._embedding_model_name: Optional[str] = None

    @property
    def size(self) -> int:
        return len(self._id_order)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._id_order.clear()
            self._mat = None
            self._dirty = True
            self._dim = None
            self._embedding_model_name = None

    def get(self, item_id: str) -> Optional[VectorItem]:
        with self._lock:
            return self._items.get(item_id)

    def delete(self, item_id: str) -> bool:
        with self._lock:
            if item_id not in self._items:
                return False
            del self._items[item_id]
            try:
                self._id_order.remove(item_id)
            except ValueError:
                pass
            self._dirty = True
            return True

    # ==========================================================
    # Add / Upsert
    # ==========================================================

    def add_texts(
        self,
        ids: List[str],
        texts: List[str],
        *,
        cfg: Optional[PipelineConfig] = None,  # kept for compatibility; not used
        metas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[np.ndarray]] = None,
    ) -> None:
        if not ids or not texts:
            return
        if len(ids) != len(texts):
            raise ValueError("ids and texts must have same length")
        if metas is not None and len(metas) != len(texts):
            raise ValueError("metas must be same length as texts")
        if embeddings is not None and len(embeddings) != len(texts):
            raise ValueError("embeddings must be same length as texts")

        with self._lock:
            # Track embedding model name from Settings (source of truth)
            settings = get_settings()
            model_name = str(settings.EMBEDDINGS_MODEL)

            if self._embedding_model_name is None:
                self._embedding_model_name = model_name
            elif self._embedding_model_name != model_name:
                raise ValueError(
                    f"Embedding model mismatch: store={self._embedding_model_name} vs settings={model_name}"
                )

            # Compute embeddings if not provided
            if embeddings is None:
                embs = embed_texts(texts)  # ✅ fixed: no cfg
                vectors = [_l2_normalize(_to_vec(e)) for e in embs]
            else:
                vectors = [_l2_normalize(_to_vec(v)) for v in embeddings]

            # Dimension check
            for v in vectors:
                if v.ndim != 1:
                    raise ValueError("Each embedding must be a 1D vector")
                if self._dim is None:
                    self._dim = int(v.shape[0])
                elif int(v.shape[0]) != self._dim:
                    raise ValueError(f"Embedding dimension mismatch: expected {self._dim}, got {v.shape[0]}")

            # Upsert items
            for idx, (i, t, v) in enumerate(zip(ids, texts, vectors)):
                if not i:
                    continue

                item_id = str(i)
                meta = metas[idx] if metas else {}
                if meta is None or not isinstance(meta, dict):
                    meta = {}

                item = VectorItem(id=item_id, text=(t or ""), emb=v, meta=meta)

                if item_id in self._items:
                    self._items[item_id] = item
                else:
                    if self._max_items is not None and self.size >= self._max_items:
                        oldest = self._id_order.pop(0)
                        self._items.pop(oldest, None)

                    self._items[item_id] = item
                    self._id_order.append(item_id)

            self._dirty = True

    # ==========================================================
    # Matrix Handling
    # ==========================================================

    def _rebuild_matrix(self) -> None:
        if not self._dirty and self._mat is not None:
            return

        if not self._id_order:
            self._mat = None
            self._dirty = False
            return

        vecs = [self._items[i].emb for i in self._id_order if i in self._items]
        if not vecs:
            self._mat = None
            self._dirty = False
            return

        self._mat = np.vstack(vecs).astype(np.float32, copy=False)
        self._dirty = False

    # ==========================================================
    # Search
    # ==========================================================

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        cfg: Optional[PipelineConfig] = None,  # kept for compatibility; not used
        filter_meta: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, str, float]]:
        q = (query or "").strip()
        if not q:
            return []

        with self._lock:
            self._rebuild_matrix()
            if self._mat is None or self.size == 0:
                return []

            qv = embed_texts([q])[0]  # ✅ fixed: no cfg
            q_vec = _l2_normalize(_to_vec(qv))

            if self._dim is not None and q_vec.shape[0] != self._dim:
                raise ValueError(f"Query embedding dim {q_vec.shape[0]} != store dim {self._dim}")

            ids = self._id_order
            mat = self._mat

            if filter_meta:
                keep_idx: List[int] = []
                for idx, item_id in enumerate(ids):
                    item = self._items.get(item_id)
                    if not item:
                        continue
                    meta = item.meta or {}
                    if all(meta.get(k) == v for k, v in filter_meta.items()):
                        keep_idx.append(idx)

                if not keep_idx:
                    return []

                mat = mat[np.array(keep_idx, dtype=np.int64)]
                ids = [ids[i] for i in keep_idx]

            scores = mat @ q_vec

            k = max(1, min(int(top_k), len(ids)))
            top_idx = np.argpartition(-scores, kth=k - 1)[:k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]

            results: List[Tuple[str, str, float]] = []
            for j in top_idx:
                item_id = ids[int(j)]
                item = self._items[item_id]
                results.append((item.id, item.text, float(scores[int(j)])))
            return results

    def search_by_vector(
        self,
        query_vec: np.ndarray,
        *,
        top_k: int = 5,
        filter_meta: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, str, float]]:
        with self._lock:
            self._rebuild_matrix()
            if self._mat is None or self.size == 0:
                return []

            q_vec = _l2_normalize(_to_vec(query_vec))
            if self._dim is not None and q_vec.shape[0] != self._dim:
                raise ValueError(f"Query embedding dim {q_vec.shape[0]} != store dim {self._dim}")

            ids = self._id_order
            mat = self._mat

            if filter_meta:
                keep_idx: List[int] = []
                for idx, item_id in enumerate(ids):
                    item = self._items.get(item_id)
                    if not item:
                        continue
                    meta = item.meta or {}
                    if all(meta.get(k) == v for k, v in filter_meta.items()):
                        keep_idx.append(idx)
                if not keep_idx:
                    return []

                mat = mat[np.array(keep_idx, dtype=np.int64)]
                ids = [ids[i] for i in keep_idx]

            scores = mat @ q_vec
            k = max(1, min(int(top_k), len(ids)))

            top_idx = np.argpartition(-scores, kth=k - 1)[:k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]

            results: List[Tuple[str, str, float]] = []
            for j in top_idx:
                item_id = ids[int(j)]
                item = self._items[item_id]
                results.append((item.id, item.text, float(scores[int(j)])))
            return results