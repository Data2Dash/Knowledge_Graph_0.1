# app/core/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


ChunkStrategy = Literal["semantic", "sections", "sliding", "pages"]


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """
    Pipeline tuning knobs (NOT secrets).

    - Safe to import anywhere
    - No environment credentials here
    - Settings (app/core/settings.py) remains source of truth for secrets + runtime env
    """

    # -----------------------------
    # LLM generation (per-run tuning)
    # -----------------------------
    model_name: str = "llama-3.1-8b-instant"
    temperature: float = 0.0
    request_timeout_s: float = 60.0
    seed: Optional[int] = 42  # optional; only effective if LLM backend supports it

    # limit raw text size sent to the LLM per chunk
    max_chunk_chars_for_llm: int = 6000

    # -----------------------------
    # Chunking / ranking
    # -----------------------------
    chunk_strategy: ChunkStrategy = "semantic"
    max_total_chunks: int = 40
    prioritize_top_k: int = 28  # keep top K ranked chunks

    semantic_min_paragraph_chars: int = 160
    semantic_target_chunk_chars: int = 2600
    semantic_max_chunk_chars: int = 3400
    semantic_overlap_paragraphs: int = 1
    semantic_sim_threshold: float = 0.78

    # -----------------------------
    # Concurrency / retries
    # -----------------------------
    max_concurrent_chunks: int = 6
    max_retries: int = 3
    retry_base_delay_s: float = 1.0
    retry_max_delay_s: float = 20.0
    retry_jitter: bool = True

    # -----------------------------
    # Extraction quality control (per-chunk)
    # -----------------------------
    entity_confidence_threshold: float = 0.5
    max_entities_per_chunk: int = 60

    relation_confidence_threshold: float = 0.5
    max_relations_per_chunk: int = 50
    relation_max_entities_in_prompt: int = 30  # how many entities to show the model for grounding

    # -----------------------------
    # Pipeline quality control (global)
    # -----------------------------
    min_relationships_target: int = 35
    max_direct_passes: int = 8

    # NOTE: embeddings are configured via Settings (EMBEDDINGS_MODEL, etc.)
    # Keeping an embedding model here risks having two sources of truth.

    def __post_init__(self) -> None:
        if not self.model_name or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")

        if not (0.0 <= float(self.temperature) <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")

        if self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be > 0")

        if self.max_chunk_chars_for_llm < 500:
            raise ValueError("max_chunk_chars_for_llm is too small to be useful")

        if self.max_total_chunks <= 0:
            raise ValueError("max_total_chunks must be > 0")

        if self.prioritize_top_k <= 0:
            raise ValueError("prioritize_top_k must be > 0")

        if self.prioritize_top_k > self.max_total_chunks:
            raise ValueError("prioritize_top_k cannot exceed max_total_chunks")

        if self.semantic_min_paragraph_chars <= 0:
            raise ValueError("semantic_min_paragraph_chars must be > 0")

        if self.semantic_target_chunk_chars <= 0 or self.semantic_max_chunk_chars <= 0:
            raise ValueError("semantic_target_chunk_chars/semantic_max_chunk_chars must be > 0")

        if self.semantic_target_chunk_chars > self.semantic_max_chunk_chars:
            raise ValueError("semantic_target_chunk_chars cannot exceed semantic_max_chunk_chars")

        if self.semantic_overlap_paragraphs < 0:
            raise ValueError("semantic_overlap_paragraphs must be >= 0")

        if not (0.0 <= float(self.semantic_sim_threshold) <= 1.0):
            raise ValueError("semantic_sim_threshold must be between 0.0 and 1.0")

        if self.max_concurrent_chunks <= 0:
            raise ValueError("max_concurrent_chunks must be > 0")

        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        if self.retry_base_delay_s <= 0:
            raise ValueError("retry_base_delay_s must be > 0")

        if self.retry_max_delay_s < self.retry_base_delay_s:
            raise ValueError("retry_max_delay_s must be >= retry_base_delay_s")

        if not (0.0 <= float(self.entity_confidence_threshold) <= 1.0):
            raise ValueError("entity_confidence_threshold must be between 0.0 and 1.0")

        if self.max_entities_per_chunk <= 0:
            raise ValueError("max_entities_per_chunk must be > 0")

        if not (0.0 <= float(self.relation_confidence_threshold) <= 1.0):
            raise ValueError("relation_confidence_threshold must be between 0.0 and 1.0")

        if self.max_relations_per_chunk <= 0:
            raise ValueError("max_relations_per_chunk must be > 0")

        if self.relation_max_entities_in_prompt <= 0:
            raise ValueError("relation_max_entities_in_prompt must be > 0")

        if self.min_relationships_target < 0:
            raise ValueError("min_relationships_target must be >= 0")

        if self.max_direct_passes <= 0:
            raise ValueError("max_direct_passes must be > 0")