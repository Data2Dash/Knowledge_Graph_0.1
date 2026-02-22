from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class PipelineConfig:
    model_name: str = "llama-3.1-8b-instant"
    temperature: float = 0.0
    max_chunk_chars_for_llm: int = 6000

    chunk_strategy: str = "semantic"  # semantic | sections | sliding | pages
    max_total_chunks: int = 40
    prioritize_top_k: int = 28

    semantic_min_paragraph_chars: int = 160
    semantic_target_chunk_chars: int = 2600
    semantic_max_chunk_chars: int = 3400
    semantic_overlap_paragraphs: int = 1
    semantic_sim_threshold: float = 0.78

    max_concurrent_chunks: int = 6
    max_retries: int = 3
    retry_base_delay: float = 1.0

    min_relationships_target: int = 35
    max_direct_passes: int = 8

    sync_neo4j: bool = False
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
