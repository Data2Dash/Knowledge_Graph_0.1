# app/core/settings.py
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ChunkStrategy = Literal["semantic", "sections", "sliding", "pages"]
EnvName = Literal["dev", "staging", "prod"]


class Settings(BaseSettings):
    """
    Centralized environment configuration (Pydantic v2).

    Design rules:
    - No side effects on import (no I/O, no directory creation)
    - Validate cross-field constraints
    - Keep secrets here (NOT in PipelineConfig)
    """

    # ======================
    # App / Environment
    # ======================
    APP_NAME: str = Field(default="knowledge_graph_v1")
    ENV: EnvName = Field(default="dev")
    DEBUG: bool = Field(default=False)

    # Where runtime artifacts go (logs, exported HTML, etc.)
    OUTPUT_DIR: Path = Field(default=Path("outputs"))
    DATA_DIR: Path = Field(default=Path("data"))
    CACHE_DIR: Path = Field(default=Path(".cache"))

    # ======================
    # LLM (Groq)
    # ======================
    LLM_ENABLED: bool = Field(default=True, description="Disable to run in offline/test mode")

    GROQ_API_KEY: SecretStr = Field(default=SecretStr(""), description="Groq API key")
    MODEL_NAME: str = Field(default="llama-3.1-8b-instant")
    TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)
    MAX_OUTPUT_TOKENS: int = Field(default=1200, gt=0)

    # Input budgets / stability
    MAX_CHUNK_CHARS_FOR_LLM: int = Field(default=6000, ge=500)
    LLM_INPUT_TOKEN_BUDGET: int = Field(default=6500, ge=512)
    LLM_TIMEOUT_S: float = Field(default=60.0, gt=1.0)
    LLM_JSON_ONLY: bool = Field(default=True)

    # ======================
    # Logging
    # ======================
    LOG_LEVEL: str = Field(default="INFO")
    # IMPORTANT: make it relative to OUTPUT_DIR by default
    LOG_DIR: Path = Field(default=Path("logs"))
    LOG_JSON: bool = Field(default=False)

    # ======================
    # Chunking
    # ======================
    CHUNK_STRATEGY: ChunkStrategy = Field(default="semantic")
    MAX_TOTAL_CHUNKS: int = Field(default=60, gt=0)
    PRIORITIZE_TOP_K: int = Field(default=40, gt=0)

    SEMANTIC_MIN_PARAGRAPH_CHARS: int = Field(default=160, ge=0)
    SEMANTIC_TARGET_CHUNK_CHARS: int = Field(default=2600, gt=0)
    SEMANTIC_MAX_CHUNK_CHARS: int = Field(default=3400, gt=0)
    SEMANTIC_OVERLAP_PARAGRAPHS: int = Field(default=1, ge=0)
    SEMANTIC_SIM_THRESHOLD: float = Field(default=0.78, ge=0.0, le=1.0)

    # Token estimation
    APPROX_CHARS_PER_TOKEN: float = Field(default=4.0, gt=0.0)
    CHUNK_TOKEN_SOFT_CAP: int = Field(default=700, gt=0)
    CHUNK_TOKEN_HARD_CAP: int = Field(default=950, gt=0)

    # ======================
    # Embeddings
    # ======================
    EMBEDDINGS_MODEL: str = Field(default="all-MiniLM-L6-v2")
    EMBEDDING_BATCH_SIZE: int = Field(default=64, gt=0)
    EMBEDDING_MAX_TEXTS_PER_CALL: int = Field(default=256, gt=0)
    ENABLE_EMBEDDING_CACHE: bool = Field(default=True)
    EMBEDDING_CACHE_DIR: Path = Field(default=Path(".cache/embeddings"))
    NORMALIZE_EMBEDDINGS: bool = Field(default=True)

    # ======================
    # Extraction (async / retries)
    # ======================
    MAX_CONCURRENT_CHUNKS: int = Field(default=6, gt=0)
    MAX_RETRIES: int = Field(default=3, ge=0)
    RETRY_BASE_DELAY: float = Field(default=1.0, gt=0.0)
    RETRY_MAX_DELAY: float = Field(default=20.0, gt=0.0)
    RETRY_JITTER: bool = Field(default=True)

    ENABLE_DIRECT_EXTRACTOR: bool = Field(default=True)
    MIN_RELATIONSHIPS_TARGET: int = Field(default=35, ge=0)
    MAX_DIRECT_PASSES: int = Field(default=8, ge=0)

    # ======================
    # GraphRAG
    # ======================
    ENABLE_GRAPHRAG: bool = Field(default=True)
    GRAPHRAG_TOP_K_CHUNKS: int = Field(default=12, gt=0)
    GRAPHRAG_MAX_HOPS: int = Field(default=2, ge=0)
    GRAPHRAG_MAX_SEED_ENTITIES: int = Field(default=40, gt=0)
    GRAPHRAG_MAX_TRIPLES: int = Field(default=80, gt=0)

    GRAPHRAG_W_CHUNK: float = Field(default=1.0, ge=0.0)
    GRAPHRAG_W_REL_CONF: float = Field(default=1.0, ge=0.0)
    GRAPHRAG_W_REL_FREQ: float = Field(default=0.2, ge=0.0)

    # ======================
    # Neo4j
    # ======================
    SYNC_NEO4J: bool = Field(default=False)
    NEO4J_URL: str = Field(default="bolt://localhost:7687")
    NEO4J_USER: str = Field(default="neo4j")
    NEO4J_PASSWORD: SecretStr = Field(default=SecretStr(""))
    NEO4J_DATABASE: str = Field(default="")
    NEO4J_CREATE_CONSTRAINTS: bool = Field(default=True)

    # ======================
    # Pydantic settings config
    # ======================
    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @model_validator(mode="after")
    def _cross_field_validate(self) -> "Settings":
        if self.PRIORITIZE_TOP_K > self.MAX_TOTAL_CHUNKS:
            raise ValueError("PRIORITIZE_TOP_K must be <= MAX_TOTAL_CHUNKS")

        if self.SEMANTIC_TARGET_CHUNK_CHARS > self.SEMANTIC_MAX_CHUNK_CHARS:
            raise ValueError("SEMANTIC_TARGET_CHUNK_CHARS must be <= SEMANTIC_MAX_CHUNK_CHARS")

        if self.CHUNK_TOKEN_SOFT_CAP > self.CHUNK_TOKEN_HARD_CAP:
            raise ValueError("CHUNK_TOKEN_SOFT_CAP must be <= CHUNK_TOKEN_HARD_CAP")

        if self.RETRY_MAX_DELAY < self.RETRY_BASE_DELAY:
            raise ValueError("RETRY_MAX_DELAY must be >= RETRY_BASE_DELAY")

        # Only require key when LLM is enabled
        if self.LLM_ENABLED:
            key = self.GROQ_API_KEY.get_secret_value().strip()
            if not key:
                raise ValueError("GROQ_API_KEY is required when LLM_ENABLED=true")

        return self

    def resolved_log_dir(self) -> Path:
        """
        LOG_DIR is treated as:
        - absolute -> use as is
        - relative -> relative to OUTPUT_DIR
        """
        p = Path(self.LOG_DIR)
        return p if p.is_absolute() else (Path(self.OUTPUT_DIR) / p)

    def ensure_runtime_dirs(self) -> None:
        """
        Call this ONCE at app startup (main/streamlit entry),
        not at import time.
        """
        Path(self.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.CACHE_DIR).mkdir(parents=True, exist_ok=True)
        self.resolved_log_dir().mkdir(parents=True, exist_ok=True)
        if self.ENABLE_EMBEDDING_CACHE:
            Path(self.EMBEDDING_CACHE_DIR).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings instance.
    Supports ENV_FILE override.

    - ENV_FILE can point to a custom env file.
    - If ENV_FILE is set to empty string, skip env file loading.
    """
    env_file = os.getenv("ENV_FILE", ".env")

    # Allow turning off env file loading explicitly
    if env_file is not None and str(env_file).strip() == "":
        return Settings()

    p = Path(env_file) if env_file else None
    if p and p.exists():
        return Settings(_env_file=str(p))

    return Settings()