from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from langchain_groq import ChatGroq

from app.core.config import PipelineConfig
from app.core.logging import setup_logging
from app.core.settings import Settings, get_settings

LOGGER = setup_logging("knowledge_graph.groq_client")

# Cache by (model, temp, max_tokens, json_mode)
_LLM_CACHE: Dict[Tuple[str, float, int, bool], ChatGroq] = {}


def _resolved_params(cfg: PipelineConfig, settings: Settings) -> Dict[str, Any]:
    model_name = (cfg.model_name or settings.MODEL_NAME).strip()
    if not model_name:
        raise ValueError("Resolved model_name is empty")

    temperature = float(getattr(cfg, "temperature", settings.TEMPERATURE))
    max_tokens = int(getattr(settings, "MAX_OUTPUT_TOKENS", 1200))
    json_only = bool(getattr(settings, "LLM_JSON_ONLY", True))

    timeout_s = float(getattr(cfg, "request_timeout_s", getattr(settings, "LLM_TIMEOUT_S", 60.0)))

    return {
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout_s": timeout_s,
        "json_only": json_only,
    }


def build_llm(
    cfg: Optional[PipelineConfig] = None,
    *,
    settings: Optional[Settings] = None,
    json_mode: bool = True,
) -> ChatGroq:
    """
    Production-safe Groq LLM builder.

    - json_mode=True  => enforce JSON responses (for extraction)
    - json_mode=False => normal text output (for Ask-the-Graph)
    """

    s = settings or get_settings()
    cfg = cfg or PipelineConfig()

    if not getattr(s, "LLM_ENABLED", True):
        raise RuntimeError("LLM is disabled (LLM_ENABLED=false).")

    api_key = s.GROQ_API_KEY.get_secret_value().strip()
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY")

    p = _resolved_params(cfg, s)

    # final decision: JSON mode only if settings say so AND caller wants it
    use_json = bool(p["json_only"] and json_mode)

    cache_key = (p["model_name"], float(p["temperature"]), int(p["max_tokens"]), use_json)
    if cache_key in _LLM_CACHE:
        return _LLM_CACHE[cache_key]

    # Build kwargs (compatible)
    llm_kwargs: Dict[str, Any] = {
        "groq_api_key": api_key,
        "model": p["model_name"],
        "temperature": p["temperature"],
    }

    # Token limit: try most common kw names safely
    # Some versions use max_tokens, others use max_output_tokens
    max_tokens = int(p["max_tokens"])
    try:
        llm_kwargs["max_output_tokens"] = max_tokens
        llm = ChatGroq(**llm_kwargs)
    except TypeError:
        llm_kwargs.pop("max_output_tokens", None)
        llm_kwargs["max_tokens"] = max_tokens
        llm = ChatGroq(**llm_kwargs)

    # JSON strict mode (Groq requires "json" in prompt — we already fixed prompts)
    if use_json:
        try:
            llm = llm.bind(response_format={"type": "json_object"})
        except Exception:
            # If bind isn't supported, fall back to passing at init (some versions)
            try:
                llm = ChatGroq(**{**llm_kwargs, "response_format": {"type": "json_object"}})
            except Exception:
                pass

    LOGGER.info(
        "Groq LLM initialized",
        extra={
            "model": p["model_name"],
            "temperature": p["temperature"],
            "max_tokens": p["max_tokens"],
            "json_mode": use_json,
        },
    )

    _LLM_CACHE[cache_key] = llm
    return llm