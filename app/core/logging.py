# app/core/logging.py
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_LOGGING_CONFIGURED = False


class _JsonLikeFormatter(logging.Formatter):
    """
    Lightweight JSON-ish formatter without external deps.
    (Not strict ECS/OTEL, but good enough for prod logs.)
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(
    *,
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_json: bool = False,
    app_name: str = "knowledge_graph_v1",
) -> None:
    """
    Configure root logging ONCE.

    Call this once at app startup (main.py / streamlit_app.py).
    Safe to call multiple times (idempotent).
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers to avoid Streamlit duplicate logs
    # (but only if they look like default/previous handlers)
    root.handlers.clear()

    formatter: logging.Formatter
    if log_json:
        formatter = _JsonLikeFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Optional file handler
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh_path = log_dir / f"{app_name}.log"
        fh = logging.FileHandler(fh_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # Reduce noise from chatty libraries
    for noisy in [
        "urllib3",
        "httpx",
        "neo4j",
        "streamlit",
        "matplotlib",
        "PIL",
    ]:
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    _LOGGING_CONFIGURED = True


def setup_logging(name: str = "knowledge_graph_v1") -> logging.Logger:
    """
    Backward-compatible helper:
    returns a named logger. Logging should be configured via configure_logging().
    """
    return logging.getLogger(name)