# app/knowledge_graph/llm/retry.py
from __future__ import annotations

import asyncio
import random
from typing import Optional


def compute_backoff_delay(
    base_s: float,
    attempt: int,
    *,
    max_delay_s: float = 30.0,
    jitter: bool = True,
    retry_after_s: Optional[float] = None,
) -> float:
    """
    Exponential backoff delay calculator.

    Default (no Retry-After):
      delay = min(max_delay_s, base_s * 2^attempt)

    If jitter=True:
      Uses "full jitter" to reduce synchronized retries:
        delay = random.uniform(0, delay)

    If retry_after_s is provided (e.g., from HTTP Retry-After header),
    we respect it as a minimum:
      delay = max(delay, retry_after_s)
    """
    # Defensive normalization
    if base_s <= 0:
        base_s = 1.0
    if attempt < 0:
        attempt = 0
    if max_delay_s <= 0:
        max_delay_s = 30.0

    delay = base_s * (2 ** attempt)
    delay = min(delay, max_delay_s)

    if jitter:
        delay = random.uniform(0.0, delay)

    if retry_after_s is not None and retry_after_s > 0:
        # Respect server guidance as a minimum.
        delay = max(delay, float(retry_after_s))

    # Never negative
    return max(0.0, float(delay))


async def backoff_sleep(
    base_s: float,
    attempt: int,
    *,
    max_delay_s: float = 30.0,
    jitter: bool = True,
    retry_after_s: Optional[float] = None,
) -> None:
    """
    Async exponential backoff sleep with cap + jitter + optional Retry-After.
    """
    delay = compute_backoff_delay(
        base_s=base_s,
        attempt=attempt,
        max_delay_s=max_delay_s,
        jitter=jitter,
        retry_after_s=retry_after_s,
    )
    await asyncio.sleep(delay)