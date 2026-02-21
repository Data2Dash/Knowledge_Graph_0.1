# app/knowledge_graph/extraction/async_runner.py
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional, TypeVar, Union

from app.core.config import PipelineConfig
from app.core.logging import setup_logging
from app.knowledge_graph.llm.retry import backoff_sleep

T = TypeVar("T")

LOGGER = setup_logging("knowledge_graph.async_runner")


# ==========================================================
# Stats
# ==========================================================

@dataclass(frozen=True, slots=True)
class AsyncRunStats:
    total: int
    succeeded: int
    failed: int
    retries: int
    duration_s: float


# ==========================================================
# Retry Logic
# ==========================================================

def _is_retryable(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()

    if isinstance(exc, asyncio.TimeoutError):
        return True

    transient_markers = [
        "rate limit", "429",
        "timeout", "timed out",
        "temporarily unavailable",
        "service unavailable", "503",
        "bad gateway", "502",
        "gateway timeout", "504",
        "connection reset", "connection aborted",
        "network is unreachable",
        "dns", "name resolution",
        "server error", "internal server error", "500",
    ]

    if any(m in msg for m in transient_markers):
        return True

    if "ratelimit" in name or "timeout" in name:
        return True

    return False


# ==========================================================
# Core Runner
# ==========================================================

async def run_bounded(
    cfg: PipelineConfig,
    jobs: List[Callable[[], Awaitable[T]]],
    *,
    return_exceptions: bool = False,
    context: Optional[dict] = None,
    job_timeout_s: Optional[float] = None,
    global_timeout_s: Optional[float] = None,
    fail_fast: bool = False,
) -> Union[List[T], List[Union[T, Exception]]]:
    """
    Production-grade bounded async runner.

    Features:
    - Bounded concurrency (Semaphore)
    - Per-job timeout
    - Retry with exponential backoff (+ jitter)
    - Optional global timeout
    - Optional fail-fast mode (cancel remaining tasks on first non-retryable final failure)
    - Structured logging + stats
    """
    context = context or {}
    start_time = time.time()

    concurrency = max(1, int(getattr(cfg, "max_concurrent_chunks", 1)))
    max_retries = max(0, int(getattr(cfg, "max_retries", 0)))
    base_delay_s = float(getattr(cfg, "retry_base_delay_s", 1.0))
    max_delay_s = float(getattr(cfg, "retry_max_delay_s", 30.0))
    jitter = bool(getattr(cfg, "retry_jitter", True)) if hasattr(cfg, "retry_jitter") else True

    sem = asyncio.Semaphore(concurrency)

    retries_used = 0
    failures = 0

    stop_event = asyncio.Event()  # used for fail_fast

    async def one(i: int, job: Callable[[], Awaitable[T]]) -> Union[T, Exception]:
        nonlocal retries_used, failures

        async with sem:
            # if fail_fast triggered while waiting for semaphore
            if fail_fast and stop_event.is_set():
                return RuntimeError("Cancelled due to fail_fast")

            for attempt in range(max_retries + 1):
                if fail_fast and stop_event.is_set():
                    return RuntimeError("Cancelled due to fail_fast")

                try:
                    if job_timeout_s:
                        return await asyncio.wait_for(job(), timeout=job_timeout_s)
                    return await job()

                except asyncio.CancelledError:
                    raise

                except Exception as e:
                    retryable = _is_retryable(e)
                    last_attempt = attempt >= max_retries

                    # log attempt
                    LOGGER.warning(
                        "Job failed",
                        extra={
                            **context,
                            "job_index": i,
                            "attempt": attempt,
                            "max_retries": max_retries,
                            "retryable": retryable,
                            "error_type": e.__class__.__name__,
                            "error": str(e)[:500],
                        },
                    )

                    if last_attempt or not retryable:
                        failures += 1
                        if fail_fast:
                            stop_event.set()
                        if return_exceptions:
                            return e
                        raise

                    retries_used += 1
                    await backoff_sleep(
                        base_s=base_delay_s,
                        attempt=attempt,
                        max_delay_s=max_delay_s,
                        jitter=jitter,
                    )

            failures += 1
            return RuntimeError("Unexpected retry flow")  # pragma: no cover

    tasks = [asyncio.create_task(one(i, job)) for i, job in enumerate(jobs)]

    async def _gather_all():
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)

    try:
        if global_timeout_s:
            results = await asyncio.wait_for(_gather_all(), timeout=global_timeout_s)
        else:
            results = await _gather_all()

    except Exception:
        # cancel all running tasks
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    # If fail_fast triggered, cancel remaining still-pending tasks
    if fail_fast and stop_event.is_set():
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    duration = time.time() - start_time
    succeeded = sum(1 for r in results if not isinstance(r, Exception))
    failed = len(results) - succeeded

    stats = AsyncRunStats(
        total=len(results),
        succeeded=succeeded,
        failed=failed,
        retries=retries_used,
        duration_s=duration,
    )

    LOGGER.info(
        "Async batch completed",
        extra={
            **context,
            "total": stats.total,
            "succeeded": stats.succeeded,
            "failed": stats.failed,
            "retries": stats.retries,
            "duration_s": round(stats.duration_s, 3),
            "concurrency": concurrency,
        },
    )

    if fail_fast and failed > 0 and not return_exceptions:
        raise RuntimeError("Fail-fast mode: at least one job failed.")

    return results