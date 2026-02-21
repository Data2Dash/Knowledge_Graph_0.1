from __future__ import annotations
import asyncio
from typing import Callable, Awaitable, TypeVar, List

from app.core.config import PipelineConfig
from app.knowledge_graph.llm.retry import backoff_sleep

T = TypeVar("T")

async def run_bounded(
    cfg: PipelineConfig,
    jobs: List[Callable[[], Awaitable[T]]],
) -> List[T]:
    sem = asyncio.Semaphore(cfg.max_concurrent_chunks)

    async def one(job: Callable[[], Awaitable[T]]) -> T:
        async with sem:
            for attempt in range(cfg.max_retries):
                try:
                    return await job()
                except Exception:
                    if attempt == cfg.max_retries - 1:
                        raise
                    await backoff_sleep(cfg.retry_base_delay, attempt)
        raise RuntimeError("Unreachable")

    return await asyncio.gather(*[one(j) for j in jobs])
