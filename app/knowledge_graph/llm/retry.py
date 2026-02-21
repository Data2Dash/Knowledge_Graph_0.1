from __future__ import annotations
import asyncio
import random

async def backoff_sleep(base: float, attempt: int) -> None:
    delay = base * (2 ** attempt) + random.random() * 0.25
    await asyncio.sleep(delay)
