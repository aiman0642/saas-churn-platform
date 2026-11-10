import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def fake_api_call(item: int, delay: float = 0.1) -> str:
    """Stand-in for a network call (LLM API, database, HTTP request)."""
    await asyncio.sleep(delay)
    return f"result_{item}"


async def run_concurrently(
    items: list[int], limit: int = 5, delay: float = 0.1
) -> list[str]:
    """Run calls in parallel, but never more than `limit` at once."""
    semaphore = asyncio.Semaphore(limit)

    async def worker(item: int) -> str:
        async with semaphore:
            return await fake_api_call(item, delay)

    return await asyncio.gather(*(worker(i) for i in items))


async def with_timeout(coro: Awaitable[T], seconds: float) -> T:
    """Fail with TimeoutError if the call takes too long."""
    return await asyncio.wait_for(coro, timeout=seconds)


async def retry(
    func: Callable[[], Awaitable[T]], attempts: int = 3, base_delay: float = 0.1
) -> T:
    """Retry a failing call with exponential backoff (delay doubles each time)."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return await func()
        except Exception as error:
            last_error = error
            await asyncio.sleep(base_delay * 2**attempt)
    assert last_error is not None
    raise last_error
