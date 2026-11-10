import time

import pytest

from mltemplate.async_utils import (
    fake_api_call,
    retry,
    run_concurrently,
    with_timeout,
)


async def test_concurrent_calls_run_in_parallel() -> None:
    start = time.perf_counter()
    results = await run_concurrently(list(range(10)), limit=10, delay=0.2)
    elapsed = time.perf_counter() - start
    assert results == [f"result_{i}" for i in range(10)]
    assert elapsed < 1.0  # sequential would take about 2.0 s


async def test_semaphore_limits_concurrency() -> None:
    start = time.perf_counter()
    await run_concurrently(list(range(4)), limit=2, delay=0.2)
    elapsed = time.perf_counter() - start
    assert elapsed >= 0.35  # two batches of 0.2 s each


async def test_timeout_raises() -> None:
    with pytest.raises(TimeoutError):
        await with_timeout(fake_api_call(1, delay=1.0), seconds=0.05)


async def test_retry_succeeds_after_failures() -> None:
    calls = {"count": 0}

    async def flaky() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise ConnectionError("temporary failure")
        return "ok"

    assert await retry(flaky, attempts=3, base_delay=0.01) == "ok"
    assert calls["count"] == 3


async def test_retry_gives_up() -> None:
    async def always_fails() -> str:
        raise ConnectionError("service down")

    with pytest.raises(ConnectionError):
        await retry(always_fails, attempts=2, base_delay=0.01)
