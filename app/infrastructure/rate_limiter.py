import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Raised by TokenBucketLimiter when no tokens are available."""


class TokenBucketLimiter:
    """
    Asynchronous token-bucket rate limiter.

    The bucket starts full and refills continuously at *requests_per_minute / 60*
    tokens per second, capped at the original capacity.  An asyncio.Lock
    serialises concurrent acquires so the token count stays consistent under
    parallel request load.
    """

    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be a positive integer.")
        self._capacity = float(requests_per_minute)
        self._tokens = float(requests_per_minute)
        self._refill_rate = requests_per_minute / 60.0  # tokens / second
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

    async def acquire_token(self) -> bool:
        """
        Consume one token and return True, or raise RateLimitExceeded if the
        bucket is empty.
        """
        async with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                logger.debug(
                    "Rate limiter: token consumed (%.1f remaining)", self._tokens
                )
                return True
            logger.warning("Rate limit exceeded — bucket empty.")
            raise RateLimitExceeded(
                "Rate limit exceeded. Too many requests — please try again shortly."
            )
