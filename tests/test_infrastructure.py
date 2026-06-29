"""
Tests for app/infrastructure/ — Tasks 2.1 and 2.2 acceptance criteria.

Run with:  pytest tests/test_infrastructure.py -v
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.infrastructure.cache import CacheService
from app.infrastructure.rate_limiter import TokenBucketLimiter, RateLimitExceeded
from app.infrastructure.usage_tracker import UsageTracker


# ══════════════════════════════════════════════════════════════════════════════
# Task 2.1 — CacheService
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateKey:
    def test_returns_64_char_hex_string(self):
        svc = CacheService.__new__(CacheService)
        key = svc._generate_key("hello", "embeddings")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_is_deterministic(self):
        svc = CacheService.__new__(CacheService)
        k1 = svc._generate_key("same text", "ns")
        k2 = svc._generate_key("same text", "ns")
        assert k1 == k2

    def test_different_text_different_key(self):
        svc = CacheService.__new__(CacheService)
        assert svc._generate_key("a", "ns") != svc._generate_key("b", "ns")

    def test_different_namespace_different_key(self):
        svc = CacheService.__new__(CacheService)
        assert svc._generate_key("text", "ns1") != svc._generate_key("text", "ns2")


class TestCacheServiceFallback:
    """Tests that run against the in-memory fallback (no Redis required)."""

    def _make_fallback_cache(self) -> CacheService:
        svc = CacheService.__new__(CacheService)
        svc._fallback = {}
        svc._using_fallback = True
        svc._redis = None
        return svc

    async def test_get_returns_none_on_miss(self):
        svc = self._make_fallback_cache()
        assert await svc.get("missing-key") is None

    async def test_set_then_get_roundtrip(self):
        svc = self._make_fallback_cache()
        await svc.set("k1", {"answer": 42})
        result = await svc.get("k1")
        assert result == {"answer": 42}

    async def test_set_overwrites_existing(self):
        svc = self._make_fallback_cache()
        await svc.set("k", "first")
        await svc.set("k", "second")
        assert await svc.get("k") == "second"

    async def test_json_serialization_preserves_types(self):
        svc = self._make_fallback_cache()
        payload = {"books": [1, 2, 3], "score": 0.95, "flag": True}
        await svc.set("complex", payload)
        assert await svc.get("complex") == payload


class TestCacheServiceRedisFailure:
    """Verifies graceful degradation when Redis raises on first use."""

    async def test_degrades_to_fallback_on_redis_get_error(self):
        svc = CacheService.__new__(CacheService)
        svc._fallback = {}
        svc._using_fallback = False
        mock_redis = AsyncMock()
        mock_redis.get.side_effect = ConnectionError("Redis down")
        svc._redis = mock_redis

        result = await svc.get("any-key")
        assert result is None
        assert svc._using_fallback is True

    async def test_degrades_to_fallback_on_redis_set_error(self):
        svc = CacheService.__new__(CacheService)
        svc._fallback = {}
        svc._using_fallback = False
        mock_redis = AsyncMock()
        mock_redis.setex.side_effect = ConnectionError("Redis down")
        svc._redis = mock_redis

        await svc.set("key", "value")
        assert svc._using_fallback is True
        # Value should still be retrievable via in-memory fallback
        assert svc._fallback.get("key") is not None


# ══════════════════════════════════════════════════════════════════════════════
# Task 2.2 — TokenBucketLimiter
# ══════════════════════════════════════════════════════════════════════════════

class TestTokenBucketLimiter:
    async def test_acquire_returns_true_when_tokens_available(self):
        limiter = TokenBucketLimiter(requests_per_minute=60)
        assert await limiter.acquire_token() is True

    async def test_raises_when_bucket_empty(self):
        limiter = TokenBucketLimiter(requests_per_minute=1)
        # Drain the single token
        await limiter.acquire_token()
        # Manually empty remaining tokens
        limiter._tokens = 0.0
        with pytest.raises(RateLimitExceeded):
            await limiter.acquire_token()

    async def test_tokens_decrease_on_each_acquire(self):
        limiter = TokenBucketLimiter(requests_per_minute=10)
        before = limiter._tokens
        await limiter.acquire_token()
        assert limiter._tokens < before

    async def test_raises_on_zero_or_negative_capacity(self):
        with pytest.raises(ValueError):
            TokenBucketLimiter(requests_per_minute=0)
        with pytest.raises(ValueError):
            TokenBucketLimiter(requests_per_minute=-5)

    async def test_concurrent_acquires_are_thread_safe(self):
        limiter = TokenBucketLimiter(requests_per_minute=5)
        limiter._tokens = 5.0

        successes = 0
        failures = 0

        async def try_acquire():
            nonlocal successes, failures
            try:
                await limiter.acquire_token()
                successes += 1
            except RateLimitExceeded:
                failures += 1

        await asyncio.gather(*(try_acquire() for _ in range(10)))
        assert successes == 5
        assert failures == 5


# ══════════════════════════════════════════════════════════════════════════════
# Task 2.2 — UsageTracker
# ══════════════════════════════════════════════════════════════════════════════

class TestUsageTracker:
    def test_daily_summary_starts_empty(self):
        tracker = UsageTracker()
        summary = tracker.daily_summary()
        assert summary["request_count"] == 0
        assert summary["total_cost_usd"] == 0.0

    def test_record_increments_request_count(self):
        tracker = UsageTracker()
        tracker.record("gpt-4o-mini", "Hello", "Hi there!")
        assert tracker.daily_summary()["request_count"] == 1

    def test_record_accumulates_across_calls(self):
        tracker = UsageTracker()
        tracker.record("gpt-4o-mini", "Q1", "A1")
        tracker.record("gpt-4o-mini", "Q2", "A2")
        assert tracker.daily_summary()["request_count"] == 2

    def test_cost_is_nonzero_after_record(self):
        tracker = UsageTracker()
        tracker.record("gpt-4o-mini", "What is AI?", "AI is artificial intelligence.")
        assert tracker.daily_summary()["total_cost_usd"] > 0.0

    def test_more_tokens_higher_cost(self):
        tracker = UsageTracker()
        r_short = tracker.record("gpt-4o-mini", "Hi", "Hello")
        r_long = tracker.record(
            "gpt-4o-mini",
            "Explain the entire history of computing in detail.",
            "Computing began with Charles Babbage and Ada Lovelace in the 19th century. "
            "Mechanical calculators evolved into electronic computers after World War II.",
        )
        assert r_long.cost_usd > r_short.cost_usd

    def test_record_returns_usage_record_with_correct_fields(self):
        tracker = UsageTracker()
        rec = tracker.record("gpt-4o", "prompt text", "completion text")
        assert rec.model == "gpt-4o"
        assert rec.prompt_tokens > 0
        assert rec.completion_tokens > 0
        assert rec.cost_usd > 0.0

    def test_unknown_model_uses_default_pricing(self):
        tracker = UsageTracker()
        rec = tracker.record("some-unknown-model-xyz", "hello", "world")
        assert rec.cost_usd > 0.0

    def test_total_tokens_accumulate(self):
        tracker = UsageTracker()
        tracker.record("gpt-4o-mini", "one two three", "four five")
        tracker.record("gpt-4o-mini", "six seven", "eight nine ten")
        summary = tracker.daily_summary()
        assert summary["total_prompt_tokens"] > 0
        assert summary["total_completion_tokens"] > 0
