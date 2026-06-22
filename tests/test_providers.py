"""
Tests for app/providers/ — Task 1.2 acceptance criteria.

Run with:  pytest tests/test_providers.py -v
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import openai

from app.providers.base import BaseAIProvider, _retry_with_backoff, _MAX_RETRIES
from app.providers.openai_provider import OpenAIProvider
from app.providers.claude_provider import ClaudeProvider
from app.providers.gemini_provider import GeminiProvider
from app.providers.orchestrator import ResilientAIService


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_openai_response(text: str):
    resp = MagicMock()
    resp.choices[0].message.content = text
    return resp


def _rate_limit_error():
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    return openai.RateLimitError("rate limited", response=mock_resp, body={})


def _service_unavailable_error():
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    return openai.APIStatusError("service unavailable", response=mock_resp, body={})


def _auth_error():
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    return openai.APIStatusError("unauthorized", response=mock_resp, body={})


# ── _retry_with_backoff ────────────────────────────────────────────────────────

async def test_retry_returns_immediately_on_success():
    coro = AsyncMock(return_value="hello")
    with patch("asyncio.sleep") as mock_sleep:
        result = await _retry_with_backoff(lambda: coro())
    assert result == "hello"
    coro.assert_called_once()
    mock_sleep.assert_not_called()


async def test_retry_retries_on_429_then_succeeds():
    exc = _rate_limit_error()
    coro = AsyncMock(side_effect=[exc, "recovered"])
    with patch("asyncio.sleep") as mock_sleep:
        result = await _retry_with_backoff(lambda: coro())
    assert result == "recovered"
    assert coro.call_count == 2
    mock_sleep.assert_called_once()


async def test_retry_retries_on_503_then_succeeds():
    exc = _service_unavailable_error()
    coro = AsyncMock(side_effect=[exc, "back up"])
    with patch("asyncio.sleep"):
        result = await _retry_with_backoff(lambda: coro())
    assert result == "back up"


async def test_retry_raises_after_all_retries_exhausted():
    exc = _rate_limit_error()
    coro = AsyncMock(side_effect=exc)
    with patch("asyncio.sleep") as mock_sleep:
        with pytest.raises(openai.RateLimitError):
            await _retry_with_backoff(lambda: coro())
    assert coro.call_count == _MAX_RETRIES + 1
    assert mock_sleep.call_count == _MAX_RETRIES


async def test_retry_does_not_retry_on_non_transient_error():
    exc = _auth_error()
    coro = AsyncMock(side_effect=exc)
    with patch("asyncio.sleep") as mock_sleep:
        with pytest.raises(openai.APIStatusError):
            await _retry_with_backoff(lambda: coro())
    coro.assert_called_once()
    mock_sleep.assert_not_called()


# ── Provider default models ───────────────────────────────────────────────────

def test_openai_provider_default_model():
    p = OpenAIProvider(api_key="k", base_url="http://fake")
    assert p.model == "gpt-4o-mini"


def test_claude_provider_default_model():
    p = ClaudeProvider(api_key="k", base_url="http://fake")
    assert p.model == "claude-3-5-haiku-20241022"


def test_gemini_provider_default_model():
    p = GeminiProvider(api_key="k", base_url="http://fake")
    assert p.model == "gemini-1.5-flash"


def test_provider_accepts_custom_model():
    p = OpenAIProvider(api_key="k", base_url="http://fake", model="gpt-4o")
    assert p.model == "gpt-4o"


# ── OpenAIProvider.generate() ─────────────────────────────────────────────────

async def test_provider_returns_response_text():
    p = OpenAIProvider(api_key="k", base_url="http://fake")
    p._client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response("The answer is 42.")
    )
    result = await p.generate("What is the answer?")
    assert result == "The answer is 42."


async def test_provider_includes_system_message_when_given():
    p = OpenAIProvider(api_key="k", base_url="http://fake")
    p._client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response("I am a librarian.")
    )
    await p.generate("Who are you?", system="You are a librarian.")
    kwargs = p._client.chat.completions.create.call_args.kwargs
    messages = kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "You are a librarian."}
    assert messages[1] == {"role": "user", "content": "Who are you?"}


async def test_provider_omits_system_message_when_empty():
    p = OpenAIProvider(api_key="k", base_url="http://fake")
    p._client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response("ok")
    )
    await p.generate("Hello", system="")
    kwargs = p._client.chat.completions.create.call_args.kwargs
    assert len(kwargs["messages"]) == 1
    assert kwargs["messages"][0]["role"] == "user"


async def test_provider_passes_temperature_and_max_tokens():
    p = OpenAIProvider(api_key="k", base_url="http://fake")
    p._client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response("ok")
    )
    await p.generate("Hello", temperature=0.2, max_tokens=512)
    kwargs = p._client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 512


# ── ResilientAIService ────────────────────────────────────────────────────────

class StubProvider(BaseAIProvider):
    """Minimal concrete provider for orchestrator tests."""

    def __init__(self, response: str | None = None, raises: Exception | None = None):
        self.response = response
        self.raises = raises
        self.call_count = 0

    async def generate(self, prompt, system="", temperature=0.7, max_tokens=1024):
        self.call_count += 1
        if self.raises:
            raise self.raises
        return self.response


def test_resilient_raises_when_created_with_no_providers():
    with pytest.raises(ValueError):
        ResilientAIService([])


async def test_resilient_returns_from_primary_provider():
    p1 = StubProvider(response="from primary")
    p2 = StubProvider(response="from fallback")
    result = await ResilientAIService([p1, p2]).generate("Hi")
    assert result == "from primary"
    assert p1.call_count == 1
    assert p2.call_count == 0


async def test_resilient_falls_back_when_primary_fails():
    p1 = StubProvider(raises=RuntimeError("primary down"))
    p2 = StubProvider(response="fallback answer")
    result = await ResilientAIService([p1, p2]).generate("Hi")
    assert result == "fallback answer"
    assert p1.call_count == 1
    assert p2.call_count == 1


async def test_resilient_raises_runtime_error_when_all_fail():
    p1 = StubProvider(raises=RuntimeError("p1 down"))
    p2 = StubProvider(raises=RuntimeError("p2 down"))
    with pytest.raises(RuntimeError, match="All 2 AI provider"):
        await ResilientAIService([p1, p2]).generate("Hi")


async def test_resilient_skips_failed_providers_in_order():
    p1 = StubProvider(raises=RuntimeError("fail"))
    p2 = StubProvider(raises=RuntimeError("fail"))
    p3 = StubProvider(response="third works")
    result = await ResilientAIService([p1, p2, p3]).generate("Hi")
    assert result == "third works"
    assert p1.call_count == p2.call_count == p3.call_count == 1
