import asyncio
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any

import openai

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds; doubles each attempt: 1s → 2s → 4s


async def _retry_with_backoff(
    coro_fn: Callable[[], Coroutine[Any, Any, str]],
    *,
    provider_name: str = "",
) -> str:
    """
    Call coro_fn() up to _MAX_RETRIES + 1 times.
    Retries only on 429 (RateLimitError) and 503 (APIStatusError).
    Any other exception propagates immediately.
    """
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            return await coro_fn()
        except (openai.RateLimitError, openai.APIStatusError) as exc:
            is_transient = isinstance(exc, openai.RateLimitError) or (
                isinstance(exc, openai.APIStatusError) and exc.status_code in (429, 503)
            )
            if not is_transient or attempt == _MAX_RETRIES:
                raise

            last_exc = exc
            delay = _BASE_DELAY * (2**attempt) + random.uniform(0.1, 0.5)
            logger.warning(
                "[%s] transient error (attempt %d/%d) — retrying in %.1fs",
                provider_name,
                attempt + 1,
                _MAX_RETRIES,
                delay,
            )
            await asyncio.sleep(delay)

    raise last_exc  # type: ignore[misc]


class BaseAIProvider(ABC):
    """Abstract interface every AI provider must implement."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Return a text response from the model."""


class _AmaliCompatibleProvider(BaseAIProvider):
    """
    Shared implementation for all AmaliTech-proxied providers.
    All vendors are reached through the same /api/v2/public/** gateway.
    The Provider header tells the proxy which upstream LLM to route to.
    """

    _default_model: str   # set by each concrete subclass
    _provider_header: str = "openai"  # overridden by ClaudeProvider

    def __init__(self, api_key: str, base_url: str, model: str | None = None) -> None:
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "X-Api-Key": api_key,
                "Provider": self._provider_header,
            },
        )
        self.model = model or self._default_model

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        return await _retry_with_backoff(
            lambda: self._call(prompt, system, temperature, max_tokens),
            provider_name=type(self).__name__,
        )

    async def _call(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
