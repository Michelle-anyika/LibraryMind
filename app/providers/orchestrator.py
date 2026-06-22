import logging

from app.providers.base import BaseAIProvider

logger = logging.getLogger(__name__)


class ResilientAIService:
    """
    Tries providers in order. If one raises after all its internal retries,
    the failure is logged and the next provider is attempted.
    Raises RuntimeError only when every provider has been exhausted.
    """

    def __init__(self, providers: list[BaseAIProvider]) -> None:
        if not providers:
            raise ValueError("ResilientAIService requires at least one provider.")
        self.providers = providers

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        last_exc: Exception | None = None

        for provider in self.providers:
            try:
                return await provider.generate(prompt, system, temperature, max_tokens)
            except Exception as exc:
                logger.error(
                    "Provider %s failed — %s. Failing over to next provider.",
                    type(provider).__name__,
                    exc,
                )
                last_exc = exc

        raise RuntimeError(
            f"All {len(self.providers)} AI provider(s) failed. Last error: {last_exc}"
        )
