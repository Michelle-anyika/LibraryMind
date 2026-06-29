import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Prices per 1 000 000 tokens (input_usd, output_usd)
_COST_TABLE: dict[str, tuple[float, float]] = {
    "gpt-4o":                      (5.00,  15.00),
    "gpt-4o-mini":                 (0.15,   0.60),
    "claude-3-5-haiku-20241022":   (0.80,   4.00),
    "claude-3-5-sonnet-20241022":  (3.00,  15.00),
    "claude-3-opus-20240229":      (15.00, 75.00),
    "gemini-1.5-flash":            (0.075,  0.30),
    "gemini-1.5-pro":              (1.25,   5.00),
    # catch-all for unknown models
    "default":                     (1.00,   3.00),
}


@dataclass
class UsageRecord:
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class UsageTracker:
    """
    Counts prompt and completion tokens for every AI call, estimates cost in
    USD from a per-model pricing table, and exposes an aggregate daily summary
    used by the /health endpoint.

    tiktoken is used when available; falls back to a naive whitespace-split
    count so the tracker never blocks the application from starting.
    """

    def __init__(self) -> None:
        self._records: list[UsageRecord] = []
        self._encoder = None

        try:
            import tiktoken  # noqa: PLC0415

            self._encoder = tiktoken.get_encoding("cl100k_base")
            logger.debug("UsageTracker: using tiktoken for token counting.")
        except Exception as exc:
            logger.warning(
                "tiktoken unavailable — falling back to word-count approximation: %s",
                exc,
            )

    # ── Internals ──────────────────────────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        return max(1, len(text.split()))

    def _estimate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        input_rate, output_rate = _COST_TABLE.get(model, _COST_TABLE["default"])
        return (prompt_tokens / 1_000_000) * input_rate + (
            completion_tokens / 1_000_000
        ) * output_rate

    # ── Public API ─────────────────────────────────────────────────────────────

    def record(self, model: str, prompt: str, completion: str) -> UsageRecord:
        """Count tokens, estimate cost, store and return a UsageRecord."""
        prompt_tokens = self._count_tokens(prompt)
        completion_tokens = self._count_tokens(completion)
        cost = self._estimate_cost(model, prompt_tokens, completion_tokens)

        rec = UsageRecord(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
        )
        self._records.append(rec)
        logger.info(
            "Usage — model=%s prompt_tokens=%d completion_tokens=%d cost=$%.6f",
            model,
            prompt_tokens,
            completion_tokens,
            cost,
        )
        return rec

    def daily_summary(self) -> dict:
        """Return aggregate totals for the /health endpoint."""
        return {
            "request_count": len(self._records),
            "total_cost_usd": round(sum(r.cost_usd for r in self._records), 6),
            "total_prompt_tokens": sum(r.prompt_tokens for r in self._records),
            "total_completion_tokens": sum(r.completion_tokens for r in self._records),
        }
