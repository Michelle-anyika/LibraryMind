import logging

import openai

from app.infrastructure.cache import CacheService

logger = logging.getLogger(__name__)

_EMBEDDING_TTL = 86_400  # 24 hours — embeddings are deterministic, cache them long


class EmbeddingService:
    """
    Generates vector embeddings via the OpenAI-compatible gateway.
    Every unique text is cached by SHA-256 hash so it is never embedded twice.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        cache: CacheService,
        model: str = "text-embedding-3-small",
    ) -> None:
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._cache = cache
        self._model = model

    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for *text*, served from cache when possible."""
        cache_key = self._cache._generate_key(text, "embedding")
        cached = await self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Embedding cache HIT for text[:40]=%r", text[:40])
            return cached

        response = await self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        vector: list[float] = response.data[0].embedding

        await self._cache.set(cache_key, vector, ttl=_EMBEDDING_TTL)
        logger.debug(
            "Embedded %d chars → %d-dim vector (model=%s)",
            len(text),
            len(vector),
            self._model,
        )
        return vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Return embeddings for a list of texts.
        Cache hits are served immediately; remaining texts are embedded in a
        single batched API call to minimise round-trips.
        """
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            key = self._cache._generate_key(text, "embedding")
            cached = await self._cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            response = await self._client.embeddings.create(
                model=self._model,
                input=uncached_texts,
            )
            for pos, embedding_obj in enumerate(response.data):
                idx = uncached_indices[pos]
                vector = embedding_obj.embedding
                results[idx] = vector
                key = self._cache._generate_key(uncached_texts[pos], "embedding")
                await self._cache.set(key, vector, ttl=_EMBEDDING_TTL)

            logger.debug(
                "Batch embedded %d texts (%d cache hits)",
                len(texts),
                len(texts) - len(uncached_texts),
            )

        return results  # type: ignore[return-value]
