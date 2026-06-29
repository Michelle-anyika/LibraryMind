import logging

from app.infrastructure.cache import CacheService
from app.infrastructure.usage_tracker import UsageTracker
from app.infrastructure.vector_store import VectorStore
from app.providers.orchestrator import ResilientAIService
from app.services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

_REFUSAL = (
    "I'm sorry, I couldn't find any books in our catalogue that are relevant to "
    "your question. Please ask me about specific titles, authors, genres, or themes "
    "from our library collection and I'll do my best to help!"
)

_SYSTEM_PROMPT = """You are a knowledgeable and helpful library assistant for a public library.

Your answers MUST follow these rules without exception:
1. Base your response ONLY on the book information provided in the CONTEXT block below.
2. Do NOT invent, fabricate, or hallucinate any book titles, authors, plot details, or facts.
3. When you reference a book, cite it inline by title (e.g. "In *Dune* by Frank Herbert...").
4. If the provided context does not contain enough information to answer the question, say so honestly and suggest the patron ask a librarian for further help.
5. Keep your tone warm, enthusiastic about literature, and accessible to all ages.
6. Do NOT answer questions unrelated to the books in the context."""


def _build_context(results: list) -> str:
    lines = []
    for i, result in enumerate(results, start=1):
        m = result.metadata
        lines.append(
            f"[{i}] Title: {m.get('title')} | Author: {m.get('author')} "
            f"| Year: {m.get('year')} | Genre: {m.get('genre')}\n"
            f"    Description: {m.get('description')}"
        )
    return "\n\n".join(lines)


class RAGEngine:
    """
    Retrieval-Augmented Generation pipeline.

    Flow:
      1. Check full-response cache (identical question → instant reply).
      2. Embed the question via EmbeddingService (embedding cache guards this too).
      3. Query VectorStore for top-K candidates.
      4. Drop any result below the relevance threshold.
      5. If nothing survives, return a polite refusal — no AI call made.
      6. Build a context block + rigid system prompt and call the AI.
      7. Record token usage, cache the response, return.
    """

    def __init__(
        self,
        ai_service: ResilientAIService,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        cache: CacheService,
        usage_tracker: UsageTracker,
        relevance_threshold: float = 0.70,
        top_k: int = 5,
        primary_model: str = "gpt-4o-mini",
    ) -> None:
        self._ai = ai_service
        self._embeddings = embedding_service
        self._vector_store = vector_store
        self._cache = cache
        self._tracker = usage_tracker
        self._threshold = relevance_threshold
        self._top_k = top_k
        self._primary_model = primary_model

    async def answer(self, question: str) -> dict:
        """
        Return a dict with keys: answer (str), sources (list), cached (bool).
        Never raises — caller gets a refusal message on empty results.
        """
        cache_key = self._cache._generate_key(question, "rag")
        cached = await self._cache.get(cache_key)
        if cached is not None:
            logger.info("RAG cache HIT for question[:60]=%r", question[:60])
            cached["cached"] = True
            return cached

        # Embed the question
        query_vector = await self._embeddings.embed(question)

        # Retrieve top-K candidates
        candidates = self._vector_store.search(query_vector, n_results=self._top_k)

        # Filter by relevance threshold
        relevant = [r for r in candidates if r.similarity >= self._threshold]
        logger.info(
            "RAG: %d candidates retrieved, %d passed threshold %.2f",
            len(candidates),
            len(relevant),
            self._threshold,
        )

        if not relevant:
            logger.info("RAG: no relevant results — returning refusal.")
            return {"answer": _REFUSAL, "sources": [], "cached": False}

        # Build prompt
        context_block = _build_context(relevant)
        prompt = f"CONTEXT:\n{context_block}\n\nQUESTION: {question}"

        # Call AI
        answer_text = await self._ai.generate(
            prompt=prompt,
            system=_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=1024,
        )

        # Track usage
        self._tracker.record(self._primary_model, prompt, answer_text)

        sources = [
            {
                "title": r.metadata.get("title", ""),
                "author": r.metadata.get("author", ""),
                "score": r.similarity,
            }
            for r in relevant
        ]

        response = {"answer": answer_text, "sources": sources, "cached": False}

        await self._cache.set(cache_key, response)
        return response
