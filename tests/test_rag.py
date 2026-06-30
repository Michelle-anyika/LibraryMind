"""
Tests for tasks 3.1 and 3.2 — EmbeddingService, VectorStore, RAGEngine.

All external dependencies (OpenAI client, ChromaDB) are mocked so this suite
runs without network access or chromadb installed.

Run with:  pytest tests/test_rag.py -v
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.infrastructure.cache import CacheService
from app.infrastructure.vector_store import SearchResult, VectorStore
from app.infrastructure.usage_tracker import UsageTracker
from app.providers.orchestrator import ResilientAIService
from app.services.embeddings import EmbeddingService
from app.services.rag_engine import RAGEngine, _REFUSAL


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_fallback_cache() -> CacheService:
    svc = CacheService.__new__(CacheService)
    svc._fallback = {}
    svc._using_fallback = True
    svc._redis = None
    return svc


def _make_embedding_service(vector: list[float] | None = None) -> EmbeddingService:
    """EmbeddingService with a mocked OpenAI client."""
    cache = _make_fallback_cache()
    svc = EmbeddingService.__new__(EmbeddingService)
    svc._cache = cache
    svc._model = "text-embedding-3-small"

    mock_client = AsyncMock()
    embedding_data = MagicMock()
    embedding_data.embedding = vector or [0.1, 0.2, 0.3]
    mock_client.embeddings.create = AsyncMock(
        return_value=MagicMock(data=[embedding_data])
    )
    svc._client = mock_client
    return svc


def _make_vector_store(results: list[SearchResult] | None = None) -> VectorStore:
    """VectorStore with mocked ChromaDB collection."""
    store = VectorStore.__new__(VectorStore)
    store._collection = MagicMock()
    store._collection.count.return_value = 22
    store._search_results = results or []

    def fake_search(query_embedding, n_results):
        return store._search_results

    store.search = fake_search  # type: ignore[method-assign]
    return store


def _sample_result(similarity: float = 0.85) -> SearchResult:
    return SearchResult(
        book_id="book-001",
        document="Title: Dune | Author: Frank Herbert",
        metadata={
            "title": "Dune",
            "author": "Frank Herbert",
            "year": 1965,
            "genre": "sci-fi",
            "description": "Desert planet adventure.",
        },
        similarity=similarity,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Task 3.1 — EmbeddingService
# ══════════════════════════════════════════════════════════════════════════════

class TestEmbeddingService:
    async def test_embed_returns_vector(self):
        svc = _make_embedding_service(vector=[0.5, 0.6, 0.7])
        vector = await svc.embed("space adventure")
        assert vector == [0.5, 0.6, 0.7]

    async def test_embed_caches_result(self):
        svc = _make_embedding_service()
        await svc.embed("hello world")
        await svc.embed("hello world")
        # API should only be called once; second call served from cache
        assert svc._client.embeddings.create.call_count == 1

    async def test_embed_different_texts_call_api_each_time(self):
        svc = _make_embedding_service()
        await svc.embed("text one")
        await svc.embed("text two")
        assert svc._client.embeddings.create.call_count == 2

    async def test_embed_batch_returns_correct_count(self):
        svc = _make_embedding_service(vector=[1.0, 2.0, 3.0])
        # Mock batch response with two embeddings
        e1 = MagicMock()
        e1.embedding = [1.0, 2.0, 3.0]
        e2 = MagicMock()
        e2.embedding = [4.0, 5.0, 6.0]
        svc._client.embeddings.create = AsyncMock(
            return_value=MagicMock(data=[e1, e2])
        )
        results = await svc.embed_batch(["first", "second"])
        assert len(results) == 2

    async def test_embed_batch_empty_list_returns_empty(self):
        svc = _make_embedding_service()
        assert await svc.embed_batch([]) == []

    async def test_embed_batch_uses_cache_for_previously_seen_text(self):
        svc = _make_embedding_service(vector=[9.0, 8.0])
        # Pre-warm cache for one text
        await svc.embed("cached text")
        svc._client.embeddings.create.reset_mock()

        e1 = MagicMock()
        e1.embedding = [1.0, 2.0]
        svc._client.embeddings.create = AsyncMock(
            return_value=MagicMock(data=[e1])
        )
        results = await svc.embed_batch(["cached text", "new text"])
        assert len(results) == 2
        # Only "new text" should hit the API
        assert svc._client.embeddings.create.call_count == 1


# ══════════════════════════════════════════════════════════════════════════════
# Task 3.1 — VectorStore (unit, no real ChromaDB)
# ══════════════════════════════════════════════════════════════════════════════

class TestVectorStore:
    def _make_store_with_raw_query(self, raw_results: dict) -> VectorStore:
        """VectorStore whose _collection.query returns controlled raw output."""
        store = VectorStore.__new__(VectorStore)
        mock_collection = MagicMock()
        mock_collection.count.return_value = len(raw_results.get("ids", [[]])[0])
        mock_collection.query.return_value = raw_results
        store._collection = mock_collection
        return store

    def test_search_converts_distance_to_similarity(self):
        raw = {
            "ids": [["book-001"]],
            "documents": [["some text"]],
            "metadatas": [[{"title": "Dune"}]],
            "distances": [[0.15]],  # distance 0.15 → similarity 0.85
        }
        store = self._make_store_with_raw_query(raw)
        results = store.search([0.1, 0.2, 0.3])
        assert len(results) == 1
        assert abs(results[0].similarity - 0.85) < 1e-4

    def test_search_returns_empty_on_empty_collection(self):
        store = VectorStore.__new__(VectorStore)
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        store._collection = mock_collection
        results = store.search([0.1, 0.2])
        assert results == []
        mock_collection.query.assert_not_called()

    def test_search_caps_n_results_to_collection_size(self):
        raw = {
            "ids": [["b1", "b2"]],
            "documents": [["d1", "d2"]],
            "metadatas": [[{"title": "A"}, {"title": "B"}]],
            "distances": [[0.1, 0.2]],
        }
        store = self._make_store_with_raw_query(raw)
        store._collection.count.return_value = 2
        store.search([0.0], n_results=10)
        call_kwargs = store._collection.query.call_args.kwargs
        assert call_kwargs["n_results"] == 2  # capped at collection size

    def test_upsert_calls_collection(self):
        store = VectorStore.__new__(VectorStore)
        store._collection = MagicMock()
        store.upsert(["id1"], [[0.1, 0.2]], [{"title": "T"}], ["doc"])
        store._collection.upsert.assert_called_once()

    def test_clear_deletes_existing_ids(self):
        store = VectorStore.__new__(VectorStore)
        store._collection = MagicMock()
        store._collection.get.return_value = {"ids": ["id1", "id2"]}
        store.clear()
        store._collection.delete.assert_called_once_with(ids=["id1", "id2"])


# ══════════════════════════════════════════════════════════════════════════════
# Task 3.2 — RAGEngine
# ══════════════════════════════════════════════════════════════════════════════

def _make_rag_engine(
    search_results: list[SearchResult] | None = None,
    ai_response: str = "Here is your answer.",
    threshold: float = 0.70,
) -> RAGEngine:
    cache = _make_fallback_cache()
    embedding_svc = _make_embedding_service()
    vector_store = _make_vector_store(search_results)
    tracker = UsageTracker()

    mock_ai = AsyncMock(spec=ResilientAIService)
    mock_ai.generate = AsyncMock(return_value=ai_response)

    return RAGEngine(
        ai_service=mock_ai,
        embedding_service=embedding_svc,
        vector_store=vector_store,
        cache=cache,
        usage_tracker=tracker,
        relevance_threshold=threshold,
    )


class TestRAGEngine:
    async def test_returns_answer_with_sources_for_relevant_results(self):
        engine = _make_rag_engine(
            search_results=[_sample_result(similarity=0.90)],
            ai_response="Dune is a great sci-fi novel about a desert planet.",
        )
        result = await engine.answer("desert planet adventure")
        assert result["answer"] == "Dune is a great sci-fi novel about a desert planet."
        assert len(result["sources"]) == 1
        assert result["sources"][0]["title"] == "Dune"
        assert result["cached"] is False

    async def test_returns_refusal_when_no_results_pass_threshold(self):
        engine = _make_rag_engine(
            search_results=[_sample_result(similarity=0.50)]  # below 0.70
        )
        result = await engine.answer("What is the meaning of life?")
        assert result["answer"] == _REFUSAL
        assert result["sources"] == []
        engine._ai.generate.assert_not_called()

    async def test_returns_refusal_when_search_empty(self):
        engine = _make_rag_engine(search_results=[])
        result = await engine.answer("random off-topic question")
        assert result["answer"] == _REFUSAL
        engine._ai.generate.assert_not_called()

    async def test_second_identical_question_is_cached(self):
        engine = _make_rag_engine(
            search_results=[_sample_result(0.85)],
            ai_response="Cached answer.",
        )
        r1 = await engine.answer("recommend a sci-fi book")
        r2 = await engine.answer("recommend a sci-fi book")
        assert r1["cached"] is False
        assert r2["cached"] is True
        assert engine._ai.generate.call_count == 1  # AI only called once

    async def test_sources_include_title_author_and_score(self):
        engine = _make_rag_engine(search_results=[_sample_result(0.88)])
        result = await engine.answer("space adventure book")
        source = result["sources"][0]
        assert "title" in source
        assert "author" in source
        assert "score" in source

    async def test_filters_low_scoring_results_from_sources(self):
        results = [
            _sample_result(0.92),
            SearchResult(
                book_id="book-002",
                document="Foundation...",
                metadata={"title": "Foundation", "author": "Asimov",
                          "year": 1951, "genre": "sci-fi", "description": ""},
                similarity=0.45,  # below threshold
            ),
        ]
        engine = _make_rag_engine(search_results=results)
        result = await engine.answer("space empire")
        assert len(result["sources"]) == 1
        assert result["sources"][0]["title"] == "Dune"

    async def test_usage_tracker_records_call(self):
        engine = _make_rag_engine(search_results=[_sample_result(0.85)])
        await engine.answer("recommend something")
        assert engine._tracker.daily_summary()["request_count"] == 1

    async def test_usage_tracker_not_called_on_refusal(self):
        engine = _make_rag_engine(search_results=[_sample_result(0.30)])
        await engine.answer("off-topic")
        assert engine._tracker.daily_summary()["request_count"] == 0

    async def test_usage_tracker_not_called_on_cache_hit(self):
        engine = _make_rag_engine(search_results=[_sample_result(0.85)])
        await engine.answer("same question")
        await engine.answer("same question")
        assert engine._tracker.daily_summary()["request_count"] == 1
