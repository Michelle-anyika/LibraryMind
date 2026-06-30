import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    book_id: str
    document: str
    metadata: dict
    similarity: float  # cosine similarity in [0, 1]; higher = more relevant


class VectorStore:
    """
    ChromaDB wrapper configured for cosine similarity search.

    ChromaDB with hnsw:space=cosine returns *distances* (lower = more similar)
    where distance = 1 − cosine_similarity.  All public methods on this class
    convert to similarity scores so callers never need to know this detail.
    """

    def __init__(
        self,
        collection_name: str = "library_books",
        persist_directory: str = "./chroma_db",
    ) -> None:
        import chromadb  # noqa: PLC0415

        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore ready — collection=%r items=%d",
            collection_name,
            self._collection.count(),
        )

    # ── Write ──────────────────────────────────────────────────────────────────

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        documents: list[str],
    ) -> None:
        """Insert or update records.  Existing ids are overwritten."""
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        logger.info("Upserted %d documents into vector store.", len(ids))

    def clear(self) -> None:
        """Delete every document in the collection (used by seed.py to re-seed cleanly)."""
        existing = self._collection.get(include=[])
        ids = existing.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
            logger.info("Cleared %d documents from vector store.", len(ids))

    # ── Read ───────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> list[SearchResult]:
        """
        Return the top-*n_results* most similar documents as SearchResult objects
        with similarity scores already converted from ChromaDB distances.
        """
        count = self._collection.count()
        if count == 0:
            logger.warning("VectorStore is empty — run seed.py first.")
            return []

        safe_n = min(n_results, count)
        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=safe_n,
            include=["metadatas", "documents", "distances"],
        )

        results: list[SearchResult] = []
        for book_id, doc, metadata, distance in zip(
            raw["ids"][0],
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            similarity = 1.0 - distance  # cosine: distance = 1 - similarity
            results.append(
                SearchResult(
                    book_id=book_id,
                    document=doc,
                    metadata=metadata,
                    similarity=round(similarity, 4),
                )
            )

        return results

    def count(self) -> int:
        return self._collection.count()
