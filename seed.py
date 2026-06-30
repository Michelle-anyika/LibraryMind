"""
seed.py — populate ChromaDB with the library catalogue.

Usage (from project root, with venv active):
    python seed.py

The script reads data/books.json, generates an embedding for each book,
and upserts everything into the local ChromaDB collection.
Run this once before starting the server, and re-run whenever books.json changes.
"""
import asyncio
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    # Import after load_dotenv so Settings can read the .env file
    from app.config import get_settings
    from app.infrastructure.cache import CacheService
    from app.infrastructure.vector_store import VectorStore
    from app.services.embeddings import EmbeddingService

    settings = get_settings()

    if not settings.amalitech_api_key or not settings.openai_api_base:
        logger.error(
            "AMALITECH_API_KEY and OPENAI_API_BASE must be set in .env to run seed.py"
        )
        sys.exit(1)

    data_path = Path(__file__).parent / "data" / "books.json"
    if not data_path.exists():
        logger.error("data/books.json not found at %s", data_path)
        sys.exit(1)

    books = json.loads(data_path.read_text(encoding="utf-8"))
    logger.info("Loaded %d books from %s", len(books), data_path)

    cache = CacheService(host=settings.redis_host, port=settings.redis_port)
    embedding_svc = EmbeddingService(
        api_key=settings.amalitech_api_key,
        base_url=settings.openai_api_base,
        cache=cache,
    )
    vector_store = VectorStore()

    # Clear existing data so re-seeding is always idempotent
    vector_store.clear()

    # Build the embedding text for each book
    documents = [
        f"Title: {b['title']} | Author: {b['author']} | Genre: {b['genre']} | "
        f"Year: {b['year']} | Description: {b['description']}"
        for b in books
    ]

    logger.info("Generating embeddings for %d books (cache will skip duplicates)…", len(books))
    embeddings = await embedding_svc.embed_batch(documents)

    ids = [b["id"] for b in books]
    metadatas = [
        {
            "title": b["title"],
            "author": b["author"],
            "year": b["year"],
            "genre": b["genre"],
            "description": b["description"],
        }
        for b in books
    ]

    vector_store.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )

    logger.info(
        "Seed complete. Vector store now contains %d documents.",
        vector_store.count(),
    )


if __name__ == "__main__":
    asyncio.run(main())
