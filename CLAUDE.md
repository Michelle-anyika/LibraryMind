# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LibraryMind is an AI-powered backend service for public libraries, built with **FastAPI + Python 3.10+**. It provides semantic book search, RAG-powered Q&A, multi-turn chat, ticket classification, and review summarization.

**Current status:** The repository contains only documentation (`README.md`, `Library-Mind.md`). The implementation has not yet been started.

## Commands

```bash
# Setup
python3 -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt

# Seed vector database (must run before server)
python seed.py

# Development server
uvicorn app.main:app --reload

# Smoke tests
python test_smoke.py
```

Swagger UI is accessible at `http://127.0.0.1:8000/docs` once the server is running.

## Environment Configuration

Create a `.env` file in the project root:

```env
PRIMARY_PROVIDER=openai
AMALITECH_API_KEY=your_production_api_key_here
OPENAI_API_BASE=API_link_here
REDIS_HOST=localhost
REDIS_PORT=6379
RATE_LIMIT_PER_MINUTE=60
RELEVANCE_THRESHOLD=0.70
```

Redis is optional — the app degrades gracefully to in-memory passthrough when unavailable. All AI providers (OpenAI, Claude, Gemini) are accessed through the single AmaliAI gateway using `AMALITECH_API_KEY`.

## Architecture

Four-tier layered structure:

```
app/
├── main.py              # FastAPI app entry, CORS middleware
├── config.py            # Env var loading and validation
├── routers/             # HTTP route handlers (no business logic)
│   ├── search.py        # /search/books, /search/ask
│   ├── chat.py          # /chat
│   ├── classify.py      # /classify/ticket
│   └── summarize.py     # /summarise/reviews
├── services/            # Business logic
│   ├── rag_engine.py    # Embed → search → filter → prompt → generate → cache
│   ├── chat_service.py  # Conversation store, history truncation, RAG integration
│   ├── classifier.py    # Low-temp generation, JSON fence stripping
│   ├── summarizer.py    # Batch review analysis, structured JSON output
│   └── embeddings.py    # Vector generation with caching
├── providers/           # AI vendor abstraction
│   ├── base.py          # Abstract interface: generate(prompt, system, temperature, max_tokens)
│   ├── openai_provider.py
│   ├── claude_provider.py
│   ├── gemini_provider.py
│   └── orchestrator.py  # ResilientAIService: ordered fallback with exponential backoff
├── infrastructure/
│   ├── cache.py         # Redis wrapper with SHA-256 key hashing, graceful degradation
│   ├── rate_limiter.py  # Thread-safe token-bucket algorithm
│   ├── usage_tracker.py # Token counting (tiktoken), per-model cost tables, daily total
│   └── vector_store.py  # ChromaDB wrapper, cosine similarity, upsert + search
└── models/
    └── schemas.py       # Pydantic request/response models

seed.py                  # Standalone script: reads data/books.json → embed → upsert to ChromaDB
data/books.json          # 20+ books, 5+ genres; each with id, title, author, year, genre, description
```

## API Endpoints

| Method | Path | Input | Output |
|--------|------|-------|--------|
| POST | `/search/books` | `query`, `limit` | Books with similarity scores |
| POST | `/search/ask` | `question` | Answer, source books, `cached` flag |
| POST | `/chat` | `conversation_id`, `message` | Reply, sources |
| POST | `/classify/ticket` | `ticket_text` | `{category, priority, sentiment, routing, summary}` |
| POST | `/summarise/reviews` | `reviews[]` | `{sentiment, rating, themes, praise, criticism, recommendation}` |
| GET | `/health` | — | `{status, daily_cost, request_count}` |

Error codes: `HTTP 422` for validation failures, `HTTP 429` for rate limit exceeded, `HTTP 503` for AI provider failures.

## Key Design Rules

**AI provider layer:** `ResilientAIService` tries providers in order (`PRIMARY_PROVIDER` first), retries each with exponential backoff before moving to the next. Raises `RuntimeError` only when all providers fail.

**Caching:** Two tiers — embedding vectors and full RAG responses — both keyed by SHA-256 hash of inputs. Cache hits must be logged and reflected in the `/health` cost (cached calls should not add to cost).

**JSON output parsing:** Classification and summarization responses must strip markdown code fences (` ```json ... ``` `) before calling `json.loads()`. This is the most common failure point.

**Conversation memory:** Truncate history to the most recent N messages using `tiktoken` token counts per turn — never a fixed message count — to stay within model context windows.

**Relevance filtering:** RAG results below `RELEVANCE_THRESHOLD` (default `0.70`) are discarded. When no results pass, return a polite refusal instead of generating an answer. Know whether ChromaDB returns distance (lower = better) or similarity (higher = better) when comparing against the threshold.

**Embedding model consistency:** If the embedding model changes, delete and re-seed the ChromaDB collection. Mismatched dimensions will silently return wrong results.

## Acceptance Test Scenarios

These are the 10 validation scenarios from the spec:

1. Search "desert planet adventure" → relevant sci-fi books with high scores
2. Ask "What is the meaning of life?" → polite refusal (off-topic)
3. Ask "Recommend a classic romance novel" → grounded answer with citations
4. Chat turn 1: "Recommend a thriller" → specific book from catalogue
5. Chat turn 2: "Tell me more about that" → elaborates on same book (memory works)
6. Classify angry card complaint → `category=technical, priority=high, sentiment=negative`
7. Summarize 3–5 mixed reviews → balanced JSON with praise and criticism
8. Same question asked twice → second call faster (cache hit)
9. Exceed rate limit → HTTP 429
10. Disable primary provider key → automatic fallback to secondary
