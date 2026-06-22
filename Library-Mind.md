# **Library Mind - intelligent library assistant**


**Module:** 10 | **Type:** Capstone lab | **Framework:** FastAPI + Python

---

## **Project overview**

LibraryMind is an AI-powered backend service for a public library. The system allows patrons to:

- Search the library catalogue using natural language instead of exact keyword matches

- Get intelligent book recommendations based on stated interests and preferences

- Ask detailed questions about books and receive grounded answers with source citations

- Summarize collections of book reviews and extract key themes automatically

- Chat with an AI librarian that remembers conversation context across multiple turns

- Submit support tickets that are automatically classified by category, priority, and sentiment

This capstone exercises every core skill from the training programme: multi-provider AI integration, vector-based semantic search, retrieval-augmented generation, prompt engineering for structured outputs, conversational memory management, and production concerns like caching, rate limiting, and cost tracking.

### **What you will build**

| **Component** | **Purpose** |
| --- | --- |
| Multi-provider AI layer | Abstraction over OpenAI and Claude with automatic fallback when one fails |
| Knowledge base | Vector database storing book descriptions, searchable by meaning |
| RAG engine | Retrieval-augmented question answering grounded in the catalogue |
| AI librarian chatbot | Multi-turn conversational agent with persistent memory |
| Classification service | Auto-categorize and route member support tickets |
| Summarization service | Condense book reviews into themes, sentiment, and recommendations |
| REST API | FastAPI application exposing all functionality through documented endpoints |
| Caching and rate limiting | Response caching and request throttling to protect budget and performance |
| Usage tracker | Token counting and cost estimation for every AI call |

---

## **Project outcomes**

By completing this lab you will demonstrate the ability to:

- Integrate multiple AI providers into a single resilient service with automatic failover

- Design and populate a vector database for semantic search over a custom dataset

- Implement a complete RAG pipeline from document chunking through answer generation with source citations

- Craft effective prompts that produce structured, parsable outputs for classification and summarization

- Build a multi-turn chatbot with conversation memory and context-window management

- Apply production patterns: caching, rate limiting, error handling, and cost observability

- Expose AI capabilities through a clean, well-documented REST API

- Measure and optimize the cost and performance of AI API calls

---

## **System architecture**

LibraryMind follows a layered architecture that separates concerns and makes each component independently testable.

**Layer 1: API layer**

The topmost layer is a FastAPI application that receives HTTP requests and returns JSON responses. It defines separate routers for each domain: search, chat, recommendations, summarization, and classification. This layer handles request validation via Pydantic models, HTTP error responses, and routing to the appropriate service. It contains no business logic itself.

**Layer 2: Service layer**

Contains all business logic across five core services:

| **Service** | **Responsibility** |
| --- | --- |
| RAG engine | Orchestrates semantic retrieval, constructs context-enriched prompts, sends to AI provider, returns answers with citations |
| Chatbot service | Manages multi-turn conversations, maintains conversation store, retrieves RAG context per message, builds prompts with history |
| Classification service | Takes free-text support tickets and returns structured JSON with category, priority, sentiment, and routing |
| Summarization service | Accepts a list of book reviews and produces structured analysis of themes, praise, and criticism |
| Embedding service | Generates vector representations of text with a caching layer to avoid redundant API calls |

**Layer 3: AI provider layer**

Abstracts over multiple LLM providers (OpenAI, Anthropic Claude, Google Gemini) behind a single common interface. A resilient orchestrator tries the primary provider first and automatically falls through to the next if it fails. Each provider includes retry logic with exponential backoff before triggering a fallback, ensuring the application never depends on a single vendor.

**Layer 4: Infrastructure layer**

| **Service** | **Details** |
| --- | --- |
| Vector database (ChromaDB) | Stores book embeddings, supports cosine-similarity search |
| Cache (Redis) | Stores AI responses, embeddings, and RAG results using hash-based keys with configurable TTL; degrades gracefully if Redis is unavailable |
| Rate limiter | Token-bucket algorithm controlling AI API call frequency to protect budget |
| Usage tracker | Counts tokens per AI call, estimates cost via per-model pricing tables, exposes daily spend via health endpoint |

**Data flow: answering a patron's question**

When a patron asks "Recommend a book about space exploration":

- API layer validates the request and passes it to the RAG engine

- RAG engine checks the cache — if a cached answer exists, it returns immediately

- If not cached, the rate limiter is consulted — returns HTTP 429 if limit is exceeded

- Embedding service generates a vector for the question (checking the embedding cache first)

- The vector queries ChromaDB, which returns the top-K most semantically similar books

- Results below the configured relevance threshold are discarded

- Remaining books are formatted into a context block and combined with the question into a prompt

- The prompt is sent through the AI provider layer with automatic fallback

- Usage tracker records token counts and estimated cost

- The answer and source list are cached for future identical queries

- The API layer serializes the response as JSON and returns it to the client

---

## **Part 0: Environment setup**

Set up your development environment with all necessary tools and dependencies.

**Requirements:**

- Create a Python 3.10+ virtual environment

- Install official SDKs for at least two AI providers (three is ideal)

- Install a vector database client for Python

- Install supporting libraries for: web framework, environment variables, caching, retry logic, token counting, and text splitting

- Create a .env file with all required configuration variables

- Create a .gitignore excluding secrets, caches, and virtual environments

- Optionally set up Redis (the system should work without it)

**Acceptance criteria:**

- Virtual environment created and activated

- All required packages install without errors

- Config module loads environment variables and validates at least one provider key is set

- .gitignore prevents .env, venv/, and __pycache__/ from being committed

---

## **Part 1: Multi-provider AI layer**

Build an AI provider abstraction layer wrapping OpenAI, Claude, and Google Gemini behind a single interface, with automatic fallback.

**Requirements:**

- Define a common interface (abstract base class or protocol) with at least a generate(prompt, system, temperature, max_tokens) method

- Implement concrete provider classes for OpenAI and Claude using their official Python SDKs or AmaliAI API

- Each provider must include retry logic with exponential backoff for transient errors (rate limits, timeouts)

- Build a ResilientAIService class that maintains an ordered list of providers and attempts generation in sequence

- Primary provider must be configurable via the PRIMARY_PROVIDER environment variable

- If all providers fail, raise a clear, descriptive error

**Acceptance criteria:**

- Calling generate() returns a text response from the primary provider

- Temporarily invalidating the primary provider's API key causes automatic fallback without crashing

- Retry logic is observable in logs before fallback occurs

- If all providers are down, a RuntimeError is raised with a helpful message

---

## **Part 2: Core infrastructure**

Implement three shared infrastructure components used by all services.

**Cache:**

- Wrap Redis behind a helper class with get/set methods

- Support JSON serialization for complex objects

- Generate deterministic cache keys using hashed inputs

- Degrade gracefully when Redis is unavailable

- Support configurable TTL per entry

**Rate limiter:**

- Implement a token-bucket algorithm that refills at a configurable rate

- Must be thread-safe

- Expose a method that acquires a token or raises an exception

- Requests-per-minute limit must come from environment configuration

**Usage tracker:**

- Count prompt and completion tokens for every AI call using a tokenizer library

- Estimate cost in USD based on per-model pricing tables

- Store records in memory (a simple list is sufficient)

- Expose a method to query the total daily cost

**Acceptance criteria:**

- The same prompt called twice results in a cache hit on the second call, observable in logs

- Exceeding the rate limit within one minute triggers a rate limit error

- Usage tracker reports non-zero cost after at least one AI call

- Application starts and runs correctly even when Redis is not running

---

## **Part 3: Knowledge base — embeddings and vector store**

Create a sample library catalogue, generate embeddings, store them in a vector database, and build a semantic search method.

**Sample data:**

- Create a JSON file containing at least 20 books

- Each book must include: id, title, author, year, genre, and a rich multi-sentence description

- Books must span at least 5 different genres for realistic search testing

**Embedding service:**

- Generate embeddings using an AI provider's embedding model

- Support single-text and batch embedding methods

- Cache embeddings to avoid regenerating identical vectors

**Vector store:**

- Use ChromaDB (recommended) with cosine similarity as the distance metric

- Support upsert operations for adding books

- Support semantic search: given a query embedding, return top-K most similar books with similarity scores and metadata

**Seed script:**

- Reads books from the JSON file

- Generates embeddings combining title, author, and description into the embedded text

- Upserts all books into the vector store

- Can be run independently from the command line

**Acceptance criteria:**

- Running the seed script populates the vector database with all books

- Searching "space travel adventure" returns sci-fi books even if those words don't appear in descriptions

- Searching "historical romance in England" returns relevant classic novels

- Each search result includes book metadata and a similarity score

---

### **8. Part 4: RAG engine**

Build a complete Retrieval-Augmented Generation pipeline that answers patron questions using relevant books as context.

**Requirements:**

- Accept a natural language question as input

- Generate an embedding for the question and search the vector store

- Filter results by a configurable relevance threshold — discard low-quality matches

- If no results pass the threshold, return a polite message and do not hallucinate

- Format relevant book records into a structured context block

- Construct a prompt instructing the AI to answer only using the provided context and cite referenced books

- Send the prompt through the multi-provider AI layer

- Return a structured response containing: answer text, source books (title, author, relevance score), and cache status

- Cache responses so identical questions return instantly

- Track token usage and cost for every generation call

**Acceptance criteria:**

- Asking "What science fiction books do you have about desert planets?" returns an answer mentioning relevant catalogue books

- The answer includes a sources list with book titles and relevance scores

- An off-topic question such as "What is the meaning of life?" returns a polite refusal rather than a fabricated answer

- The same question asked twice returns faster the second time (cache hit)

- Usage tracker shows cost for the first call but not the cached second call

Design your RAG system prompt carefully. Instruct the model to ground answers in the provided context, cite books by title, and admit when it lacks sufficient information. Prompt quality directly determines answer quality.

---

## **Part 5: AI librarian chatbot**

Build a multi-turn conversational chatbot that lets patrons have a natural dialogue with an AI librarian, using RAG to ground answers in the catalogue.

**Requirements:**

- Each conversation is identified by a unique conversation ID

- The chatbot maintains a history of user and assistant messages per conversation

- For every new user message, the chatbot retrieves relevant context from the RAG engine

- The prompt includes both conversation history and retrieved context

- Conversation history must be truncated to stay within the model's context window (keep the most recent N messages)

- The chatbot's personality should be warm, helpful, and knowledgeable — like a friendly librarian

- When no relevant catalogue information is found, the chatbot must respond naturally but must never fabricate book titles or authors

**Acceptance criteria:**

- Starting a new conversation with "Hi!" produces a friendly greeting

- Asking "Recommend a science fiction book" returns a grounded recommendation

- Following up with "Tell me more about that one" produces a detailed answer about the previously recommended book, proving memory works

- Different conversation IDs maintain separate histories

---

## **Part 6: Classification and summarization**

**Task A: Ticket classifier**

Accepts a free-text library support ticket and returns a structured classification.

**Requirements:**

- Accept a raw ticket text string as input

- Return a JSON object containing: category (account, borrowing, technical, complaint, suggestion, general), priority (low, medium, high, urgent), sentiment (positive, neutral, negative), suggested routing department, and a one-sentence summary

- Use a low temperature for consistent, deterministic outputs

- Handle cases where the model wraps JSON in markdown code fences

- Always return valid, parsable JSON — raise a clear error if the model returns invalid output

**Task B: Review summarizer**

Accepts a list of book reviews and produces a structured analysis.

**Requirements:**

- Accept a list of review strings (1–50 reviews)

- Return a JSON object containing: overall sentiment, estimated average rating (1–5), key themes, common points of praise, common points of criticism, and a one-sentence recommendation

- The prompt should instruct the model to consider all reviews holistically, not summarize them one by one

- Handle markdown fence stripping and JSON validation the same way as the classifier

**Acceptance criteria:**

- Classifying "My library card isn't working at self-checkout and I'm very frustrated" returns category=technical, priority=high, sentiment=negative

- Classifying "I love the new reading room, thank you!" returns sentiment=positive with lower priority

- Summarizing 3–5 mixed reviews produces balanced output with both praise and criticism

- All outputs are valid JSON parsable by json.loads() without errors

---

## **Part 7: REST API**

Expose all services through a FastAPI application with clean, documented HTTP endpoints.

**Required endpoints:**

| **Method** | **Path** | **Description** |
| --- | --- | --- |
| POST | /search/books | Semantic search — input: query string and limit; output: list of matching books with scores |
| POST | /search/ask | RAG-powered Q&A — input: question; output: answer, sources, cached flag |
| POST | /chat | Multi-turn chatbot — input: conversation_id, message; output: reply, sources |
| POST | /classify/ticket | Ticket classification — input: ticket text; output: structured JSON classification |
| POST | /summarise/reviews | Review summarization — input: list of review strings; output: structured JSON summary |
| GET | /health | Health check — output: status, daily cost, total request count |

**Requirements:**

- All request bodies validated with Pydantic models (enforce min/max lengths and required fields)

- All endpoints return structured JSON responses

- Rate limit errors return HTTP 429 with a clear message

- AI provider failures return HTTP 503 with a description

- CORS middleware enabled for development convenience

- Application runnable with uvicorn and auto-reload during development

- Interactive Swagger UI at /docs usable for manual testing

**Acceptance criteria:**

- All six endpoints return successful responses with valid input

- Invalid input (empty strings, missing fields) returns HTTP 422 with validation details

- Swagger UI at /docs is accessible and documents all endpoints

- /health endpoint returns current daily spend and request count

---

## **Part 8: Testing and validation**

Write a smoke test script and manually validate every component against the scenarios below.

| **Scenario** | **Expected behaviour** |
| --- | --- |
| Search: "desert planet adventure" | Relevant sci-fi books appear with high scores |
| Ask: "What is the meaning of life?" | Polite refusal — not in catalogue |
| Ask: "Recommend a classic romance novel" | Grounded answer citing books from the catalogue |
| Chat turn 1: "Recommend a thriller" | Suggests a specific book from the catalogue |
| Chat turn 2: "Tell me more about that" | Elaborates on the same book from turn 1 — memory works |
| Classify: angry complaint about card not working | category=technical, priority=high, sentiment=negative |
| Summarize: 3–5 mixed reviews | Balanced sentiment, distinct praise and criticism |
| Same question asked twice | Second call is faster — cache hit |
| Exceed rate limit | Returns HTTP 429 |
| Disable primary provider key | Automatic fallback to secondary provider |

---

## **Submission requirements**

**What to submit:**

- A Git repository (GitHub or GitLab) containing the full LibraryMind codebase

- A README.md with: project description, setup instructions, environment variable documentation, and sample curl or httpx commands for each endpoint

- A reflection document (500–1000 words) covering: key design decisions, challenges faced and how you overcame them, at least one mistake or debugging story, and any extensions attempted

**Grading rubric:**

| **Criterion** | **Weight** | **Notes** |
| --- | --- | --- |
| Multi-provider AI layer with working fallback | 15% | Demonstrate automatic failover |
| Vector store seeded and semantic search works | 15% | At least 20 books, relevant search results |
| RAG engine returns grounded answers with sources | 20% | Relevance filtering, source citations |
| Multi-turn chatbot with conversation memory | 15% | Context preserved across turns |
| Classification and summarization with structured JSON | 10% | Valid JSON, handles edge cases |
| Caching, rate limiting, and usage tracking | 10% | Production concerns addressed |
| Clean, documented FastAPI endpoints | 10% | Swagger UI, validation, error handling |
| Code quality, structure, and README | 5% | Readable, organized, documented |

---

## **Tips and common pitfalls**

**General advice:**

- Start small — get the multi-provider AI layer working end-to-end before touching vectors or RAG

- Commit to Git frequently so you can roll back failed experiments

- Use a cheaper model during development to save your API budget

- Check /health regularly to monitor cumulative daily spend

- Log the full raw AI response before attempting to parse it — this reveals parsing issues immediately

- Read error messages carefully — AI SDK errors are usually very descriptive

**Common pitfalls:**

JSON parsing failures — AI models frequently wrap JSON output in markdown code fences. Your parsing logic must strip these before calling json.loads(). This is the most common failure in classification and summarization tasks.

Rate limit 429 errors from providers — if you hit your provider's rate limit, your retry logic should handle most cases automatically. If it persists, reduce RATE_LIMIT_PER_MINUTE to stay under the provider's threshold.

Distance vs. similarity confusion — some vector databases return cosine distance (lower = more similar) while others return cosine similarity (higher = more similar). Know which your chosen database returns and set your relevance threshold accordingly.

Stale cache during development — if you modify a prompt template but the output doesn't change, you may be receiving a cached result. Flush your cache or include a version identifier in cache keys when iterating on prompts.

Embedding model mismatch — if you change your embedding model after seeding the vector database, query embeddings will be in a different dimensional space than stored vectors. Delete and re-seed the collection whenever you switch embedding models.

Context window overflow — including too much conversation history in a chatbot prompt can exceed the model's context window. Always truncate history to the most recent N messages based on typical message length and the model's token limit.

This lab is intentionally challenging. It requires synthesizing knowledge from all modules and making your own design decisions about code structure, prompt wording, error handling, and performance trade-offs. There is no single correct implementation — the grading criteria focus on whether your system works, handles failures gracefully, and demonstrates understanding of the underlying concepts. The patterns you implement here — provider abstraction, RAG pipelines, prompt engineering, cost management — are exactly what engineering teams use when building AI features for real products.