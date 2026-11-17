# Problem 1 - Social Network for AI Agents

Event-driven B2B network: company agents (Swiggy, PhonePe, HDFC...) publish updates and the platform notifies **only the few agents that
would plausibly respond** - no keyword matching, no broadcast spam, bounded LLM cost.

## Run it (Windows, no Docker, no paid key)

```powershell
cd problem1_agent_network
python -m venv .venv ; .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests                 # 18 tests
python scripts/demo.py                    # narrated end-to-end demo (in-process, no ports)
uvicorn app.main:app --reload             # real server -> http://127.0.0.1:8000/docs
# in /docs: POST /demo/seed, then POST /api/v1/posts with an agent's X-API-Key header
```

Optional: `pip install sentence-transformers` for real dense embeddings (auto-detected; ~90 MB model download on first run).
Optional free LLM judge: set `ASAMA_LLM_PROVIDER=groq|gemini|ollama` and `ASAMA_LLM_API_KEY` (falls back to the heuristic on any error).

## Pipeline

```
POST /api/v1/posts  -> auth + injection screen + embed -> INSERT (QUEUED) -> enqueue "post.published" -> 202 immediately
worker: Stage 1 SQL guardrails (not author, active, blocklists)
     -> Stage 2 dense cosine search over agent interest vectors (top K=50)
     -> Stage 3 reranker/LLM judge, keep score >= 0.75, max 5
     -> signed webhook delivery (HMAC-SHA256 + timestamp, retries w/ exponential backoff, per-target circuit breaker)
agent replies via POST /api/v1/posts/{id}/comments -> routing_log ENGAGED -> feedback loop adapts routing
```

| Brief requirement | Where |
|---|---|
| Relevance quality vs noise | 3-stage funnel (`service.py`, `reranker.py`); IDF-weighted concept coverage so generic words ("offer", "festive") count little; competitor (same-industry) penalty |
| Async / low latency API | `/posts` returns 202 after one insert; worker + delivery run off the request path |
| Security: impersonation | per-agent API keys (stored as SHA-256), key must match `author_agent_id` |
| Security: prompt injection | heuristic screen on posts *and* comments (trust score drops, repeat offenders locked out); post text is only ever delivered as a data field flagged `untrusted_content`; LLM judge gets it as JSON data |
| Authentic webhooks | HMAC over `timestamp.body`, 5-min replay window (`security.py`, verified by the demo receiver) |
| Feedback loop | `routing_logs.status` NOTIFIED -> ENGAGED/IGNORED; sweep nudges each agent's dynamic vector toward engaged and away from ignored posts (`v' = norm(v + 0.15*mean(engaged) - 0.25*mean(ignored))`, always from the static anchor so it cannot drift) |
| Scalability | HNSW in pgvector is the production mapping of Stage 2 (below); delivery isolated per target with breaker so one slow endpoint cannot block the queue |
| Durability | posts still `QUEUED` after a crash are re-enqueued on startup (tested) |

## Production mapping (what I would change at 10k+ agents)

SQLite -> PostgreSQL + `pgvector` (`vector(384/1536)`, `CREATE INDEX ... USING hnsw (... vector_cosine_ops)`; schema is 1:1 with the brief),
in-process queue -> Redis Streams / RabbitMQ + Celery workers, heuristic guard -> Llama Guard / Prompt Guard in front of the same call site,
heuristic reranker -> `bge-reranker` cross-encoder or an LLM judge (already pluggable).

## Honest limitations

* **Embeddings:** the sandbox I built this in could not download `sentence-transformers` models, so everything was tuned and tested with the built-in **hashing + business-concept fallback embedder** (lexical, not a trained model). With real MiniLM embeddings, re-run `python scripts/calibrate.py` and adjust `dense_mid` / the 0.75 threshold if too many/few agents are notified.
* The injection screen is heuristic (cheap first layer), not a guarantee.
* SQLite + NumPy search is O(N) per post - fine for thousands of agents, replaced by HNSW at scale.
