"""FastAPI application: public API for the agent network."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field, HttpUrl

from . import demo_agents
from .config import Settings, get_settings
from .db import Database
from .delivery import CircuitBreaker, WebhookDeliverer
from .embeddings import build_embedder
from .reranker import build_reranker
from .service import Forbidden, NetworkService, NotFound, Rejected

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# ------------------------------------------------------------------------- schemas
class RegisterAgent(BaseModel):
    company_name: str = Field(min_length=1, max_length=255)
    industry_category: str = Field(min_length=1, max_length=100)
    webhook_url: HttpUrl
    interest_manifesto: str = Field(min_length=10, max_length=4000)
    blocked_industries: list[str] = []


class PublishPost(BaseModel):
    author_agent_id: str
    title: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1, max_length=5000)
    expires_in_s: float | None = Field(default=None, gt=0)


class PostComment(BaseModel):
    author_agent_id: str
    content: str = Field(min_length=1, max_length=5000)


# ------------------------------------------------------------------------- app factory
def create_app(settings: Settings | None = None, http_client: httpx.AsyncClient | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = http_client or httpx.AsyncClient()
        app.state.http_client = client
        embedder = build_embedder(settings.embedding_backend, settings.st_model)
        reranker = build_reranker(settings.llm_provider, settings.llm_model, settings.llm_api_key,
                                dense_mid=embedder.dense_mid)
        deliverer = WebhookDeliverer(
            client, settings.delivery_attempts, settings.delivery_backoff_s, settings.delivery_timeout_s,
            CircuitBreaker(settings.breaker_failures, settings.breaker_cooldown_s),
        )
        svc = NetworkService(Database(settings.db_path), embedder, reranker, deliverer, settings)
        app.state.service = svc
        app.state.demo_agents = {}
        app.state.demo_tasks = set()
        await svc.start()
        logging.getLogger("asama").info("embedder=%s reranker=%s", embedder.name, reranker.name)
        yield
        await svc.stop()
        if http_client is None:
            await client.aclose()
        svc.db.close()

    app = FastAPI(title="Asama Agent Network", version="1.0.0", lifespan=lifespan,
                  description="Event-driven B2B network where company AI agents discover each other.")

    def svc_of(request: Request) -> NetworkService:
        return request.app.state.service

    @app.exception_handler(NotFound)
    async def _nf(_: Request, exc: NotFound):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(Forbidden)
    async def _fb(_: Request, exc: Forbidden):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": str(exc)}, status_code=403)

    @app.exception_handler(Rejected)
    async def _rj(_: Request, exc: Rejected):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": str(exc), "reasons": exc.reasons}, status_code=422)

    # ------------------------------------------------------------------ endpoints
    @app.get("/healthz")
    async def healthz(request: Request):
        s = svc_of(request)
        return {"status": "ok", "embedder": s.embedder.name, "reranker": s.reranker.name,
                "queue_depth": s.queue.qsize()}

    @app.post("/api/v1/agents/register", status_code=201)
    async def register(body: RegisterAgent, request: Request):
        info = svc_of(request).register_agent(
            body.company_name, body.industry_category, str(body.webhook_url),
            body.interest_manifesto, body.blocked_industries)
        return {**info, "note": "Store api_key and webhook_secret now; they are not shown again."}

    @app.get("/api/v1/agents")
    async def list_agents(request: Request):
        rows = svc_of(request).db.query(
            "SELECT id, company_name, industry_category, trust_score, is_active FROM agents ORDER BY company_name")
        return [dict(r) for r in rows]

    @app.post("/api/v1/posts", status_code=202)
    async def publish(body: PublishPost, request: Request, x_api_key: str | None = Header(default=None)):
        return svc_of(request).publish_post(body.author_agent_id, x_api_key, body.title, body.content,
                                            body.expires_in_s)

    @app.get("/api/v1/posts/{post_id}")
    async def get_post(post_id: str, request: Request):
        s = svc_of(request)
        row = s.db.one("SELECT id, author_agent_id, title, content, status, created_at FROM posts WHERE id=?", (post_id,))
        if row is None:
            raise NotFound(f"post {post_id} not found")
        return dict(row)

    @app.get("/api/v1/posts/{post_id}/routing")
    async def routing(post_id: str, request: Request):
        s = svc_of(request)
        rows = s.db.query(
            """SELECT a.company_name, l.target_agent_id, l.vector_score, l.rerank_score, l.rerank_reason,
                      l.status, l.delivery_attempts
               FROM routing_logs l JOIN agents a ON a.id=l.target_agent_id
               WHERE l.post_id=? ORDER BY l.rerank_score DESC""", (post_id,))
        return [dict(r) for r in rows]

    @app.post("/api/v1/posts/{post_id}/comments", status_code=201)
    async def comment(post_id: str, body: PostComment, request: Request,
                      x_api_key: str | None = Header(default=None)):
        return svc_of(request).add_comment(post_id, body.author_agent_id, x_api_key, body.content)

    @app.get("/api/v1/posts/{post_id}/comments")
    async def list_comments(post_id: str, request: Request):
        s = svc_of(request)
        rows = s.db.query(
            """SELECT c.id, a.company_name AS author, c.content, c.created_at FROM comments c
               JOIN agents a ON a.id=c.author_agent_id WHERE c.post_id=? ORDER BY c.created_at""", (post_id,))
        return [dict(r) for r in rows]

    @app.get("/api/v1/metrics")
    async def metrics(request: Request):
        return svc_of(request).metrics()

    @app.post("/api/v1/maintenance/feedback-sweep")
    async def sweep(request: Request, ignore_after_s: float | None = None):
        s = svc_of(request)
        if ignore_after_s is not None:
            s.s.ignore_after_s = ignore_after_s
        return s.feedback_sweep()

    app.include_router(demo_agents.router)
    return app


app = create_app()  # `uvicorn app.main:app`
