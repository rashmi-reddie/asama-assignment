"""Core domain logic: registration, publishing, the 3-stage routing funnel, comments, feedback."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass

import numpy as np

from .config import Settings
from .db import Database, from_blob, new_id, to_blob
from .delivery import WebhookDeliverer
from .embeddings import Embedder
from .reranker import Candidate, Reranker
from .security import (generate_api_key, generate_webhook_secret, hash_api_key,
                        screen_content, verify_api_key)

log = logging.getLogger("asama.service")


class NotFound(Exception):
    pass


class Forbidden(Exception):
    pass


class Rejected(Exception):
    def __init__(self, reasons: list[str]):
        super().__init__("content rejected by safety screen: " + ", ".join(reasons))
        self.reasons = reasons


@dataclass
class RoutingOutcome:
    post_id: str
    eligible: int
    ann_candidates: int
    notified: list[dict]
    dropped_below_threshold: int


class NetworkService:
    def __init__(self, db: Database, embedder: Embedder, reranker: Reranker,
                 deliverer: WebhookDeliverer, settings: Settings):
        self.db = db
        self.embedder = embedder
        self.reranker = reranker
        self.deliverer = deliverer
        self.s = settings
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._inflight: set[asyncio.Task] = set()
        self._pending = 0  # posts enqueued but not yet fully routed

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        # Durability: anything still QUEUED (e.g. after a crash) is re-enqueued.
        for row in self.db.query("SELECT id FROM posts WHERE status='QUEUED' ORDER BY created_at"):
            self._enqueue(row["id"])
        self._worker = asyncio.create_task(self._run_worker(), name="routing-worker")

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
        for t in list(self._inflight):
            t.cancel()

    def _enqueue(self, post_id: str) -> None:
        self._pending += 1
        self.queue.put_nowait(post_id)

    async def drain(self, timeout: float = 30.0) -> None:
        """Wait until the queue and in-flight deliveries are done (used by tests/demo)."""
        async def _wait():
            while self._pending > 0 or self._inflight:
                await asyncio.sleep(0.01)
        await asyncio.wait_for(_wait(), timeout)

    async def _run_worker(self) -> None:
        while True:
            post_id = await self.queue.get()
            try:
                await self.route_post(post_id)
            except Exception:
                log.exception("routing failed for post %s", post_id)
            finally:
                self.queue.task_done()
                self._pending -= 1

    # ------------------------------------------------------------------ agents
    def register_agent(self, company_name: str, industry: str, webhook_url: str,
                       manifesto: str, blocked_industries: list[str] | None = None) -> dict:
        agent_id = new_id()
        api_key = generate_api_key()
        secret = generate_webhook_secret()
        emb = self.embedder.embed([f"{company_name}. {industry}. {manifesto}"])[0]
        self.db.execute(
            """INSERT INTO agents (id, company_name, industry_category, webhook_url, interest_manifesto,
               interest_embedding, blocked_industries, api_key_hash, webhook_secret, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (agent_id, company_name, industry, webhook_url, manifesto, to_blob(emb),
             json.dumps([b.lower() for b in (blocked_industries or [])]),
             hash_api_key(api_key), secret, time.time()),
        )
        return {"id": agent_id, "api_key": api_key, "webhook_secret": secret}

    def authenticate(self, agent_id: str, api_key: str | None) -> dict:
        row = self.db.one("SELECT * FROM agents WHERE id=?", (agent_id,))
        if row is None:
            raise NotFound(f"agent {agent_id} not found")
        if not verify_api_key(api_key, row["api_key_hash"]):
            raise Forbidden("invalid API key for this agent")
        if not row["is_active"]:
            raise Forbidden("agent is deactivated")
        return dict(row)

    # ------------------------------------------------------------------ posts
    def publish_post(self, author_agent_id: str, api_key: str | None, title: str, content: str,
                     expires_in_s: float | None = None) -> dict:
        author = self.authenticate(author_agent_id, api_key)
        if author["trust_score"] < self.s.min_author_trust:
            raise Forbidden("author trust score too low to publish")
        guard = screen_content(title, content)
        if not guard.safe:
            new_trust = max(0.0, author["trust_score"] - 0.2)
            self.db.execute("UPDATE agents SET trust_score=? WHERE id=?", (new_trust, author_agent_id))
            post_id = new_id()
            self.db.execute(
                "INSERT INTO posts (id, author_agent_id, title, content, status, created_at) VALUES (?,?,?,?, 'REJECTED', ?)",
                (post_id, author_agent_id, title, content, time.time()),
            )
            raise Rejected(guard.reasons)
        emb = self.embedder.embed([f"{title}. {content}"])[0]
        post_id = new_id()
        now = time.time()
        self.db.execute(
            """INSERT INTO posts (id, author_agent_id, title, content, post_embedding, status, expires_at, created_at)
               VALUES (?,?,?,?,?,'QUEUED',?,?)""",
            (post_id, author_agent_id, title, content, to_blob(emb),
             now + expires_in_s if expires_in_s else None, now),
        )
        self._enqueue(post_id)  # <- the "post.published" event
        return {"post_id": post_id, "status": "QUEUED"}

    # ------------------------------------------------------------------ routing funnel
    def _stage1_sql_filter(self, author: dict) -> list[dict]:
        """Stage 1: cheap metadata guardrails."""
        rows = self.db.query(
            "SELECT * FROM agents WHERE is_active=1 AND id != ? AND trust_score >= ?",
            (author["id"], 0.0),
        )
        author_industry = author["industry_category"].lower()
        eligible = []
        for r in rows:
            blocked = json.loads(r["blocked_industries"])
            if author_industry in blocked:
                continue
            if r["industry_category"].lower() in json.loads(author["blocked_industries"]):
                continue
            eligible.append(dict(r))
        return eligible

    def _stage2_ann(self, post_vec: np.ndarray, eligible: list[dict]) -> list[tuple[dict, float]]:
        """Stage 2: dense cosine search over the eligible agents' (dynamic) interest vectors."""
        if not eligible:
            return []
        mat = np.stack([
            from_blob(a["dynamic_embedding"]) if a["dynamic_embedding"] else from_blob(a["interest_embedding"])
            for a in eligible
        ])
        sims = mat @ post_vec  # vectors are L2-normalised => cosine similarity
        k = min(self.s.ann_top_k, len(eligible))
        top = np.argpartition(-sims, k - 1)[:k] if k < len(eligible) else np.arange(len(eligible))
        top = top[np.argsort(-sims[top])]
        return [(eligible[i], float(sims[i])) for i in top]

    async def route_post(self, post_id: str) -> RoutingOutcome:
        post = self.db.one("SELECT * FROM posts WHERE id=?", (post_id,))
        if post is None:
            raise NotFound(post_id)
        if post["status"] != "QUEUED":
            return RoutingOutcome(post_id, 0, 0, [], 0)
        author = dict(self.db.one("SELECT * FROM agents WHERE id=?", (post["author_agent_id"],)))
        if post["expires_at"] and post["expires_at"] < time.time():
            self.db.execute("UPDATE posts SET status='ROUTED' WHERE id=?", (post_id,))
            return RoutingOutcome(post_id, 0, 0, [], 0)

        post_vec = from_blob(post["post_embedding"])
        eligible = self._stage1_sql_filter(author)                       # Stage 1
        ann = self._stage2_ann(post_vec, eligible)                       # Stage 2
        cands = [Candidate(a["id"], a["company_name"], a["industry_category"],
                           a["interest_manifesto"], sim) for a, sim in ann]
        scored = await self.reranker.score(post["title"], post["content"],  # Stage 3
                                           author["industry_category"], cands)
        by_id = {s.agent_id: s for s in scored}
        passing = sorted(
            (c for c in cands if by_id[c.agent_id].score >= self.s.rerank_threshold),
            key=lambda c: by_id[c.agent_id].score, reverse=True,
        )[: self.s.max_notified]
        dropped = len(cands) - len(passing)

        agents_by_id = {a["id"]: a for a in eligible}
        notified = []
        for c in passing:
            sc = by_id[c.agent_id]
            self.db.execute(
                """INSERT OR IGNORE INTO routing_logs (id, post_id, target_agent_id, vector_score,
                   rerank_score, rerank_reason, status, created_at) VALUES (?,?,?,?,?,?, 'NOTIFIED', ?)""",
                (new_id(), post_id, c.agent_id, c.vector_score, sc.score, sc.reason, time.time()),
            )
            notified.append({"agent_id": c.agent_id, "company": c.company_name,
                             "vector_score": round(c.vector_score, 4), "rerank_score": sc.score})
            self._spawn_delivery(agents_by_id[c.agent_id], post, author, sc.score)

        self.db.execute("UPDATE posts SET status='ROUTED' WHERE id=?", (post_id,))
        return RoutingOutcome(post_id, len(eligible), len(ann), notified, dropped)

    # ------------------------------------------------------------------ delivery
    def _spawn_delivery(self, target: dict, post, author: dict, score: float) -> None:
        # Untrusted post text is carried as plain data fields, clearly labelled.
        event = {
            "event_type": "post.relevant",
            "post": {"id": post["id"], "title": post["title"], "content": post["content"],
                     "author": {"agent_id": author["id"], "company": author["company_name"],
                                "industry": author["industry_category"]},
                     "untrusted_content": True},
            "relevance_score": score,
            "reply_endpoint": f"/api/v1/posts/{post['id']}/comments",
        }

        async def _go():
            res = await self.deliverer.deliver(target["id"], target["webhook_url"],
                                               target["webhook_secret"], event)
            status = "NOTIFIED" if res.ok else "FAILED"
            self.db.execute(
                """UPDATE routing_logs SET delivery_attempts=?, status=CASE WHEN status='ENGAGED' THEN status ELSE ? END
                   WHERE post_id=? AND target_agent_id=?""",
                (res.attempts, status, post["id"], target["id"]),
            )

        task = asyncio.create_task(_go())
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    # ------------------------------------------------------------------ comments
    def add_comment(self, post_id: str, author_agent_id: str, api_key: str | None, content: str) -> dict:
        self.authenticate(author_agent_id, api_key)
        post = self.db.one("SELECT id FROM posts WHERE id=?", (post_id,))
        if post is None:
            raise NotFound(f"post {post_id} not found")
        guard = screen_content(content)
        if not guard.safe:
            self.db.execute("UPDATE agents SET trust_score=MAX(0, trust_score-0.2) WHERE id=?", (author_agent_id,))
            raise Rejected(guard.reasons)
        cid = new_id()
        self.db.execute(
            "INSERT INTO comments (id, post_id, author_agent_id, content, created_at) VALUES (?,?,?,?,?)",
            (cid, post_id, author_agent_id, content, time.time()),
        )
        # Closed loop: a comment from a notified agent marks the notification ENGAGED.
        self.db.execute(
            "UPDATE routing_logs SET status='ENGAGED' WHERE post_id=? AND target_agent_id=?",
            (post_id, author_agent_id),
        )
        return {"comment_id": cid}

    # ------------------------------------------------------------------ feedback loop
    def feedback_sweep(self, now: float | None = None) -> dict:
        """Mark stale notifications IGNORED, then adapt each agent's dynamic preference vector."""
        now = now or time.time()
        cutoff = now - self.s.ignore_after_s
        cur = self.db.execute(
            "UPDATE routing_logs SET status='IGNORED' WHERE status='NOTIFIED' AND created_at <= ?", (cutoff,)
        )
        marked = cur.rowcount
        adapted = []
        for a in self.db.query("SELECT id FROM agents WHERE is_active=1"):
            if self._adapt_agent(a["id"]):
                adapted.append(a["id"])
        return {"marked_ignored": marked, "agents_adapted": adapted}

    def _adapt_agent(self, agent_id: str) -> bool:
        """v' = normalise(v + w_pos * mean(engaged posts) - w_neg * mean(ignored posts))."""
        rows = self.db.query(
            """SELECT l.status, p.post_embedding FROM routing_logs l JOIN posts p ON p.id=l.post_id
               WHERE l.target_agent_id=? AND l.status IN ('ENGAGED','IGNORED')""", (agent_id,),
        )
        neg = [from_blob(r["post_embedding"]) for r in rows if r["status"] == "IGNORED"]
        pos = [from_blob(r["post_embedding"]) for r in rows if r["status"] == "ENGAGED"]
        if len(neg) < self.s.adapt_min_samples and len(pos) < self.s.adapt_min_samples:
            return False
        agent = self.db.one("SELECT interest_embedding FROM agents WHERE id=?", (agent_id,))
        v = from_blob(agent["interest_embedding"]).copy()  # always adapt from the static anchor
        if len(pos) >= self.s.adapt_min_samples:
            v += self.s.adapt_positive_weight * np.mean(pos, axis=0)
        if len(neg) >= self.s.adapt_min_samples:
            v -= self.s.adapt_negative_weight * np.mean(neg, axis=0)
        n = np.linalg.norm(v)
        if n == 0:
            return False
        self.db.execute("UPDATE agents SET dynamic_embedding=? WHERE id=?", (to_blob(v / n), agent_id))
        return True

    def metrics(self) -> list[dict]:
        rows = self.db.query(
            """SELECT a.id, a.company_name, a.trust_score,
                      SUM(CASE WHEN l.id IS NOT NULL THEN 1 ELSE 0 END) AS notifications,
                      SUM(CASE WHEN l.status='ENGAGED' THEN 1 ELSE 0 END) AS engaged,
                      SUM(CASE WHEN l.status='IGNORED' THEN 1 ELSE 0 END) AS ignored,
                      SUM(CASE WHEN l.status='FAILED' THEN 1 ELSE 0 END) AS failed,
                      CASE WHEN a.dynamic_embedding IS NULL THEN 0 ELSE 1 END AS adapted
               FROM agents a LEFT JOIN routing_logs l ON l.target_agent_id=a.id
               GROUP BY a.id ORDER BY a.company_name"""
        )
        out = []
        for r in rows:
            n = r["notifications"] or 0
            out.append({"agent_id": r["id"], "company": r["company_name"], "trust_score": r["trust_score"],
                        "notifications": n, "engaged": r["engaged"] or 0, "ignored": r["ignored"] or 0,
                        "failed": r["failed"] or 0, "engagement_rate": round((r["engaged"] or 0) / n, 3) if n else None,
                        "preference_vector_adapted": bool(r["adapted"])})
        return out
