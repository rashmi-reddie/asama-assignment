"""SQLite persistence layer.

The brief suggests PostgreSQL + pgvector.  To keep the project runnable on a plain Windows
laptop (no Docker, no Postgres) this uses SQLite and does the vector search with NumPy.
The schema mirrors the brief's tables 1:1, so moving to Postgres + pgvector is a
mechanical change (see README, "Production mapping").
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from typing import Any, Iterable

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    company_name TEXT NOT NULL,
    industry_category TEXT NOT NULL,
    webhook_url TEXT NOT NULL,
    interest_manifesto TEXT NOT NULL,
    interest_embedding BLOB NOT NULL,          -- static, from the manifesto
    dynamic_embedding BLOB,                    -- learned from feedback (NULL = use static)
    blocked_industries TEXT NOT NULL DEFAULT '[]',
    api_key_hash TEXT NOT NULL,
    webhook_secret TEXT NOT NULL,
    trust_score REAL NOT NULL DEFAULT 1.0,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS posts (
    id TEXT PRIMARY KEY,
    author_agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    post_embedding BLOB,
    status TEXT NOT NULL DEFAULT 'QUEUED',     -- QUEUED, ROUTED, REJECTED
    expires_at REAL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS routing_logs (
    id TEXT PRIMARY KEY,
    post_id TEXT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    target_agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    vector_score REAL NOT NULL,
    rerank_score REAL NOT NULL,
    rerank_reason TEXT,
    status TEXT NOT NULL DEFAULT 'NOTIFIED',   -- NOTIFIED, ENGAGED, IGNORED, FAILED
    delivery_attempts INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    UNIQUE(post_id, target_agent_id)
);

CREATE TABLE IF NOT EXISTS comments (
    id TEXT PRIMARY KEY,
    post_id TEXT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    author_agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agents_active ON agents(is_active);
CREATE INDEX IF NOT EXISTS idx_logs_post ON routing_logs(post_id);
CREATE INDEX IF NOT EXISTS idx_logs_agent ON routing_logs(target_agent_id, status);
CREATE INDEX IF NOT EXISTS idx_comments_post ON comments(post_id);
"""


def new_id() -> str:
    return str(uuid.uuid4())


def to_blob(v: np.ndarray | None) -> bytes | None:
    return None if v is None else np.asarray(v, dtype=np.float32).tobytes()


def from_blob(b: bytes | None) -> np.ndarray | None:
    return None if b is None else np.frombuffer(b, dtype=np.float32).copy()


class Database:
    """Tiny thread-safe wrapper around one SQLite connection."""

    def __init__(self, path: str = "asama.db"):
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cur

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    def one(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ----- convenience ------------------------------------------------------------
    @staticmethod
    def now() -> float:
        return time.time()

    @staticmethod
    def dumps(obj: Any) -> str:
        return json.dumps(obj)
