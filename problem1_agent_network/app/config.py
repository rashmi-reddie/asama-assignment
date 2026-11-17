"""Runtime configuration, all overridable through environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass
class Settings:
    db_path: str = field(default_factory=lambda: os.getenv("ASAMA_DB_PATH", "asama.db"))

    # --- Routing funnel -------------------------------------------------
    ann_top_k: int = field(default_factory=lambda: _i("ASAMA_ANN_TOP_K", 50))
    rerank_threshold: float = field(default_factory=lambda: _f("ASAMA_RERANK_THRESHOLD", 0.75))
    max_notified: int = field(default_factory=lambda: _i("ASAMA_MAX_NOTIFIED", 5))
    min_author_trust: float = field(default_factory=lambda: _f("ASAMA_MIN_AUTHOR_TRUST", 0.2))

    # --- Embeddings: auto | st | hash ----------------------------------
    embedding_backend: str = field(default_factory=lambda: os.getenv("ASAMA_EMBEDDINGS", "auto"))
    st_model: str = field(
        default_factory=lambda: os.getenv("ASAMA_ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    # --- Optional LLM judge (free tiers): none | groq | gemini | ollama --
    llm_provider: str = field(default_factory=lambda: os.getenv("ASAMA_LLM_PROVIDER", "none"))
    llm_model: str = field(default_factory=lambda: os.getenv("ASAMA_LLM_MODEL", ""))
    llm_api_key: str = field(default_factory=lambda: os.getenv("ASAMA_LLM_API_KEY", ""))

    # --- Webhook delivery ------------------------------------------------
    delivery_attempts: int = field(default_factory=lambda: _i("ASAMA_DELIVERY_ATTEMPTS", 3))
    delivery_backoff_s: float = field(default_factory=lambda: _f("ASAMA_DELIVERY_BACKOFF_S", 0.5))
    delivery_timeout_s: float = field(default_factory=lambda: _f("ASAMA_DELIVERY_TIMEOUT_S", 5.0))
    breaker_failures: int = field(default_factory=lambda: _i("ASAMA_BREAKER_FAILURES", 3))
    breaker_cooldown_s: float = field(default_factory=lambda: _f("ASAMA_BREAKER_COOLDOWN_S", 60.0))

    # --- Feedback loop ---------------------------------------------------
    ignore_after_s: float = field(default_factory=lambda: _f("ASAMA_IGNORE_AFTER_S", 3600.0))
    adapt_min_samples: int = field(default_factory=lambda: _i("ASAMA_ADAPT_MIN_SAMPLES", 3))
    adapt_negative_weight: float = field(default_factory=lambda: _f("ASAMA_ADAPT_NEG_W", 0.25))
    adapt_positive_weight: float = field(default_factory=lambda: _f("ASAMA_ADAPT_POS_W", 0.15))


def get_settings() -> Settings:
    return Settings()
