"""Embedding backends.

* ``SentenceTransformerEmbedder``  - real dense embeddings (free, runs locally on CPU).
  Used automatically when ``sentence-transformers`` is installed and the model can be loaded.
* ``HashingEmbedder``              - zero-dependency fallback so the project always runs
  offline (CI, tests, a laptop without the model downloaded).  It is a hashed bag of
  unigrams/bigrams with a small *business concept lexicon* so that e.g. "Diwali offer" and
  "UPI cashback" land near each other.  It is NOT a substitute for a trained model; it exists
  so the whole pipeline is runnable and testable with no network.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Protocol

import numpy as np

log = logging.getLogger("asama.embeddings")


class Embedder(Protocol):
    name: str
    dim: int
    dense_mid: float  # cosine meaning "somewhat related" (used to calibrate the reranker)

    def embed(self, texts: list[str]) -> np.ndarray:  # rows are L2-normalised
        ...


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


# --------------------------------------------------------------------------------------
# Fallback embedder
# --------------------------------------------------------------------------------------
_STOP = set(
    "a an the and or of for to in on at by with from is are be as it its this that these those "
    "we our you your their they them will can may per all any up out into over under about "
    "interested looking offering new get gets flat above during".split()
)

# token -> concept tokens it implies.  Keeps the fallback honest about *business* semantics.
_CONCEPTS: dict[str, list[str]] = {
    "diwali": ["festive", "festival", "offer", "discount", "promotion", "campaign"],
    "festive": ["festival", "offer", "discount", "promotion"],
    "festival": ["festive", "offer", "discount", "promotion"],
    "discount": ["offer", "promotion", "cashback", "deal"],
    "offer": ["discount", "promotion", "deal"],
    "off": ["discount", "offer", "promotion"],
    "cashback": ["payment", "discount", "reward", "promotion", "offer"],
    "upi": ["payment", "digital", "wallet", "cashback", "fintech"],
    "payment": ["payments", "upi", "card", "checkout", "fintech"],
    "payments": ["payment", "upi", "card", "checkout", "fintech"],
    "card": ["payment", "credit", "bank", "cashback", "emi"],
    "credit": ["card", "bank", "loan", "emi", "fintech"],
    "emi": ["credit", "loan", "financing", "payment", "bank"],
    "loan": ["credit", "emi", "financing", "bank"],
    "bank": ["banking", "card", "credit", "payment", "loan"],
    "delivery": ["food", "logistics", "order", "restaurant"],
    "food": ["delivery", "restaurant", "order", "dining"],
    "restaurant": ["food", "dining", "delivery"],
    "ecommerce": ["shopping", "retail", "marketplace", "order", "merchant"],
    "e-commerce": ["shopping", "retail", "marketplace", "order", "merchant"],
    "shopping": ["ecommerce", "retail", "marketplace"],
    "marketplace": ["ecommerce", "shopping", "merchant", "retail"],
    "merchant": ["ecommerce", "payment", "retail", "cashback"],
    "retail": ["shopping", "ecommerce", "merchant"],
    "electronics": ["ecommerce", "shopping", "gadget", "emi"],
    "smartphone": ["electronics", "gadget", "emi", "shopping"],
    "bike": ["vehicle", "automobile", "motorcycle", "emi", "loan"],
    "motorcycle": ["bike", "vehicle", "automobile", "emi", "loan"],
    "automobile": ["vehicle", "bike", "car", "loan", "emi"],
    "vehicle": ["automobile", "bike", "loan", "emi"],
    "ev": ["electric", "vehicle", "automobile", "charging"],
    "electric": ["ev", "vehicle", "charging"],
    "ride": ["mobility", "transport", "cab", "travel"],
    "cab": ["ride", "mobility", "transport", "travel"],
    "travel": ["flight", "hotel", "booking", "tourism"],
    "flight": ["travel", "booking", "airline"],
    "hotel": ["travel", "booking", "stay"],
    "streaming": ["entertainment", "subscription", "media", "ott"],
    "ott": ["streaming", "entertainment", "subscription"],
    "subscription": ["membership", "recurring", "payment"],
    "insurance": ["protection", "policy", "bank", "fintech"],
    "grocery": ["delivery", "food", "quick-commerce", "retail"],
    "quick-commerce": ["grocery", "delivery", "retail"],
    "customer": ["users", "audience"],
    "users": ["customer", "audience"],
}

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-]*")


def _stem(tok: str) -> str:
    for suf in ("ing", "ies", "es", "s", "ed"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            return tok[: -len(suf)] + ("y" if suf == "ies" else "")
    return tok


def _tokens(text: str) -> list[str]:
    toks = [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]
    return toks


def expand_concepts(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for t in tokens:
        out.extend(_CONCEPTS.get(t, []))
        out.extend(_CONCEPTS.get(t + "s", []))
    return out


class HashingEmbedder:
    name = "hashing-concept-fallback"
    dense_mid = 0.30

    def __init__(self, dim: int = 768):
        self.dim = dim

    def _slot(self, feature: str) -> tuple[int, float]:
        h = hashlib.md5(feature.encode()).digest()
        idx = int.from_bytes(h[:4], "little") % self.dim
        sign = 1.0 if h[4] & 1 else -1.0
        return idx, sign

    def embed(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, text in enumerate(texts):
            toks = _tokens(text)
            feats: dict[str, float] = {}
            for t in toks:
                feats[t] = feats.get(t, 0.0) + 1.0
            for a, b in zip(toks, toks[1:]):
                feats[f"{a}_{b}"] = feats.get(f"{a}_{b}", 0.0) + 0.5
            for c in expand_concepts(toks):
                feats[c] = feats.get(c, 0.0) + 0.6
            for f, w in feats.items():
                idx, sign = self._slot(f)
                mat[r, idx] += sign * (1.0 + np.log(w)) if w >= 1 else sign * w
        return _normalise(mat)


# --------------------------------------------------------------------------------------
# Real model
# --------------------------------------------------------------------------------------
class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers:{model_name}"
        self.dense_mid = 0.40  # MiniLM-style models: related business text typically sits at 0.4-0.7
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


def build_embedder(backend: str = "auto", st_model: str = "") -> Embedder:
    if backend in ("auto", "st"):
        try:
            emb = SentenceTransformerEmbedder(st_model or "sentence-transformers/all-MiniLM-L6-v2")
            log.info("Using %s (dim=%s)", emb.name, emb.dim)
            return emb
        except Exception as exc:  # not installed / offline / model missing
            if backend == "st":
                raise
            log.warning("sentence-transformers unavailable (%s); using hashing fallback", exc)
    return HashingEmbedder()
