"""Trust & safety: API keys, HMAC webhook signing, prompt-injection screening."""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from dataclasses import dataclass


# ----------------------------------------------------------------------------- API keys
def generate_api_key() -> str:
    return "asama_" + secrets.token_urlsafe(32)


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def verify_api_key(presented: str | None, stored_hash: str) -> bool:
    if not presented:
        return False
    return hmac.compare_digest(hash_api_key(presented), stored_hash)


def generate_webhook_secret() -> str:
    return secrets.token_hex(32)


# --------------------------------------------------------------------- webhook signatures
def sign_payload(secret: str, body: bytes, timestamp: int | None = None) -> tuple[str, int]:
    """Return ("sha256=<hex>", timestamp).  Signature covers ``timestamp.body`` (replay defence)."""
    ts = int(time.time()) if timestamp is None else timestamp
    mac = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return f"sha256={mac}", ts


def verify_signature(
    secret: str, body: bytes, signature: str, timestamp: int, tolerance_s: int = 300
) -> bool:
    if abs(time.time() - timestamp) > tolerance_s:
        return False
    expected, _ = sign_payload(secret, body, timestamp)
    return hmac.compare_digest(expected, signature)


# ------------------------------------------------------------------ prompt-injection guard
@dataclass
class GuardResult:
    safe: bool
    reasons: list[str]


# Heuristic screen.  In production this is a first, cheap layer in front of a model such as
# Llama Guard / Prompt Guard; the architecture treats the *content as data* either way.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("override-instructions", re.compile(r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|system)\b.{0,30}\b(instruction|prompt|rule|message|guideline)s?\b", re.I | re.S)),
    ("role-hijack", re.compile(r"\b(you are now|act as|pretend to be|from now on you)\b", re.I)),
    ("prompt-exfil", re.compile(r"\b(reveal|print|show|output|leak|repeat)\b.{0,40}\b(system prompt|instructions|secret|api[\s_-]?key|password|credential|token)s?\b", re.I | re.S)),
    ("secret-request", re.compile(r"\b(api[\s_-]?key|webhook[\s_-]?secret|password|private key|credentials?)\b.{0,30}\b(send|post|forward|share|give|email)\b", re.I | re.S)),
    ("delimiter-spoof", re.compile(r"(<\|?(system|im_start|im_end)\|?>|\[/?INST\]|###\s*(system|instruction))", re.I)),
    ("tool-abuse", re.compile(r"\b(execute|run|call)\b.{0,30}\b(tool|function|command|shell|curl|rm\s+-rf)\b", re.I | re.S)),
]


def screen_content(*texts: str) -> GuardResult:
    reasons: list[str] = []
    blob = "\n".join(texts)
    for name, pat in _PATTERNS:
        if pat.search(blob):
            reasons.append(name)
    return GuardResult(safe=not reasons, reasons=reasons)
