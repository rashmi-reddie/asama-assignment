"""Webhook delivery with HMAC signing, exponential backoff and a per-target circuit breaker.

A slow or dead third-party endpoint must never stall the routing pipeline, so every delivery
runs as its own asyncio task with a hard timeout, and a target that keeps failing is
short-circuited ("breaker open") for a cool-down period.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass

import httpx

from .security import sign_payload

log = logging.getLogger("asama.delivery")


@dataclass
class DeliveryResult:
    ok: bool
    attempts: int
    error: str | None = None


class CircuitBreaker:
    def __init__(self, failures: int, cooldown_s: float):
        self.failures = failures
        self.cooldown_s = cooldown_s
        self._fail_count: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}

    def allow(self, key: str) -> bool:
        opened = self._opened_at.get(key)
        if opened is None:
            return True
        if time.monotonic() - opened >= self.cooldown_s:  # half-open: let one through
            del self._opened_at[key]
            self._fail_count[key] = self.failures - 1
            return True
        return False

    def record_success(self, key: str) -> None:
        self._fail_count.pop(key, None)
        self._opened_at.pop(key, None)

    def record_failure(self, key: str) -> None:
        n = self._fail_count.get(key, 0) + 1
        self._fail_count[key] = n
        if n >= self.failures:
            self._opened_at[key] = time.monotonic()

    def is_open(self, key: str) -> bool:
        return key in self._opened_at


class WebhookDeliverer:
    def __init__(self, client: httpx.AsyncClient, attempts: int, backoff_s: float,
                 timeout_s: float, breaker: CircuitBreaker):
        self.client = client
        self.attempts = attempts
        self.backoff_s = backoff_s
        self.timeout_s = timeout_s
        self.breaker = breaker

    async def deliver(self, agent_id: str, url: str, secret: str, event: dict) -> DeliveryResult:
        if not self.breaker.allow(agent_id):
            return DeliveryResult(False, 0, "circuit breaker open")
        body = json.dumps(event, separators=(",", ":"), sort_keys=True).encode()
        last_err = "unknown"
        for attempt in range(1, self.attempts + 1):
            signature, ts = sign_payload(secret, body)
            headers = {
                "Content-Type": "application/json",
                "X-Asama-Signature": signature,
                "X-Asama-Timestamp": str(ts),
                "X-Asama-Event": event.get("event_type", "post.relevant"),
                "User-Agent": "asama-network/1.0",
            }
            try:
                resp = await self.client.post(url, content=body, headers=headers, timeout=self.timeout_s)
                if resp.status_code < 300:
                    self.breaker.record_success(agent_id)
                    return DeliveryResult(True, attempt)
                last_err = f"HTTP {resp.status_code}"
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    break  # client error: retrying will not help
            except (httpx.HTTPError, asyncio.TimeoutError) as exc:
                last_err = f"{type(exc).__name__}: {exc}"
            if attempt < self.attempts:
                await asyncio.sleep(self.backoff_s * (2 ** (attempt - 1)))
        self.breaker.record_failure(agent_id)
        log.warning("delivery to %s failed: %s", agent_id, last_err)
        return DeliveryResult(False, attempt, last_err)
