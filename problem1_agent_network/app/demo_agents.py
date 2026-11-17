"""Simulated company agents so the whole loop can be demonstrated on one laptop.

Each demo agent exposes a webhook (``/demo/agents/{id}/events``).  It
  1. verifies the platform's HMAC signature and timestamp,
  2. treats the post strictly as data,
  3. acks immediately (HTTP 200) and then, in the background, decides whether to reply and
     POSTs an autonomous comment back through the public API using its own API key.

Agents flagged ``responsive=False`` never reply - this drives the feedback-loop demo.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

import httpx
from fastapi import APIRouter, HTTPException, Request

from .security import verify_signature

log = logging.getLogger("asama.demo")
router = APIRouter(prefix="/demo", tags=["demo"])


@dataclass
class DemoAgent:
    agent_id: str
    api_key: str
    company: str
    reply_template: str
    responsive: bool = True
    base_url: str = "http://asama.local"


SEED_COMPANIES = [
    dict(company="Swiggy", industry="Food Delivery", responsive=True,
         manifesto="Food delivery and quick commerce. Looking for festive campaigns, payment cashback partnerships, restaurant and grocery promotions.",
         reply="Swiggy can co-promote this with a festive banner and bundle it with our delivery offers."),
    dict(company="PhonePe", industry="Payments & Fintech", responsive=True,
         manifesto="Interested in e-commerce promotions, food delivery festive discounts, card payment partnerships, UPI and merchant cashback opportunities.",
         reply="PhonePe will provide extra Rs 50 cashback when users pay via PhonePe UPI on this offer."),
    dict(company="Flipkart", industry="E-commerce", responsive=True,
         manifesto="Online shopping marketplace. Interested in payment offers, bank card discounts, EMI partnerships, festive sale tie-ups, electronics and smartphone promotions.",
         reply="Flipkart can feature this in our festive sale and stack it with partner card discounts."),
    dict(company="HDFC Bank", industry="Banking", responsive=True,
         manifesto="Credit card, EMI and loan products. Seeking merchant partnerships for card cashback, no-cost EMI on electronics, shopping and travel offers.",
         reply="HDFC Bank can add an extra 5% card cashback and no-cost EMI for this campaign."),
    dict(company="Bajaj Auto", industry="Automobile", responsive=False,
         manifesto="Motorcycles and electric vehicles. Interested in vehicle financing, loan and EMI partnerships, EV charging infrastructure and dealer promotions.",
         reply="Bajaj Auto is interested in a financing tie-up."),
    dict(company="Zomato", industry="Food Delivery", responsive=True,
         manifesto="Restaurant discovery and food delivery. Interested in dining offers, bank card discounts, payment cashback and festive restaurant promotions.",
         reply="Zomato can run a matching dining-out offer for the same week."),
    dict(company="MakeMyTrip", industry="Travel", responsive=True,
         manifesto="Flights, hotels and holiday packages. Interested in bank card offers, payment cashback, EMI on holidays and festive travel deals.",
         reply="MakeMyTrip can bundle this into festive holiday packages."),
    dict(company="Ola Electric", industry="Automobile", responsive=False,
         manifesto="Electric scooters and EV charging. Interested in vehicle loan, EMI partnerships and green-mobility campaigns.",
         reply="Ola Electric can offer a financing-linked promotion."),
]


def _registry(request: Request) -> dict[str, DemoAgent]:
    return request.app.state.demo_agents


@router.post("/seed")
async def seed(request: Request):
    """Register the sample companies (idempotent per company name) and return their ids."""
    svc = request.app.state.service
    reg = _registry(request)
    base = str(request.base_url).rstrip("/")
    created = []
    existing = {r["company_name"] for r in svc.db.query("SELECT company_name FROM agents")}
    for spec in SEED_COMPANIES:
        if spec["company"] in existing:
            continue
        # id is only known after registration, but the webhook url needs it -> register then patch
        info = svc.register_agent(spec["company"], spec["industry"], "http://placeholder", spec["manifesto"])
        url = f"{base}/demo/agents/{info['id']}/events"
        svc.db.execute("UPDATE agents SET webhook_url=? WHERE id=?", (url, info["id"]))
        reg[info["id"]] = DemoAgent(info["id"], info["api_key"], spec["company"], spec["reply"], spec["responsive"], base)
        created.append({"id": info["id"], "company": spec["company"], "industry": spec["industry"],
                        "api_key": info["api_key"]})
    return {"created": created, "total_agents": len(svc.db.query("SELECT id FROM agents"))}


@router.post("/agents/{agent_id}/events")
async def receive(agent_id: str, request: Request):
    svc = request.app.state.service
    agent = _registry(request).get(agent_id)
    row = svc.db.one("SELECT webhook_secret FROM agents WHERE id=?", (agent_id,))
    if agent is None or row is None:
        raise HTTPException(404, "unknown demo agent")
    body = await request.body()
    sig = request.headers.get("X-Asama-Signature", "")
    try:
        ts = int(request.headers.get("X-Asama-Timestamp", "0"))
    except ValueError:
        ts = 0
    if not verify_signature(row["webhook_secret"], body, sig, ts):
        raise HTTPException(401, "bad signature")  # never act on unauthenticated events
    event = json.loads(body)
    if agent.responsive:
        task = asyncio.create_task(_reply(request.app, agent, event))
        request.app.state.demo_tasks.add(task)
        task.add_done_callback(request.app.state.demo_tasks.discard)
    return {"ok": True}


async def _reply(app, agent: DemoAgent, event: dict) -> None:
    await asyncio.sleep(0.05)  # "thinking"
    post = event["post"]
    client: httpx.AsyncClient = app.state.http_client
    # The post is data. The reply is composed from the agent's own template, never from post text.
    text = f"[{agent.company}] Re '{post['title']}': {agent.reply_template}"
    try:
        r = await client.post(
            f"{agent.base_url}{event['reply_endpoint']}",
            headers={"X-API-Key": agent.api_key},
            json={"author_agent_id": agent.agent_id, "content": text},
        )
        r.raise_for_status()
    except Exception as exc:
        log.warning("demo agent %s failed to reply: %s", agent.company, exc)
