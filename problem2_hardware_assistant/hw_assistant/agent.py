"""Conversation orchestrator: route -> execute (cached) -> answer -> verify."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable

from . import llm
from .parsers import clean
from .render import render_answer
from .tools import REGISTRY, ToolExecutor

SCOPE_HELP = ("I can answer questions about this computer's hardware: RAM and upgrades, CPU, storage, GPU, battery and "
              "system info. For example: \"Can I upgrade my RAM, and by how much?\" or \"What GPU do I have?\"")


@dataclass
class Reply:
    text: str
    tools: list[str] = field(default_factory=list)
    mode: str = "deterministic"          # deterministic | llm-verified | llm-rejected-fallback | refusal
    facts: list[dict] = field(default_factory=list)
    cached: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class HardwareAssistant:
    def __init__(self, executor: ToolExecutor | None = None, provider: llm.Provider | None = None,
                 clock: Callable[[], float] = time.monotonic, history_turns: int = 6):
        self.executor = executor or ToolExecutor()
        self.provider = provider
        self._clock = clock
        self._cache: dict[str, tuple[float, dict]] = {}
        self.history: list[tuple[str, str]] = []
        self.last_tools: list[str] = []
        self._max_hist = history_turns

    # ------------------------------------------------------------------ public API
    def ask(self, question: str) -> Reply:
        question = clean(question, 500)
        if not question:
            return Reply("Please type a question about your hardware.", mode="refusal")
        notes: list[str] = []
        calls = self._route(question, notes)
        if not calls:
            self._remember(question, SCOPE_HELP)
            return Reply(SCOPE_HELP, mode="refusal", notes=notes)

        results, cached = [], []
        for name, args in calls:
            res, hit = self._run_cached(name, args)
            results.append(res)
            if hit:
                cached.append(name)
        self.last_tools = [n for n, _ in calls]

        text, mode = self._answer(question, results, notes)
        self._remember(question, text)
        return Reply(text, tools=self.last_tools, mode=mode, facts=results, cached=cached, notes=notes)

    # ------------------------------------------------------------------- routing
    def _route(self, question: str, notes: list[str]) -> list[tuple[str, dict]]:
        if self.provider:
            try:
                user = json.dumps({"question": question, "available_tools": self.executor.tool_schemas(),
                                   "already_gathered": [n for n in self._cache], "recent_history": self.history[-3:]})
                calls, out_of_scope = llm.parse_router_reply(self.provider.chat(llm.ROUTER_SYSTEM, user, json_mode=True))
                if calls:
                    return calls
                if out_of_scope:
                    return []
                notes.append("LLM router chose no valid tool; used keyword router")
            except Exception as exc:
                notes.append(f"LLM routing failed ({type(exc).__name__}); used keyword router")
        names = llm.keyword_route(question, self.last_tools)
        if names is None:  # follow-up with no topic words: reuse the previous tools
            names = self.last_tools
            notes.append("follow-up: reusing previous tools")
        return [(n, {}) for n in names]

    # ------------------------------------------------------------------ execution
    def _run_cached(self, name: str, args: dict) -> tuple[dict, bool]:
        key = name + json.dumps(args, sort_keys=True)
        now = self._clock()
        hit = self._cache.get(key)
        if hit and now - hit[0] < REGISTRY[name].ttl_s and "error" not in hit[1]:
            return hit[1], True
        res = self.executor.execute(name, args)
        self._cache[key] = (now, res)
        return res, False

    # ------------------------------------------------------------------- answering
    def _answer(self, question: str, results: list[dict], notes: list[str]) -> tuple[str, str]:
        fallback = render_answer(results)
        if not self.provider or all("error" in r for r in results):
            return fallback, "deterministic"
        facts = [r for r in results]
        try:
            user = json.dumps({"QUESTION": question, "FACTS": facts})
            answer = self.provider.chat(llm.ANSWER_SYSTEM, user).strip()
        except Exception as exc:
            notes.append(f"LLM answer failed ({type(exc).__name__}); used verified template answer")
            return fallback, "deterministic"
        bad = llm.ungrounded_numbers(answer, facts)
        if bad or not answer:
            notes.append(f"LLM answer rejected: numbers not found in system data: {bad[:4]}")
            return fallback, "llm-rejected-fallback"
        return answer, "llm-verified"

    def _remember(self, q: str, a: str) -> None:
        self.history.append((q, a[:300]))
        self.history = self.history[-self._max_hist:]
