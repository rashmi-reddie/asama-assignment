"""Command-line interface.  Run:  python -m hw_assistant   (add --help for options)."""
from __future__ import annotations

import argparse
import getpass
import json
import sys

from . import llm
from .agent import HardwareAssistant
from .tools import TOOL_NAMES, ToolExecutor, current_platform, sudoers_lines

BANNER = """Hardware Q&A Assistant  |  platform: {platform}  |  brain: {brain}
Ask about RAM/upgrades, CPU, storage, GPU, battery, system info.  Commands: /tools  /raw  /quit
"""


def _utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles
        except Exception:
            pass


def doctor(ex: ToolExecutor) -> int:
    """Run every tool once and report; useful to verify a new machine."""
    print(f"platform: {ex.platform}\n")
    bad = 0
    for name in TOOL_NAMES:
        r = ex.execute(name)
        ok = "error" not in r
        bad += not ok
        print(f"[{'OK ' if ok else 'FAIL'}] {name}")
        for w in r.get("warnings", []):
            print(f"       warning: {w}")
        if not ok:
            print(f"       error: {r['error']}")
    print(f"\n{len(TOOL_NAMES) - bad}/{len(TOOL_NAMES)} tools working")
    return 0 if not bad else 1


def main(argv: list[str] | None = None) -> int:
    _utf8()
    ap = argparse.ArgumentParser(prog="hw_assistant", description="Ask questions about this computer's hardware.")
    ap.add_argument("--ask", help="ask one question and exit")
    ap.add_argument("--provider", default="auto", choices=["auto", "ollama", "groq", "gemini", "offline"],
                    help="LLM backend (auto: GROQ_API_KEY, GEMINI_API_KEY, local Ollama, else offline)")
    ap.add_argument("--model", help="override the provider's model name")
    ap.add_argument("--json", action="store_true", help="also print the raw tool JSON used for the answer")
    ap.add_argument("--debug", action="store_true", help="print routing notes and verification mode")
    ap.add_argument("--doctor", action="store_true", help="test every tool on this machine and exit")
    ap.add_argument("--print-sudoers", action="store_true", help="print the exact sudoers rule(s) needed on Linux")
    args = ap.parse_args(argv)

    if args.print_sudoers:
        for line in sudoers_lines(getpass.getuser()):
            print(line)
        print("\n# Save as /etc/sudoers.d/hw-assistant (use: sudo visudo -f /etc/sudoers.d/hw-assistant)", file=sys.stderr)
        return 0

    ex = ToolExecutor()
    if args.doctor:
        return doctor(ex)

    try:
        provider = llm.build_provider(args.provider, args.model)
    except llm.LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    bot = HardwareAssistant(ex, provider)
    brain = provider.name if provider else "offline (deterministic, no LLM)"

    def show(q: str) -> None:
        r = bot.ask(q)
        print(f"\n{r.text}\n")
        if args.debug:
            print(f"  [mode={r.mode} tools={r.tools} cached={r.cached} notes={r.notes}]\n")
        if args.json:
            print(json.dumps(r.facts, indent=2)[:6000])

    if args.ask:
        show(args.ask)
        return 0

    print(BANNER.format(platform=current_platform(), brain=brain))
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if q in ("/quit", "/exit", "quit", "exit"):
            return 0
        if q == "/tools":
            print("\n".join(f"  {t['name']}: {t['description']}" for t in ex.tool_schemas()))
            continue
        if q == "/raw":
            print(json.dumps(bot._cache and [v[1] for v in bot._cache.values()], indent=2)[:6000])
            continue
        if q:
            show(q)


if __name__ == "__main__":
    raise SystemExit(main())
