# Problem 2 - Local Hardware Q&A Assistant

A chat assistant that answers questions about the laptop it runs on ("Can I upgrade my RAM, and by how much?")
**without hallucinating**: the LLM never runs commands and never does arithmetic. It can only pick from a fixed set of tools;
their output is parsed into JSON, derived facts are computed in code, and the answer must be grounded in that JSON.

* **Zero dependencies** - Python 3.10+ standard library only. Works on **Windows** (PowerShell/CIM backend) and **Linux** (lscpu/lsblk/dmidecode backend).
* **Works with no API key** - deterministic mode answers from templates. Optionally plug in a free LLM (Ollama / Groq / Gemini).

## Run it (Windows)

```powershell
cd problem2_hardware_assistant
python -m hw_assistant --doctor            # runs every tool on your laptop, shows OK/FAIL
python -m hw_assistant                     # interactive chat
python -m hw_assistant --ask "Can I upgrade my RAM, and by how much?" --debug --json
python -m pytest -q tests                  # 46 tests (pip install pytest)
```

Optional free LLM (otherwise it runs offline/deterministic):

```powershell
# A) fully local + private: install Ollama (ollama.com), then
ollama pull llama3.2:3b ;  python -m hw_assistant --provider ollama
# B) Groq free key:    $env:GROQ_API_KEY="..."    ; python -m hw_assistant --provider groq
# C) Gemini free key:  $env:GEMINI_API_KEY="..."  ; python -m hw_assistant --provider gemini
# Model names change over time; override with --model <name> (or HW_LLM_MODEL) if a default is retired.
```

Linux: `python3 -m hw_assistant --doctor`. For slot/max-capacity data `dmidecode` needs root; run
`python3 -m hw_assistant --print-sudoers` and install the single printed line with `sudo visudo -f /etc/sudoers.d/hw-assistant`.
Without it the assistant degrades gracefully and says which detail is missing.

## Architecture

```
question -> [Router: LLM picks tool NAMES (validated vs allow-list)  |  keyword router when no LLM]
         -> [ToolExecutor: validate args -> run fixed argv, shell=False, timeout -> parse -> derive facts]   (cached per tool, TTL)
         -> [Answer: LLM sees only the JSON  |  deterministic renderer]
         -> [Grounding check: every number in an LLM answer must exist in the JSON, else replaced by the deterministic answer]
```

| File | Role |
|---|---|
| `tools.py` | Tool registry. Per platform: ordered fallback sources per probe, parser, analyzer, `cannot_determine` list |
| `runner.py` | Safe execution: allow-list identity check, `shell=False`, `sudo -n`, timeout, output cap, strict arg validation |
| `parsers.py` | Pure functions (testable with captured output); sanitises every device-supplied string |
| `agent.py` | Conversation state, tool-result cache, routing, verification |
| `llm.py` | Ollama/Groq/Gemini via urllib, router-reply validation, grounding check |
| `render.py` | Deterministic answers (also the fallback when the LLM misbehaves) |

## Security & correctness decisions (talking points)

1. **No LLM-generated commands.** Tools are names; commands are fixed argv tuples (PowerShell scripts are module constants). Tool arguments (e.g. `device_name`) are regex-validated and only *filter parsed output* - they never reach a command line (tested with `; rm -rf /`, `$(...)`, backticks, `../`).
2. **No shell** (`shell=False`) so `; && |` are inert. Runner also refuses any `Source` object that is not owned by the registry.
3. **Privilege:** assistant never runs as root. `dmidecode` runs through `sudo -n` against an **exact-argument** sudoers rule (`... NOPASSWD: /usr/sbin/dmidecode -t memory`) - tighter than whitelisting the binary. Windows queries need no admin.
4. **Anti-hallucination, three layers:** (a) system prompt restricts to JSON; (b) upgrade logic (free slots, headroom, soldered detection, `upgrade_verdict`) is computed in code; (c) grounding check rejects answers containing numbers not in the data. `cannot_determine` lists (e.g. "free M.2 slots") are passed to the LLM so unknowns are stated, not guessed.
5. **Fallbacks:** dmidecode -> /proc/meminfo; lspci -> missing-data message; WMI 4 GB VRAM cap detected and overridden by `nvidia-smi` when present.
6. **Prompt-injection hygiene:** device strings are stripped of control characters and length-capped; FACTS/QUESTION are labelled as data.
7. **Privacy:** serial numbers, hostname, MAC/IP are deliberately never collected; with Ollama nothing leaves the laptop.
8. **Multi-turn:** results cached with per-tool TTL (RAM 30 s, battery 20 s, CPU/GPU 10 min); follow-ups like "what about the GPU?" reuse context.

## Honest limitations

* Developed and live-tested on Linux. The **Windows backend is tested against realistic captured-style JSON fixtures, not on a real Windows machine** - run `--doctor` on your laptop first; if a tool shows FAIL, the warning says which query failed.
* A laptop's *physical* upgradeability (free M.2 slot, max per-slot size) is not exposed by the OS; the assistant says so rather than guessing.
* Free-tier model names/limits change; offline mode is the guaranteed path.
