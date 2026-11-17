"""Safe execution engine.

Security model (defence in depth):
  1. The LLM never produces commands.  It can only name a tool from the registry.
  2. Every command is a fixed argv tuple declared in ``tools.py``; user/LLM text is never
     interpolated into it.  Arguments from the LLM are validated against a schema and are
     used to *filter parsed output*, never to build a command line.
  3. ``subprocess.run(argv, shell=False)`` - no shell, so ``;`` ``&&`` ``|`` have no meaning.
  4. The runner refuses any ``Source`` object that is not part of the registry.
  5. Hard timeout and output-size cap per command; sanitised environment.
  6. Privileged commands use ``sudo -n`` (never prompts) against an exact-argument sudoers rule
     - the assistant itself never runs as root.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

TIMEOUT_S = 12
MAX_OUTPUT = 1_000_000


@dataclass(frozen=True)
class Source:
    """One way to obtain a piece of data.  kind: cmd | file | battery_sysfs | dmi_sysfs."""
    kind: str
    parser: str
    argv: tuple[str, ...] = ()
    path: str = ""
    needs_root: bool = False

    def describe(self) -> str:
        if self.kind != "cmd":
            return self.path or self.kind
        argv = self.argv[:1] if self.argv[0].endswith(("powershell.exe", "powershell")) else self.argv
        return (("sudo -n " if self.needs_root else "") + " ".join(argv))[:80]


@dataclass
class SourceResult:
    ok: bool
    raw: Any = None
    error: str | None = None
    error_kind: str | None = None  # missing | privilege | failed | timeout | unsupported
    source: str = ""


_BATTERY_FILES = ("status", "capacity", "energy_full", "energy_full_design", "charge_full",
                  "charge_full_design", "cycle_count", "model_name", "technology")
_DMI_FILES = ("sys_vendor", "product_name", "bios_version", "bios_date")  # no serial/UUID: privacy


class SafeRunner:
    def __init__(self, platform: str, allowed: set[int],
                 run: Callable[..., Any] = subprocess.run,
                 which: Callable[[str], str | None] = shutil.which,
                 is_root: bool | None = None):
        self.platform = platform
        self._allowed = allowed          # ids of Source objects owned by the registry
        self._run = run
        self._which = which
        self.is_root = (os.geteuid() == 0) if is_root is None and hasattr(os, "geteuid") else bool(is_root)
        self.calls: list[tuple[str, ...]] = []  # audit trail (also used by tests)

    # ------------------------------------------------------------------------- public
    def read(self, src: Source) -> SourceResult:
        if id(src) not in self._allowed:
            return SourceResult(False, error="source is not in the allow-list", error_kind="unsupported")
        try:
            if src.kind == "cmd":
                return self._run_cmd(src)
            if src.kind == "file":
                return self._read_file(src)
            if src.kind == "battery_sysfs":
                return self._read_battery()
            if src.kind == "dmi_sysfs":
                return self._read_dmi()
        except Exception as exc:  # never let a probe crash the assistant
            return SourceResult(False, error=f"{type(exc).__name__}: {exc}", error_kind="failed", source=src.describe())
        return SourceResult(False, error=f"unknown source kind {src.kind}", error_kind="unsupported")

    # ------------------------------------------------------------------------ helpers
    def _resolve(self, name: str) -> str | None:
        if name == "powershell.exe":
            return self._which("powershell") or self._which("pwsh")
        return self._which(name)

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.platform == "linux":
            env.update({"LC_ALL": "C", "LANG": "C"})  # stable, parseable output
        return env

    def _run_cmd(self, src: Source) -> SourceResult:
        exe = self._resolve(src.argv[0])
        label = src.describe()
        if not exe:
            return SourceResult(False, error=f"'{src.argv[0]}' is not installed", error_kind="missing", source=label)
        argv = [exe, *src.argv[1:]]
        if src.needs_root and not self.is_root:
            sudo = self._which("sudo")
            if not sudo:
                return SourceResult(False, error="needs root and sudo is unavailable", error_kind="privilege", source=label)
            argv = [sudo, "-n", *argv]  # -n: fail instead of prompting for a password
        self.calls.append(tuple(argv))
        kwargs: dict[str, Any] = dict(capture_output=True, text=True, encoding="utf-8", errors="replace",
                                      timeout=TIMEOUT_S, shell=False, env=self._env(), stdin=subprocess.DEVNULL)
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = self._run(argv, **kwargs)
        except subprocess.TimeoutExpired:
            return SourceResult(False, error=f"timed out after {TIMEOUT_S}s", error_kind="timeout", source=label)
        except FileNotFoundError:
            return SourceResult(False, error=f"'{src.argv[0]}' not found", error_kind="missing", source=label)
        if proc.returncode != 0:
            err = " ".join((proc.stderr or "").split())[:160]
            kind = "privilege" if src.needs_root or "sudo" in err.lower() or "permission" in err.lower() else "failed"
            return SourceResult(False, error=err or f"exit code {proc.returncode}", error_kind=kind, source=label)
        return SourceResult(True, raw=(proc.stdout or "")[:MAX_OUTPUT], source=label)

    def _read_file(self, src: Source) -> SourceResult:
        try:
            with open(src.path, "r", encoding="utf-8", errors="replace") as fh:
                return SourceResult(True, raw=fh.read(MAX_OUTPUT), source=src.path)
        except FileNotFoundError:
            return SourceResult(False, error=f"{src.path} does not exist", error_kind="missing", source=src.path)
        except PermissionError:
            return SourceResult(False, error=f"no permission to read {src.path}", error_kind="privilege", source=src.path)

    def _read_battery(self) -> SourceResult:
        found: dict[str, dict[str, str]] = {}
        for d in sorted(glob.glob("/sys/class/power_supply/BAT*")):
            vals = {}
            for name in _BATTERY_FILES:
                try:
                    with open(os.path.join(d, name), "r", encoding="utf-8", errors="replace") as fh:
                        vals[name] = fh.read(256).strip()
                except OSError:
                    pass
            found[os.path.basename(d)] = vals
        if not found:
            return SourceResult(False, error="no battery found (desktop or VM?)", error_kind="missing", source="/sys/class/power_supply")
        return SourceResult(True, raw=found, source="/sys/class/power_supply/BAT*")

    def _read_dmi(self) -> SourceResult:
        vals = {}
        for name in _DMI_FILES:
            try:
                with open(f"/sys/class/dmi/id/{name}", "r", encoding="utf-8", errors="replace") as fh:
                    vals[name] = fh.read(256).strip()
            except OSError:
                pass
        if not vals:
            return SourceResult(False, error="DMI info not exposed (VM or container?)", error_kind="missing", source="/sys/class/dmi/id")
        return SourceResult(True, raw=vals, source="/sys/class/dmi/id")


# ---------------------------------------------------------------------- parameter validation
class ParamError(ValueError):
    pass


_NAME_RE = r"^[A-Za-z0-9._-]{1,40}$"


def validate_args(schema: dict[str, dict], args: dict | None) -> dict:
    """Strict validation: unknown keys rejected, types enforced, strings regex-constrained."""
    import re
    args = args or {}
    if not isinstance(args, dict):
        raise ParamError("arguments must be an object")
    unknown = set(args) - set(schema)
    if unknown:
        raise ParamError(f"unknown argument(s): {', '.join(sorted(map(str, unknown)))}")
    out: dict[str, Any] = {}
    for key, spec in schema.items():
        if key not in args:
            if "default" in spec:
                out[key] = spec["default"]
            continue
        val = args[key]
        t = spec["type"]
        if t == "boolean" and not isinstance(val, bool):
            raise ParamError(f"{key} must be true or false")
        if t == "string":
            if not isinstance(val, str) or not re.match(spec.get("pattern", _NAME_RE), val):
                raise ParamError(f"{key} has an invalid value")
        if "enum" in spec and val not in spec["enum"]:
            raise ParamError(f"{key} must be one of {spec['enum']}")
        out[key] = val
    return out
