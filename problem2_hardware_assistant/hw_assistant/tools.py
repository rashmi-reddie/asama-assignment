"""The tool registry: the *only* things the assistant can do.

Each tool declares, per platform, an ordered list of fallback ``Source`` alternatives for each
probe, a parser, and an analyzer that derives facts deterministically (so the LLM never has to
do arithmetic).  ``cannot_determine`` lists things the OS genuinely cannot tell us, which are
handed to the LLM so it says "I can't determine that" instead of guessing.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from . import parsers as P
from .runner import SafeRunner, Source, SourceResult, ParamError, validate_args

PLATFORMS = ("linux", "windows")


def current_platform() -> str:
    return "windows" if sys.platform == "win32" else "linux"


@dataclass(frozen=True)
class Probe:
    key: str
    alternatives: tuple[Source, ...]
    required: bool = True


@dataclass
class ToolSpec:
    name: str
    description: str
    cannot_determine: list[str]
    probes: dict[str, tuple[Probe, ...]]               # platform -> probes
    analyze: Callable[[dict[str, Any], dict], dict]    # (parsed probes, validated args) -> data
    params: dict[str, dict] = field(default_factory=dict)
    ttl_s: int = 600


# ------------------------------------------------------------------------ PowerShell scripts
# Fixed constants.  Nothing is ever interpolated into them.
def _ps(script: str) -> Source:
    return Source("cmd", "win_json", ("powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script))


PS_MEMORY = (
    "$ErrorActionPreference='SilentlyContinue';"
    "@{array=@(Get-CimInstance Win32_PhysicalMemoryArray | Select-Object MaxCapacity,MaxCapacityEx,MemoryDevices);"
    "modules=@(Get-CimInstance Win32_PhysicalMemory | Select-Object DeviceLocator,BankLabel,Capacity,Speed,ConfiguredClockSpeed,Manufacturer,PartNumber,SMBIOSMemoryType,FormFactor);"
    "os=(Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory)}"
    " | ConvertTo-Json -Depth 4 -Compress"
)
PS_CPU = ("Get-CimInstance Win32_Processor | Select-Object Name,Manufacturer,Architecture,NumberOfCores,"
          "NumberOfLogicalProcessors,MaxClockSpeed,L2CacheSize,L3CacheSize,VirtualizationFirmwareEnabled | ConvertTo-Json -Compress")
PS_STORAGE = (
    "$ErrorActionPreference='SilentlyContinue';"
    "@{disks=@(Get-PhysicalDisk | Select-Object DeviceId,FriendlyName,MediaType,BusType,Size,HealthStatus);"
    "volumes=@(Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,Size,FreeSpace,FileSystem)}"
    " | ConvertTo-Json -Depth 4 -Compress"
)
PS_GPU = (
    "$ErrorActionPreference='SilentlyContinue';"
    "$nv=''; if (Get-Command nvidia-smi) { $nv=(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits) -join \"`n\" };"
    "@{controllers=@(Get-CimInstance Win32_VideoController | Select-Object Name,AdapterCompatibility,AdapterRAM,DriverVersion);"
    "nvidia_csv=$nv} | ConvertTo-Json -Depth 4 -Compress"
)
PS_OVERVIEW = (
    "$ErrorActionPreference='SilentlyContinue';"
    "@{cs=(Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model);"
    "os=(Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,OSArchitecture);"
    "bios=(Get-CimInstance Win32_BIOS | Select-Object SMBIOSBIOSVersion,@{n='ReleaseDate';e={$_.ReleaseDate.ToString('yyyy-MM-dd')}})}"
    " | ConvertTo-Json -Depth 4 -Compress"
)
PS_BATTERY = (
    "$ErrorActionPreference='SilentlyContinue';"
    "@{battery=@(Get-CimInstance Win32_Battery | Select-Object Name,BatteryStatus,EstimatedChargeRemaining);"
    "full=@(Get-CimInstance -Namespace root/WMI -ClassName BatteryFullChargedCapacity | Select-Object FullChargedCapacity);"
    "design=@(Get-CimInstance -Namespace root/WMI -ClassName BatteryStaticData | Select-Object DesignedCapacity)}"
    " | ConvertTo-Json -Depth 4 -Compress"
)


# --------------------------------------------------------------------------------- analyzers
def _slots_summary(devices: list[dict], arrays: list[dict]) -> dict:
    total_slots = sum(a["slots"] for a in arrays if a.get("slots")) or (len(devices) or None)
    installed = [d for d in devices if d["installed"]]
    return {"total_slots": total_slots, "used_slots": len(installed),
            "free_slots": (total_slots - len(installed)) if total_slots is not None else None,
            "installed": installed}


def analyze_memory(p: dict[str, Any], args: dict) -> dict:
    mod = p.get("modules") or {}
    usage = p.get("usage") or mod.get("usage") or {}
    data: dict[str, Any] = {}
    if "devices" in mod:  # full SMBIOS view (dmidecode / WMI)
        arrays, devices = mod["arrays"], mod["devices"]
        s = _slots_summary(devices, arrays)
        installed = s["installed"]
        total_bytes = sum(d["size_bytes"] for d in installed)
        max_bytes = max((a["max_capacity_bytes"] or 0 for a in arrays), default=0) or None
        types = {d["type"] for d in installed if d.get("type")}
        forms = {d["form_factor"] for d in installed if d.get("form_factor")}
        soldered = bool(
            any(t and t.startswith("LPDDR") for t in types)
            or forms & {"BGA", "FPBGA", "LGA", "Row Of Chips", "Die", "SMD"}
        )
        data.update({
            "installed_total_gb": P.gb(total_bytes), "total_slots": s["total_slots"], "used_slots": s["used_slots"],
            "free_slots": s["free_slots"], "max_supported_gb": P.gb(max_bytes),
            "upgrade_headroom_gb": P.gb(max_bytes - total_bytes) if max_bytes is not None else None,
            "max_per_slot_gb_if_slots_equal": P.gb(max_bytes / s["total_slots"]) if max_bytes and s["total_slots"] else None,
            "memory_type": sorted(types)[0] if len(types) == 1 else (sorted(types) or None),
            "form_factor": sorted(forms)[0] if len(forms) == 1 else (sorted(forms) or None),
            "speed_mts": max((d["speed_mts"] or 0 for d in installed), default=0) or None,
            "likely_soldered": soldered,
            "modules": [{"slot": d["slot"], "size_gb": P.gb(d["size_bytes"]), "type": d["type"],
                         "speed_mts": d["speed_mts"], "manufacturer": d["manufacturer"]} for d in installed],
            "empty_slots": [d["slot"] for d in devices if not d["installed"]] if args.get("include_empty_slots", True) else "omitted",
        })
        if max_bytes and total_bytes and max_bytes < total_bytes:
            data["firmware_inconsistent_max"] = True
        # deterministic upgrade verdict the LLM must base its advice on
        if soldered:
            verdict = "likely_soldered"
        elif data["free_slots"] and data["upgrade_headroom_gb"] and data["upgrade_headroom_gb"] > 0:
            verdict = "add_module_in_free_slot"
        elif data["free_slots"] == 0 and data["upgrade_headroom_gb"] and data["upgrade_headroom_gb"] > 0:
            verdict = "replace_modules_with_larger"
        elif data["upgrade_headroom_gb"] == 0:
            verdict = "already_at_maximum"
        else:
            verdict = "unknown_limits"
        data["upgrade_verdict"] = verdict
    else:  # fallback: only totals are known
        data["installed_total_gb"] = P.gb(mod.get("total_bytes") or usage.get("total_bytes"))
        data["upgrade_verdict"] = "unknown_limits"
    if usage:
        data["in_use_gb"] = P.gb(usage.get("used_bytes"))
        data["available_gb"] = P.gb(usage.get("available_bytes"))
    return data


def analyze_passthrough(p: dict[str, Any], args: dict) -> dict:
    merged: dict[str, Any] = {}
    for v in p.values():
        if isinstance(v, dict):
            merged.update(v)
    return merged


def analyze_storage(p: dict[str, Any], args: dict) -> dict:
    disks = list((p.get("disks") or {}).get("disks", []))
    vols = list((p.get("usage") or {}).get("volumes", []))
    if not vols:
        vols = list((p.get("disks") or {}).get("volumes", []))
    want = args.get("device_name")
    if want:  # filter the *parsed* output; the value never reaches a command line
        disks = [d for d in disks if want.lower() in (d.get("name") or "").lower() or want.lower() in (d.get("model") or "").lower()]
    total = sum(d["size_bytes"] for d in disks)
    return {"drive_count": len(disks), "total_capacity_gb": P.gb(total, 1), "drives": disks, "volumes": vols}


def analyze_gpu(p: dict[str, Any], args: dict) -> dict:
    base = p.get("gpus") or {}
    gpus = list(base.get("gpus", []))
    nv = list((p.get("nvidia") or {}).get("nvidia", [])) or list(base.get("nvidia", []))
    by_name = {g["name"].lower(): g for g in nv}
    for g in gpus:  # enrich lspci/WMI entries with accurate VRAM from nvidia-smi when possible
        for n, info in by_name.items():
            if n in g["name"].lower() or g["name"].lower() in n or "nvidia" in g["name"].lower():
                g["vram_mb"] = info["vram_mb"]
                g["driver_version"] = info["driver_version"]
                g.pop("reported_vram_mb", None); g.pop("vram_may_be_capped", None)
                break
    if not gpus and nv:
        gpus = [{"name": n["name"], "vram_mb": n["vram_mb"], "driver_version": n["driver_version"]} for n in nv]
    return {"gpu_count": len(gpus), "gpus": gpus}


def analyze_battery(p: dict[str, Any], args: dict) -> dict:
    b = (p.get("battery") or {}).get("batteries", [])
    return {"battery_present": bool(b), "batteries": b}


# --------------------------------------------------------------------------------- registry
def _build() -> dict[str, ToolSpec]:
    L = {  # ---- linux sources
        "lscpu": Source("cmd", "lscpu", ("lscpu",)),
        "dmidecode_mem": Source("cmd", "dmidecode_memory", ("dmidecode", "-t", "memory"), needs_root=True),
        "meminfo": Source("file", "meminfo", path="/proc/meminfo"),
        "free": Source("cmd", "free_b", ("free", "-b")),
        "lsblk": Source("cmd", "lsblk_json", ("lsblk", "-J", "-b", "-o", "NAME,SIZE,TYPE,MOUNTPOINT,MODEL,ROTA,TRAN,FSTYPE")),
        "df": Source("cmd", "df_posix", ("df", "-P", "-B1", "-x", "tmpfs", "-x", "devtmpfs", "-x", "squashfs", "-x", "overlay")),
        "lspci": Source("cmd", "lspci_mm", ("lspci", "-mm")),
        "nvidia": Source("cmd", "nvidia_csv", ("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits")),
        "osrel": Source("file", "os_release", path="/etc/os-release"),
        "uname": Source("cmd", "uname_r", ("uname", "-r")),
        "dmi": Source("dmi_sysfs", "dmi_files"),
        "battery": Source("battery_sysfs", "battery_files"),
    }
    W = {  # ---- windows sources
        "mem": _ps(PS_MEMORY), "cpu": _ps(PS_CPU), "storage": _ps(PS_STORAGE),
        "gpu": _ps(PS_GPU), "overview": _ps(PS_OVERVIEW), "battery": _ps(PS_BATTERY),
    }
    tools = [
        ToolSpec(
            "get_system_overview", "Laptop manufacturer/model, operating system and BIOS version.",
            ["serial number, hostname and user identity (deliberately not collected)", "warranty status"],
            {"linux": (Probe("dmi", (L["dmi"],), False), Probe("os", (L["osrel"],)), Probe("kernel", (L["uname"],), False)),
             "windows": (Probe("ov", (W["overview"],)),)},
            analyze_passthrough),
        ToolSpec(
            "get_cpu_info", "CPU model, core/thread counts, clock speeds and cache.",
            ["whether the CPU can be upgraded (usually soldered on laptops)", "current CPU temperature or load"],
            {"linux": (Probe("cpu", (L["lscpu"],)),), "windows": (Probe("cpu", (W["cpu"],)),)},
            analyze_passthrough),
        ToolSpec(
            "get_memory_info",
            "Installed RAM, memory slots, free slots, maximum supported capacity, memory type/speed, current usage and a deterministic upgrade verdict.",
            ["exact compatible module part numbers", "per-slot maximum when slots are not identical (the value given assumes equal slots)",
             "whether the BIOS/firmware has additional undocumented limits"],
            {"linux": (Probe("modules", (L["dmidecode_mem"], L["meminfo"])), Probe("usage", (L["free"],), False)),
             "windows": (Probe("modules", (W["mem"],)),)},
            analyze_memory,
            params={"include_empty_slots": {"type": "boolean", "default": True}}, ttl_s=30),
        ToolSpec(
            "get_storage_info", "Physical drives (model, size, SSD/NVMe/HDD, interface) and volume usage / free space.",
            ["how many free M.2 or SATA slots exist inside the laptop", "whether the drive is soldered or replaceable"],
            {"linux": (Probe("disks", (L["lsblk"],)), Probe("usage", (L["df"],), False)),
             "windows": (Probe("disks", (W["storage"],)),)},
            analyze_storage,
            params={"device_name": {"type": "string", "pattern": r"^[A-Za-z0-9._ -]{1,40}$"}}, ttl_s=60),
        ToolSpec(
            "get_gpu_info", "Graphics adapters (integrated and discrete), VRAM and driver version where available.",
            ["current GPU temperature or utilisation", "whether the GPU can be upgraded (almost never on laptops)"],
            {"linux": (Probe("gpus", (L["lspci"],)), Probe("nvidia", (L["nvidia"],), False)),
             "windows": (Probe("gpus", (W["gpu"],)),)},
            analyze_gpu),
        ToolSpec(
            "get_battery_info", "Battery presence, charge level, status and (where reported) health versus design capacity.",
            ["remaining runtime in minutes", "whether a replacement battery is available"],
            {"linux": (Probe("battery", (L["battery"],)),), "windows": (Probe("battery", (W["battery"],)),)},
            analyze_battery, ttl_s=20),
    ]
    return {t.name: t for t in tools}


REGISTRY: dict[str, ToolSpec] = _build()
TOOL_NAMES = list(REGISTRY)


def all_sources() -> list[Source]:
    out = []
    for t in REGISTRY.values():
        for probes in t.probes.values():
            for pr in probes:
                out.extend(pr.alternatives)
    return out


_PARSERS: dict[str, Callable[[Any], Any]] = {
    "lscpu": P.parse_lscpu, "dmidecode_memory": P.parse_dmidecode_memory, "meminfo": P.parse_meminfo,
    "free_b": P.parse_free_b, "lsblk_json": P.parse_lsblk_json, "df_posix": P.parse_df_posix,
    "lspci_mm": P.parse_lspci_mm, "nvidia_csv": P.parse_nvidia_smi_csv, "os_release": P.parse_os_release,
    "uname_r": P.parse_uname_r, "dmi_files": P.parse_dmi_files, "battery_files": P.parse_battery_files,
    "win_json": None,  # decided per tool below
}
_WIN_PARSERS = {
    "get_memory_info": P.parse_win_memory, "get_cpu_info": P.parse_win_cpu, "get_storage_info": P.parse_win_storage,
    "get_gpu_info": P.parse_win_gpu, "get_system_overview": P.parse_win_overview, "get_battery_info": P.parse_win_battery,
}


class ToolExecutor:
    """Validates a tool call, runs its probes with fallbacks, parses and analyses the result."""

    def __init__(self, runner: SafeRunner | None = None, platform: str | None = None):
        self.platform = platform or current_platform()
        self.runner = runner or SafeRunner(self.platform, {id(s) for s in all_sources()})

    def tool_schemas(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "parameters": t.params} for t in REGISTRY.values()]

    def execute(self, name: str, args: dict | None = None) -> dict:
        spec = REGISTRY.get(name)
        if spec is None:
            return {"tool": name, "error": "unknown tool (not in the allow-list)"}
        try:
            clean_args = validate_args(spec.params, args)
        except ParamError as exc:
            return {"tool": name, "error": f"invalid arguments: {exc}"}
        probes = spec.probes.get(self.platform)
        if not probes:
            return {"tool": name, "error": f"not supported on {self.platform}"}

        parsed: dict[str, Any] = {}
        sources, warnings = [], []
        for probe in probes:
            got = None
            for i, src in enumerate(probe.alternatives):
                res = self.runner.read(src)
                if res.ok:
                    try:
                        got = self._parse(spec, src, res.raw)
                    except Exception as exc:  # malformed output on an unusual distro
                        warnings.append(f"could not parse output of {res.source}: {type(exc).__name__}")
                        continue
                    sources.append(res.source)
                    if i > 0:
                        warnings.append(f"used fallback source {res.source}; some detail is unavailable")
                    break
                msg = {"privilege": "needs elevated privileges (see README: scoped sudoers rule)",
                       "missing": "is not available on this system"}.get(res.error_kind or "", res.error or "failed")
                warnings.append(f"{res.source} {msg}")
            if got is not None:
                parsed[probe.key] = got
            elif probe.required:
                return {"tool": name, "error": "no data source worked", "warnings": warnings}
        data = spec.analyze(parsed, clean_args)
        if not any(v for v in data.values() if v not in (None, [], {}, False, 0)) and name != "get_battery_info":
            return {"tool": name, "error": "no usable data returned", "warnings": warnings}
        return {"tool": name, "platform": self.platform, "source": sources, "warnings": _dedupe(warnings),
                "data": data, "cannot_determine": spec.cannot_determine}

    def _parse(self, spec: ToolSpec, src: Source, raw: Any) -> Any:
        if src.parser == "win_json":
            return _WIN_PARSERS[spec.name](P.parse_win_json(raw))
        return _PARSERS[src.parser](raw)


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for i in items:
        if i not in seen:
            seen.add(i); out.append(i)
    return out


def sudoers_lines(user: str, which=None) -> list[str]:
    """Exact-argument sudoers rules for every privileged source (tighter than 'NOPASSWD: /usr/sbin/dmidecode')."""
    import shutil
    which = which or shutil.which
    lines = []
    for s in all_sources():
        if s.kind == "cmd" and s.needs_root:
            exe = which(s.argv[0]) or f"/usr/sbin/{s.argv[0]}"
            lines.append(f"{user} ALL=(root) NOPASSWD: {exe} {' '.join(s.argv[1:])}".rstrip())
    return lines
