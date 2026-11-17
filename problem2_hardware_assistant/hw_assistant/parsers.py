"""Pure parsing functions: raw command output -> plain dicts.

Everything here is side-effect free so it can be unit-tested with captured fixtures from any
machine.  All strings that originate from the device (model names, manufacturers, ...) pass
through :func:`clean` - they are *untrusted data* and are later handed to an LLM.
"""
from __future__ import annotations

import json
import re
import shlex
from typing import Any

_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
GIB = 1024 ** 3
MIB = 1024 ** 2


def clean(value: Any, limit: int = 120) -> str:
    """Strip control characters, collapse whitespace and cap length (injection hygiene)."""
    s = _CTRL.sub("", str(value if value is not None else ""))
    s = " ".join(s.split())
    return s[:limit]


_UNITS = {"b": 1, "kb": 1024, "kib": 1024, "mb": MIB, "mib": MIB, "gb": GIB, "gib": GIB,
          "tb": 1024 ** 4, "tib": 1024 ** 4}


def parse_size_to_bytes(text: str | None) -> int | None:
    """'8192 MB' / '16 GB' / '16384 kB' -> bytes.  Returns None for 'No Module Installed' etc."""
    if not text:
        return None
    m = re.match(r"^\s*([\d.]+)\s*([A-Za-z]+)\s*$", str(text))
    if not m:
        return None
    unit = _UNITS.get(m.group(2).lower())
    if unit is None:
        return None
    return int(float(m.group(1)) * unit)


def gb(n_bytes: int | float | None, digits: int = 2) -> float | None:
    return None if n_bytes is None else round(n_bytes / GIB, digits)


def _kv(text: str, sep: str = ":") -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if sep in line:
            k, v = line.split(sep, 1)
            out.setdefault(k.strip(), v.strip())  # first occurrence wins (hybrid CPUs repeat keys)
    return out


def _to_int(v: Any) -> int | None:
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------- Linux: CPU
def parse_lscpu(text: str) -> dict:
    kv = _kv(text)
    sockets = _to_int(kv.get("Socket(s)"))
    cores_per_socket = _to_int(kv.get("Core(s) per socket"))
    physical = sockets * cores_per_socket if sockets and cores_per_socket else None
    return {
        "model": clean(kv.get("Model name") or kv.get("Model") or "") or None,
        "vendor": clean(kv.get("Vendor ID", "")) or None,
        "architecture": clean(kv.get("Architecture", "")) or None,
        "logical_processors": _to_int(kv.get("CPU(s)")),
        "physical_cores": physical,
        "threads_per_core": _to_int(kv.get("Thread(s) per core")),
        "sockets": sockets,
        "max_mhz": round(float(kv["CPU max MHz"]), 1) if kv.get("CPU max MHz") else None,
        "min_mhz": round(float(kv["CPU min MHz"]), 1) if kv.get("CPU min MHz") else None,
        "l2_cache": clean(kv.get("L2 cache", "")) or None,
        "l3_cache": clean(kv.get("L3 cache", "")) or None,
        "virtualization": clean(kv.get("Virtualization", "")) or None,
        "is_virtual_machine": bool(kv.get("Hypervisor vendor")),
    }


# -------------------------------------------------------------------------- Linux: storage
_PSEUDO_DISK = re.compile(r"^(zram|loop|ram|sr|fd|dm-)\d*")


def _truthy(v: Any) -> bool:
    return v is True or str(v).lower() in ("1", "true", "yes")


def parse_lsblk_json(text: str) -> dict:
    data = json.loads(text)
    disks = []
    for dev in data.get("blockdevices", []):
        if dev.get("type") != "disk" or _PSEUDO_DISK.match(str(dev.get("name", ""))):
            continue
        size = _to_int(dev.get("size")) or 0
        if size < GIB:  # ignore pseudo/firmware devices smaller than 1 GiB
            continue
        tran = clean(dev.get("tran") or "").lower() or None
        if tran == "nvme" or str(dev.get("name", "")).startswith("nvme"):
            kind = "NVMe SSD"
        elif not _truthy(dev.get("rota")):
            kind = "SSD"
        else:
            kind = "HDD"  # also what virtual disks report
        parts = []
        for ch in dev.get("children", []) or []:
            mp = ch.get("mountpoint") or next(iter(ch.get("mountpoints") or []), None)
            parts.append({"name": clean(ch.get("name")), "size_bytes": _to_int(ch.get("size")),
                          "filesystem": clean(ch.get("fstype") or "") or None,
                          "mountpoint": clean(mp or "") or None})
        disks.append({"name": clean(dev.get("name")), "model": clean(dev.get("model") or "") or None,
                      "size_bytes": size, "size_gb": gb(size, 1), "type": kind,
                      "interface": tran, "partitions": parts})
    return {"disks": disks}


_PSEUDO_FS = {"tmpfs", "devtmpfs", "overlay", "squashfs", "efivarfs", "udev", "none"}


def parse_df_posix(text: str) -> dict:
    """`df -P -B1` -> volumes with used/free."""
    vols = []
    for line in text.splitlines()[1:]:
        parts = line.split(None, 5)
        if len(parts) < 6 or parts[0] in _PSEUDO_FS:
            continue
        size, used, avail = _to_int(parts[1]), _to_int(parts[2]), _to_int(parts[3])
        if not size or size < GIB:
            continue
        vols.append({"device": clean(parts[0]), "mountpoint": clean(parts[5]), "size_gb": gb(size, 1),
                     "used_gb": gb(used, 1), "free_gb": gb(avail, 1)})
    return {"volumes": vols}


# ---------------------------------------------------------------------------- Linux: memory
def parse_free_b(text: str) -> dict:
    for line in text.splitlines():
        if line.lower().startswith("mem:"):
            f = line.split()
            total, used, free = _to_int(f[1]), _to_int(f[2]), _to_int(f[3])
            avail = _to_int(f[6]) if len(f) > 6 else free
            return {"total_bytes": total, "used_bytes": used, "available_bytes": avail}
    return {}


def parse_meminfo(text: str) -> dict:
    kv = _kv(text)
    total = parse_size_to_bytes(kv.get("MemTotal"))
    avail = parse_size_to_bytes(kv.get("MemAvailable"))
    return {"total_bytes": total, "available_bytes": avail,
            "used_bytes": (total - avail) if total and avail is not None else None}


def parse_dmidecode_memory(text: str) -> dict:
    """Parse `dmidecode -t memory` block by block (type 16 = array, type 17 = device)."""
    arrays: list[dict] = []
    devices: list[dict] = []
    for block in re.split(r"\n\s*\n", text):
        lines = [l for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        title = next((l.strip() for l in lines if not l.startswith("Handle") and ":" not in l), "")
        kv = _kv("\n".join(l for l in lines if l.startswith(("\t", " "))))
        if title == "Physical Memory Array":
            arrays.append({
                "max_capacity_bytes": parse_size_to_bytes(kv.get("Maximum Capacity")),
                "slots": _to_int(kv.get("Number Of Devices")),
                "ecc": clean(kv.get("Error Correction Type", "")) or None,
            })
        elif title == "Memory Device":
            size_txt = kv.get("Size", "")
            size = parse_size_to_bytes(size_txt)
            installed = size is not None and size > 0
            speed = re.match(r"(\d+)", kv.get("Configured Memory Speed") or kv.get("Configured Clock Speed") or kv.get("Speed", ""))
            devices.append({
                "slot": clean(kv.get("Locator", "")) or None,
                "installed": installed,
                "size_bytes": size if installed else 0,
                "type": clean(kv.get("Type", "")) or None,
                "form_factor": clean(kv.get("Form Factor", "")) or None,
                "speed_mts": int(speed.group(1)) if speed else None,
                "manufacturer": clean(kv.get("Manufacturer", "")) or None,
                "part_number": clean(kv.get("Part Number", "")) or None,
            })
    return {"arrays": arrays, "devices": devices}


# ------------------------------------------------------------------------------ Linux: GPU
_GPU_CLASSES = ("VGA compatible controller", "3D controller", "Display controller")


def parse_lspci_mm(text: str) -> dict:
    gpus = []
    for line in text.splitlines():
        try:
            f = shlex.split(line)
        except ValueError:
            continue
        if len(f) >= 4 and f[1] in _GPU_CLASSES:
            gpus.append({"name": clean(f"{f[2]} {f[3]}"), "vendor": clean(f[2]), "kind": clean(f[1])})
    return {"gpus": gpus}


def parse_nvidia_smi_csv(text: str) -> dict:
    gpus = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0]:
            gpus.append({"name": clean(parts[0]), "vram_mb": _to_int(parts[1]), "driver_version": clean(parts[2])})
    return {"nvidia": gpus}


# ------------------------------------------------------------------- Linux: overview / battery
def parse_os_release(text: str) -> dict:
    kv = {k: v.strip().strip('"') for k, v in _kv(text, "=").items()}
    return {"os": clean(kv.get("PRETTY_NAME") or kv.get("NAME") or "") or None}


def parse_uname_r(text: str) -> dict:
    return {"kernel": clean(text.strip())}


def parse_battery_files(files: dict[str, dict[str, str]]) -> dict:
    """files: {'BAT0': {'status': 'Charging', 'energy_full': '40000000', ...}}"""
    batteries = []
    for name, f in sorted(files.items()):
        full = _to_int(f.get("energy_full") or f.get("charge_full"))
        design = _to_int(f.get("energy_full_design") or f.get("charge_full_design"))
        health = round(100 * full / design, 1) if full and design else None
        batteries.append({
            "name": clean(name), "status": clean(f.get("status", "")) or None,
            "charge_percent": _to_int(f.get("capacity")), "health_percent": health,
            "cycle_count": _to_int(f.get("cycle_count")) or None,
            "model": clean(f.get("model_name", "")) or None,
            "technology": clean(f.get("technology", "")) or None,
        })
    return {"batteries": batteries}


def parse_dmi_files(files: dict[str, str]) -> dict:
    return {
        "manufacturer": clean(files.get("sys_vendor", "")) or None,
        "model": clean(files.get("product_name", "")) or None,
        "bios_version": clean(files.get("bios_version", "")) or None,
        "bios_date": clean(files.get("bios_date", "")) or None,
    }


# ------------------------------------------------------------------------------- Windows
def parse_win_json(text: str) -> Any:
    text = text.strip().lstrip("﻿")
    return json.loads(text) if text else {}


def as_list(v: Any) -> list:
    """PowerShell's ConvertTo-Json emits a bare object for 1 item and an array for many."""
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


_SMBIOS_MEM_TYPE = {20: "DDR", 21: "DDR2", 24: "DDR3", 26: "DDR4", 27: "LPDDR", 28: "LPDDR2",
                    29: "LPDDR3", 30: "LPDDR4", 34: "DDR5", 35: "LPDDR5"}
_WMI_FORM_FACTOR = {8: "DIMM", 12: "SODIMM", 7: "SIMM", 11: "RIMM", 13: "SRIMM", 21: "BGA",
                    22: "FPBGA", 23: "LGA", 9: "TSOP", 14: "SMD", 6: "Proprietary"}
_CPU_ARCH = {0: "x86", 5: "ARM", 9: "x86_64", 12: "ARM64"}
_BATTERY_STATUS = {1: "Discharging", 2: "On AC", 3: "Fully charged", 4: "Low", 5: "Critical",
                   6: "Charging", 7: "Charging (high)", 8: "Charging (low)", 9: "Charging (critical)"}


def parse_win_memory(obj: dict) -> dict:
    arrays = []
    for a in as_list(obj.get("array")):
        max_kb = _to_int(a.get("MaxCapacity"))
        ex_kb = _to_int(a.get("MaxCapacityEx"))
        if max_kb == 2147483648 and ex_kb:  # sentinel meaning "see MaxCapacityEx"
            max_kb = ex_kb
        arrays.append({"max_capacity_bytes": max_kb * 1024 if max_kb else None,
                       "slots": _to_int(a.get("MemoryDevices")), "ecc": None})
    devices = []
    for d in as_list(obj.get("modules")):
        cap = _to_int(d.get("Capacity")) or 0
        speed = _to_int(d.get("ConfiguredClockSpeed")) or _to_int(d.get("Speed"))
        devices.append({
            "slot": clean(d.get("DeviceLocator") or d.get("BankLabel") or "") or None,
            "installed": cap > 0, "size_bytes": cap,
            "type": _SMBIOS_MEM_TYPE.get(_to_int(d.get("SMBIOSMemoryType")), None),
            "form_factor": _WMI_FORM_FACTOR.get(_to_int(d.get("FormFactor"))),
            "speed_mts": speed or None,
            "manufacturer": clean(d.get("Manufacturer", "")) or None,
            "part_number": clean(d.get("PartNumber", "")) or None,
        })
    os_ = obj.get("os") or {}
    total_kb, free_kb = _to_int(os_.get("TotalVisibleMemorySize")), _to_int(os_.get("FreePhysicalMemory"))
    usage = {}
    if total_kb and free_kb is not None:
        usage = {"total_bytes": total_kb * 1024, "available_bytes": free_kb * 1024,
                 "used_bytes": (total_kb - free_kb) * 1024}
    return {"arrays": arrays, "devices": devices, "usage": usage}


def parse_win_cpu(obj: Any) -> dict:
    procs = as_list(obj)
    if not procs:
        return {}
    p = procs[0]
    return {
        "model": clean(p.get("Name", "")) or None, "vendor": clean(p.get("Manufacturer", "")) or None,
        "architecture": _CPU_ARCH.get(_to_int(p.get("Architecture"))),
        "logical_processors": _to_int(p.get("NumberOfLogicalProcessors")),
        "physical_cores": _to_int(p.get("NumberOfCores")),
        "threads_per_core": None, "sockets": len(procs),
        "max_mhz": _to_int(p.get("MaxClockSpeed")), "min_mhz": None,
        "l2_cache": f"{p['L2CacheSize']} KB" if p.get("L2CacheSize") else None,
        "l3_cache": f"{p['L3CacheSize']} KB" if p.get("L3CacheSize") else None,
        "virtualization": None if p.get("VirtualizationFirmwareEnabled") is None
        else ("enabled in firmware" if p.get("VirtualizationFirmwareEnabled") else "disabled in firmware"),
        "is_virtual_machine": False,
    }


def parse_win_storage(obj: dict) -> dict:
    disks = []
    for d in as_list(obj.get("disks")):
        size = _to_int(d.get("Size")) or 0
        media, bus = clean(d.get("MediaType", "")), clean(d.get("BusType", ""))
        if bus.lower() == "nvme":
            kind = "NVMe SSD"
        elif media.upper() in ("SSD", "HDD"):
            kind = media.upper()
        else:
            kind = "Unknown (media type not reported)"
        disks.append({"name": clean(d.get("DeviceId", "")) or None, "model": clean(d.get("FriendlyName", "")) or None,
                      "size_bytes": size, "size_gb": gb(size, 1), "type": kind,
                      "interface": bus.lower() or None, "health": clean(d.get("HealthStatus", "")) or None,
                      "partitions": []})
    vols = []
    for v in as_list(obj.get("volumes")):
        size, free = _to_int(v.get("Size")) or 0, _to_int(v.get("FreeSpace"))
        if size:
            vols.append({"device": clean(v.get("DeviceID")), "mountpoint": clean(v.get("DeviceID")),
                         "size_gb": gb(size, 1), "used_gb": gb(size - (free or 0), 1), "free_gb": gb(free, 1),
                         "filesystem": clean(v.get("FileSystem", "")) or None})
    return {"disks": disks, "volumes": vols}


def parse_win_gpu(obj: dict) -> dict:
    gpus = []
    for g in as_list(obj.get("controllers")):
        ram = _to_int(g.get("AdapterRAM"))
        gpus.append({"name": clean(g.get("Name", "")), "vendor": clean(g.get("AdapterCompatibility", "")) or None,
                     "kind": "controller", "driver_version": clean(g.get("DriverVersion", "")) or None,
                     "reported_vram_mb": round(ram / MIB) if ram else None,
                     "vram_may_be_capped": bool(ram and ram >= 4_290_000_000)})
    nv = parse_nvidia_smi_csv(obj.get("nvidia_csv") or "").get("nvidia", [])
    return {"gpus": gpus, "nvidia": nv}


def parse_win_overview(obj: dict) -> dict:
    cs, os_, bios = obj.get("cs") or {}, obj.get("os") or {}, obj.get("bios") or {}
    return {"manufacturer": clean(cs.get("Manufacturer", "")) or None, "model": clean(cs.get("Model", "")) or None,
            "os": clean(f"{os_.get('Caption', '')} {os_.get('OSArchitecture', '')}") or None,
            "os_build": clean(os_.get("Version", "")) or None,
            "bios_version": clean(bios.get("SMBIOSBIOSVersion", "")) or None,
            "bios_date": clean(bios.get("ReleaseDate", "")) or None}


def parse_win_battery(obj: dict) -> dict:
    full = _to_int((as_list(obj.get("full")) or [{}])[0].get("FullChargedCapacity"))
    design = _to_int((as_list(obj.get("design")) or [{}])[0].get("DesignedCapacity"))
    out = []
    for b in as_list(obj.get("battery")):
        out.append({"name": clean(b.get("Name", "")) or "Battery", "status": _BATTERY_STATUS.get(_to_int(b.get("BatteryStatus"))),
                    "charge_percent": _to_int(b.get("EstimatedChargeRemaining")),
                    "health_percent": round(100 * full / design, 1) if full and design else None,
                    "cycle_count": None, "model": None, "technology": None})
    return {"batteries": out}
