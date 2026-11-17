"""Deterministic answer renderer.

Used (a) when no LLM is configured, and (b) as the safety net when an LLM answer fails the
grounding check.  Every sentence is generated from verified tool data, so it cannot hallucinate.
"""
from __future__ import annotations

from typing import Any


def _g(x: float | None) -> str:
    if x is None:
        return "unknown"
    return f"{int(x)} GB" if float(x).is_integer() else f"{x:g} GB"


def _n(x) -> str:
    return f"{x:,.1f}".rstrip("0").rstrip(".")


def _unavailable(r: dict) -> str:
    why = r.get("error", "unknown error")
    warn = "; ".join(r.get("warnings", [])[:2])
    return f"I cannot determine this from system data ({why}{': ' + warn if warn else ''})."


def render_memory(d: dict) -> str:
    out = []
    total = d.get("installed_total_gb")
    if d.get("total_slots") is not None:
        out.append(f"You have {_g(total)} of RAM installed, using {d['used_slots']} of {d['total_slots']} memory slot(s) "
                   f"({d['free_slots']} free).")
        mods = d.get("modules") or []
        if mods:
            desc = ", ".join(f"{_g(m['size_gb'])} {m.get('type') or ''}".strip() + (f" @ {m['speed_mts']} MT/s" if m.get("speed_mts") else "") for m in mods)
            out.append(f"Installed module(s): {desc}.")
        v = d.get("upgrade_verdict")
        if d.get("firmware_inconsistent_max"):
            out.append("The firmware reports a maximum smaller than what is installed, so I cannot trust the maximum-capacity figure.")
        elif v == "likely_soldered":
            out.append("The memory looks soldered to the motherboard (low-power or non-DIMM chips), so it most likely cannot be upgraded.")
            if d.get("max_supported_gb"):
                out.append(f"The firmware lists a maximum of {_g(d['max_supported_gb'])}, but that does not mean extra RAM can be added.")
        elif v == "add_module_in_free_slot":
            out.append(f"The system supports up to {_g(d['max_supported_gb'])}, so you can add up to {_g(d['upgrade_headroom_gb'])} more "
                       f"using the free slot(s).")
        elif v == "replace_modules_with_larger":
            out.append(f"All slots are used. The system supports up to {_g(d['max_supported_gb'])}, so you could gain up to "
                       f"{_g(d['upgrade_headroom_gb'])} by replacing the existing module(s) with larger ones"
                       + (f" (about {_g(d['max_per_slot_gb_if_slots_equal'])} per slot if the slots are equal)." if d.get("max_per_slot_gb_if_slots_equal") else "."))
        elif v == "already_at_maximum":
            out.append(f"You are already at the maximum of {_g(d['max_supported_gb'])} the system supports.")
        else:
            out.append("I cannot determine the maximum supported capacity from system data.")
    else:
        out.append(f"The system has about {_g(total)} of RAM in total.")
        out.append("I cannot determine the number of memory slots or the maximum supported capacity (that needs SMBIOS/DMI data, "
                   "which requires the privileged lookup described in the README).")
    if d.get("in_use_gb") is not None:
        out.append(f"Right now {_g(d['in_use_gb'])} is in use and {_g(d.get('available_gb'))} is available.")
    return " ".join(out)


def render_cpu(d: dict) -> str:
    parts = [f"CPU: {d.get('model') or 'unknown model'}."]
    if d.get("physical_cores") and d.get("logical_processors"):
        parts.append(f"{d['physical_cores']} physical core(s) and {d['logical_processors']} logical processor(s).")
    elif d.get("logical_processors"):
        parts.append(f"{d['logical_processors']} logical processor(s).")
    if d.get("max_mhz"):
        parts.append(f"Maximum clock speed: {d['max_mhz'] / 1000:.2f} GHz.")
    caches = [f"{k.upper()} {d[k + '_cache']}" for k in ("l2", "l3") if d.get(k + "_cache")]
    if caches:
        parts.append("Cache: " + ", ".join(caches) + ".")
    if d.get("virtualization"):
        parts.append(f"Virtualization: {d['virtualization']}.")
    return " ".join(parts)


def render_storage(d: dict) -> str:
    if not d.get("drives"):
        return "I could not find any matching physical drives."
    out = [f"{d['drive_count']} drive(s), {_n(d['total_capacity_gb'])} GB total."]
    for dr in d["drives"]:
        out.append(f"- {dr.get('model') or dr['name']}: {_n(dr['size_gb'])} GB, {dr['type']}"
                   + (f" ({dr['interface']})" if dr.get("interface") else "")
                   + (f", health {dr['health']}" if dr.get("health") else ""))
    for v in d.get("volumes", [])[:6]:
        out.append(f"- Volume {v['mountpoint']}: {_n(v['free_gb'])} GB free of {_n(v['size_gb'])} GB")
    out.append("I cannot determine how many free M.2/SATA slots the laptop has, or whether the drive is replaceable.")
    return "\n".join(out)


def render_gpu(d: dict) -> str:
    if not d.get("gpus"):
        return "I could not detect any graphics adapter."
    out = [f"{d['gpu_count']} graphics adapter(s) detected:"]
    for g in d["gpus"]:
        line = f"- {g['name']}"
        if g.get("vram_mb"):
            line += f", {g['vram_mb'] / 1024:g} GB VRAM"
        elif g.get("reported_vram_mb"):
            line += f", about {g['reported_vram_mb'] / 1024:g} GB VRAM reported" + (" (Windows may cap this value at 4 GB)" if g.get("vram_may_be_capped") else "")
        else:
            line += ", VRAM not reported (integrated GPUs share system memory)"
        if g.get("driver_version"):
            line += f", driver {g['driver_version']}"
        out.append(line)
    return "\n".join(out)


def render_battery(d: dict) -> str:
    if not d.get("battery_present"):
        return "No battery was detected (this may be a desktop or a virtual machine)."
    out = []
    for b in d["batteries"]:
        line = f"Battery {b['name']}: "
        bits = []
        if b.get("charge_percent") is not None:
            bits.append(f"{b['charge_percent']}% charged")
        if b.get("status"):
            bits.append(b["status"].lower())
        if b.get("health_percent") is not None:
            bits.append(f"health {b['health_percent']}% of design capacity")
        if b.get("cycle_count"):
            bits.append(f"{b['cycle_count']} charge cycles")
        out.append(line + (", ".join(bits) if bits else "no details reported") + ".")
    out.append("I cannot determine the remaining runtime in minutes.")
    return " ".join(out)


def render_overview(d: dict) -> str:
    bits = []
    if d.get("manufacturer") or d.get("model"):
        bits.append(f"Machine: {' '.join(x for x in (d.get('manufacturer'), d.get('model')) if x)}.")
    if d.get("os"):
        bits.append(f"OS: {d['os']}" + (f" (build {d['os_build']})" if d.get("os_build") else "") + ".")
    if d.get("kernel"):
        bits.append(f"Kernel: {d['kernel']}.")
    if d.get("bios_version"):
        bits.append(f"BIOS: {d['bios_version']}" + (f" ({d['bios_date']})" if d.get("bios_date") else "") + ".")
    return " ".join(bits) or "I cannot determine the system model from system data."


_RENDERERS = {
    "get_memory_info": render_memory, "get_cpu_info": render_cpu, "get_storage_info": render_storage,
    "get_gpu_info": render_gpu, "get_battery_info": render_battery, "get_system_overview": render_overview,
}


def render_answer(results: list[dict[str, Any]]) -> str:
    parts = []
    for r in results:
        if "error" in r:
            parts.append(_unavailable(r))
        else:
            parts.append(_RENDERERS[r["tool"]](r["data"]))
    return "\n\n".join(parts) if parts else "I have no hardware data to answer with."
