from hw_assistant import parsers as P
from hw_assistant.tools import REGISTRY, ToolExecutor
from tests import fixtures as F
from tests.helpers import FakeRunner


def test_size_parsing():
    assert P.parse_size_to_bytes("8 GB") == 8 * 1024**3
    assert P.parse_size_to_bytes("8192 MB") == 8 * 1024**3
    assert P.parse_size_to_bytes("No Module Installed") is None
    assert P.parse_size_to_bytes("Unknown") is None


def test_clean_strips_control_chars_and_caps_length():
    assert P.clean("a\x00b\x1b[31m  c") == "ab[31m c"
    assert len(P.clean("x" * 1000)) == 120


def test_lscpu():
    d = P.parse_lscpu(F.LSCPU)
    assert d["model"].startswith("11th Gen Intel") and d["physical_cores"] == 4 and d["logical_processors"] == 8
    assert d["max_mhz"] == 4200.0


def test_dmidecode_one_free_slot():
    d = P.parse_dmidecode_memory(F.DMIDECODE_ONE_FREE)
    assert d["arrays"][0]["slots"] == 2 and d["arrays"][0]["max_capacity_bytes"] == 64 * 1024**3
    assert [x["installed"] for x in d["devices"]] == [True, False]
    assert d["devices"][0]["type"] == "DDR4" and d["devices"][0]["speed_mts"] == 3200


def test_lsblk_skips_pseudo_devices_and_classifies():
    disks = {x["name"]: x for x in P.parse_lsblk_json(F.LSBLK)["disks"]}
    assert set(disks) == {"nvme0n1", "sda"}
    assert disks["nvme0n1"]["type"] == "NVMe SSD" and disks["sda"]["type"] == "HDD"
    assert len(disks["nvme0n1"]["partitions"]) == 2


def test_gpu_parsers():
    g = P.parse_lspci_mm(F.LSPCI_MM)["gpus"]
    assert [x["vendor"] for x in g] == ["Intel Corporation", "NVIDIA Corporation"]  # audio device ignored
    assert P.parse_nvidia_smi_csv(F.NVIDIA_SMI)["nvidia"][0]["vram_mb"] == 4096


def test_windows_single_object_vs_array_normalised():
    one = P.parse_win_memory(P.parse_win_json(F.WIN_MEMORY_ONE))
    two = P.parse_win_memory(P.parse_win_json(F.WIN_MEMORY_TWO_LPDDR))
    assert len(one["devices"]) == 1 and len(two["devices"]) == 2
    assert one["devices"][0]["type"] == "DDR4" and one["devices"][0]["form_factor"] == "SODIMM"
    assert one["arrays"][0]["max_capacity_bytes"] == 32 * 1024**3        # KB -> bytes
    assert one["usage"]["used_bytes"] > 0


def test_windows_gpu_vram_cap_flag():
    g = P.parse_win_gpu(P.parse_win_json(F.WIN_GPU))
    assert g["gpus"][1]["vram_may_be_capped"] is True and g["nvidia"][0]["vram_mb"] == 4096


def test_windows_battery_health():
    b = P.parse_win_battery(P.parse_win_json(F.WIN_BATTERY))["batteries"][0]
    assert b["charge_percent"] == 87 and b["health_percent"] == 87.8


def memory_result(platform, raw_by_parser):
    return ToolExecutor(FakeRunner(raw_by_parser), platform).execute("get_memory_info")


def test_upgrade_verdict_add_module():
    d = memory_result("linux", {"dmidecode_memory": F.DMIDECODE_ONE_FREE, "free_b": F.FREE_B})["data"]
    assert (d["installed_total_gb"], d["free_slots"], d["max_supported_gb"], d["upgrade_headroom_gb"]) == (8.0, 1, 64.0, 56.0)
    assert d["upgrade_verdict"] == "add_module_in_free_slot" and d["memory_type"] == "DDR4"


def test_upgrade_verdict_replace_when_full():
    d = memory_result("linux", {"dmidecode_memory": F.DMIDECODE_FULL})["data"]
    assert d["free_slots"] == 0 and d["upgrade_verdict"] == "replace_modules_with_larger"
    assert d["max_per_slot_gb_if_slots_equal"] == 32.0


def test_upgrade_verdict_soldered():
    d = memory_result("linux", {"dmidecode_memory": F.DMIDECODE_SOLDERED})["data"]
    assert d["likely_soldered"] and d["upgrade_verdict"] == "likely_soldered"


def test_windows_lpddr_detected_as_soldered():
    d = memory_result("windows", {"win_json": F.WIN_MEMORY_TWO_LPDDR})["data"]
    assert d["likely_soldered"] and d["installed_total_gb"] == 16.0


def test_windows_one_stick_two_slots():
    d = memory_result("windows", {"win_json": F.WIN_MEMORY_ONE})["data"]
    assert d["free_slots"] == 1 and d["upgrade_verdict"] == "add_module_in_free_slot" and d["max_supported_gb"] == 32.0


def test_every_windows_tool_parses_its_fixture():
    fx = {"get_cpu_info": F.WIN_CPU, "get_storage_info": F.WIN_STORAGE, "get_gpu_info": F.WIN_GPU,
          "get_system_overview": F.WIN_OVERVIEW, "get_battery_info": F.WIN_BATTERY}
    for name, raw in fx.items():
        r = ToolExecutor(FakeRunner({"win_json": raw}), "windows").execute(name)
        assert "error" not in r, (name, r)
    gpu = ToolExecutor(FakeRunner({"win_json": F.WIN_GPU}), "windows").execute("get_gpu_info")["data"]["gpus"]
    assert any(g.get("vram_mb") == 4096 for g in gpu)           # nvidia-smi overrides the 4 GB WMI cap
