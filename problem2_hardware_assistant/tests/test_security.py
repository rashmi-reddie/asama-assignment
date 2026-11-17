import subprocess
import pytest

from hw_assistant.runner import SafeRunner, Source, ParamError, validate_args
from hw_assistant.tools import REGISTRY, ToolExecutor, all_sources, sudoers_lines


class Proc:
    def __init__(self, out="", err="", rc=0):
        self.stdout, self.stderr, self.returncode = out, err, rc


def make_runner(run, which=lambda n: f"/usr/bin/{n}", is_root=False):
    return SafeRunner("linux", {id(s) for s in all_sources()}, run=run, which=which, is_root=is_root)


DMI = next(s for s in all_sources() if s.parser == "dmidecode_memory")


def test_unknown_tool_rejected():
    r = ToolExecutor().execute("run_command", {"cmd": "rm -rf /"})
    assert "allow-list" in r["error"]


def test_source_not_in_registry_is_refused():
    runner = make_runner(lambda *a, **k: Proc("ok"))
    rogue = Source("cmd", "lscpu", ("lscpu",))          # same shape, but not a registry object
    res = runner.read(rogue)
    assert not res.ok and res.error_kind == "unsupported"


def test_shell_false_and_fixed_argv():
    seen = {}

    def run(argv, **kw):
        seen["argv"], seen["kw"] = argv, kw
        return Proc("Architecture: x86_64\nCPU(s): 2\n")
    ToolExecutor(make_runner(run), "linux").execute("get_cpu_info")
    assert seen["kw"]["shell"] is False and isinstance(seen["argv"], list)
    assert seen["argv"] == ["/usr/bin/lscpu"]


@pytest.mark.parametrize("evil", ["nvme0n1; rm -rf /", "$(reboot)", "`id`", "a|b", "../../etc/passwd", "x" * 200, "a&&b"])
def test_injection_in_args_rejected(evil):
    r = ToolExecutor(make_runner(lambda *a, **k: Proc("{}")), "linux").execute("get_storage_info", {"device_name": evil})
    assert "invalid arguments" in r["error"]


def test_unknown_and_mistyped_args_rejected():
    with pytest.raises(ParamError):
        validate_args(REGISTRY["get_memory_info"].params, {"shell": "true"})
    with pytest.raises(ParamError):
        validate_args(REGISTRY["get_memory_info"].params, {"include_empty_slots": "yes"})


def test_device_filter_is_applied_to_output_not_command(monkeypatch):
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return Proc('{"blockdevices":[{"name":"sda","size":500107862016,"type":"disk","rota":true,"tran":"sata","model":"WDC"},'
                    '{"name":"nvme0n1","size":512110190592,"type":"disk","rota":false,"tran":"nvme","model":"SAM"}]}')
    r = ToolExecutor(make_runner(run), "linux").execute("get_storage_info", {"device_name": "nvme0n1"})
    assert [d["name"] for d in r["data"]["drives"]] == ["nvme0n1"]
    assert all("nvme0n1" not in " ".join(c) for c in calls)         # never reached a command line


def test_sudo_used_non_interactively_for_privileged_source():
    seen = {}

    def run(argv, **kw):
        seen["argv"] = argv
        return Proc("")
    make_runner(run).read(DMI)
    assert seen["argv"] == ["/usr/bin/sudo", "-n", "/usr/bin/dmidecode", "-t", "memory"]


def test_no_sudo_when_already_root():
    seen = {}
    make_runner(lambda argv, **kw: (seen.update(argv=argv), Proc(""))[1], is_root=True).read(DMI)
    assert seen["argv"][0] == "/usr/bin/dmidecode"


def test_sudo_password_required_falls_back_to_proc_meminfo(monkeypatch):
    """dmidecode fails (no sudoers rule) -> assistant degrades gracefully and says what is missing."""
    original_read_file = SafeRunner._read_file

    def read_file(self, src):
        if src.path == "/proc/meminfo":
            return type(original_read_file(self, src))(
                True,
                raw="MemTotal:       8000000 kB\\nMemAvailable:   6000000 kB\\n",
                source=src.path,
            )
        return original_read_file(self, src)

    monkeypatch.setattr(SafeRunner, "_read_file", read_file)

    def run(argv, **kw):
        if "dmidecode" in " ".join(argv):
            return Proc("", "sudo: a password is required", 1)
        if argv[0].endswith("free"):
            return Proc("              total used free shared buff/cache available\nMem: 8000000000 2000000000 3000000000 0 3000000000 6000000000\n")
        return Proc("")
    r = ToolExecutor(make_runner(run), "linux").execute("get_memory_info")
    assert r["data"]["upgrade_verdict"] == "unknown_limits"
    assert any("elevated privileges" in w for w in r["warnings"])


def test_timeout_is_reported_not_raised():
    def run(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 1)
    res = make_runner(run).read(next(s for s in all_sources() if s.parser == "lscpu"))
    assert not res.ok and res.error_kind == "timeout"


def test_missing_binary_reported():
    res = make_runner(lambda *a, **k: Proc(""), which=lambda n: None).read(next(s for s in all_sources() if s.parser == "lscpu"))
    assert res.error_kind == "missing"


def test_sudoers_rule_pins_exact_arguments():
    lines = sudoers_lines("rashmitha", which=lambda n: f"/usr/sbin/{n}")
    assert lines == ["rashmitha ALL=(root) NOPASSWD: /usr/sbin/dmidecode -t memory"]


def test_no_registry_source_ever_contains_shell_metacharacters():
    for s in all_sources():
        if s.kind == "cmd" and not s.argv[0].startswith("powershell"):
            assert not any(ch in a for a in s.argv for ch in ";|&$`><"), s


def test_powershell_scripts_are_exactly_the_fixed_constants():
    from hw_assistant import tools
    constants = {v for k, v in vars(tools).items() if k.startswith("PS_") and isinstance(v, str)}
    ps = [s for s in all_sources() if s.argv and s.argv[0] == "powershell.exe"]
    assert len(ps) == 6
    for s in ps:
        assert s.argv[:4] == ("powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive")
        assert s.argv[4] == "-Command" and s.argv[5] in constants and len(s.argv) == 6
