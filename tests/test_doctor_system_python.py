"""Doctor reports whether each host's system Python starts and is new enough.

The Worker generation and installation helpers run on ``SYSTEM_PYTHON``, not
on the configured venv, so doctor probes it on every host with its own shell
command. They call str.removeprefix, so an interpreter older than
``SYSTEM_PYTHON_MIN`` fails the row. A transport loss reads as unobserved and
never moves the exit code.
"""
from __future__ import annotations

import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from dgx_monarch.cli import doctor, doctor_probes, probe_certainty
from dgx_monarch.cli.lifecycle_generation import SYSTEM_PYTHON, SYSTEM_PYTHON_MIN
from dgx_monarch.config import ClusterConfig, HostConfig

HOST = HostConfig(name="h1", address="tcp://10.0.0.2:26600")
CONFIG = ClusterConfig(hosts=(HOST,), client_bind="tcp://10.0.0.1:0",
                       transport_security="trusted_fabric")


class _Runner:
    def __init__(self, result):
        self.result = result
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, _config, _host, script, **kwargs):
        self.calls.append((script, kwargs))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def _local_bash(_config, _host, script, timeout):
    return subprocess.run(["bash", "-s"], input=script, capture_output=True, text=True, timeout=timeout)


def test_the_probe_is_the_isolated_no_site_start_of_the_shared_constant():
    probe = doctor_probes.SYSTEM_PYTHON_PROBE
    assert SYSTEM_PYTHON_MIN == (3, 9)
    assert shlex.split(probe) == [
        SYSTEM_PYTHON, "-I", "-S", "-c", doctor_probes.SYSTEM_PYTHON_PROBE_SOURCE]
    assert "raise SystemExit(sys.version_info < (3, 9))" in doctor_probes.SYSTEM_PYTHON_PROBE_SOURCE
    runner = _Runner(subprocess.CompletedProcess([], 0, "3.12.3\n", ""))
    status, name, detail = doctor_probes.system_python_row(CONFIG, HOST, runner)
    assert (status, name) == (probe_certainty.OK, "h1 system python")
    assert detail == "/usr/bin/python3 is Python 3.12.3"
    assert runner.calls == [(probe, {"timeout": doctor_probes.SYSTEM_PYTHON_PROBE_TIMEOUT_S})]


@pytest.mark.parametrize("stdout", ["3.8.10\n", "login banner\n3.8.10\n"])
def test_an_interpreter_below_the_floor_fails_with_its_version(stdout):
    """A 3.8 interpreter starts, so only the version check can catch it. Its
    helpers would raise AttributeError at the first quarantine or snapshot."""
    runner = _Runner(subprocess.CompletedProcess([], 1, stdout, ""))
    status, _name, detail = doctor_probes.system_python_row(CONFIG, HOST, runner)
    assert status == probe_certainty.FAIL
    assert detail == ("/usr/bin/python3 is Python 3.8.10. The Worker lifecycle helpers "
                      "need Python 3.9 or newer; see docs/CLUSTER.md")


def test_the_probe_source_exits_1_below_the_floor_and_0_at_it():
    """Run the probe's own source with the version it reads stood in for."""
    for version, code in (((3, 8, 10), 1), ((3, 9, 0), 0)):
        stand_in = f"import sys; sys.version_info = {(*version, 'final', 0)!r}\n"
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", stand_in + doctor_probes.SYSTEM_PYTHON_PROBE_SOURCE],
            capture_output=True, text=True, timeout=30)
        assert (result.returncode, result.stdout) == (code, ".".join(map(str, version)) + "\n"), result.stderr


def test_a_missing_interpreter_fails_with_the_shell_error():
    runner = _Runner(subprocess.CompletedProcess(
        [], 127, "", "bash: line 1: /usr/bin/python3: No such file or directory\n"))
    status, _name, detail = doctor_probes.system_python_row(CONFIG, HOST, runner)
    assert status == probe_certainty.FAIL
    assert "exited 127: bash: line 1: /usr/bin/python3: No such file or directory" in detail
    assert "docs/CLUSTER.md" in detail


@pytest.mark.parametrize("result", [
    subprocess.CompletedProcess([], 255, "", "ssh: connect to host h1: No route to host"),
    OSError("no route"),
    subprocess.TimeoutExpired(["ssh"], 15),
])
def test_a_transport_loss_is_unobserved_not_a_failure(result):
    status, _name, detail = doctor_probes.system_python_row(CONFIG, HOST, _Runner(result))
    assert status == probe_certainty.WARN
    assert detail.startswith("unobserved:")


def test_the_probe_runs_through_a_real_shell_on_this_box():
    status, _name, detail = doctor_probes.system_python_row(CONFIG, HOST, _local_bash)
    assert status == probe_certainty.OK, detail
    assert detail.startswith("/usr/bin/python3 is Python 3."), detail


def test_a_real_shell_reports_an_old_interpreter(monkeypatch):
    old = f"import sys; sys.version_info = (3, 8, 10, 'final', 0)\n{doctor_probes.SYSTEM_PYTHON_PROBE_SOURCE}"
    monkeypatch.setattr(doctor_probes, "SYSTEM_PYTHON_PROBE",
                        shlex.join((sys.executable, "-I", "-S", "-c", old)))
    status, _name, detail = doctor_probes.system_python_row(CONFIG, HOST, _local_bash)
    assert status == probe_certainty.FAIL
    assert "is Python 3.8.10" in detail and "3.9 or newer" in detail


def test_a_real_shell_reports_an_absent_interpreter(monkeypatch, tmp_path):
    absent = tmp_path / "python3"
    monkeypatch.setattr(doctor_probes, "SYSTEM_PYTHON_PROBE", shlex.join(
        (str(absent), "-I", "-S", "-c", doctor_probes.SYSTEM_PYTHON_PROBE_SOURCE)))
    status, _name, detail = doctor_probes.system_python_row(CONFIG, HOST, _local_bash)
    assert status == probe_certainty.FAIL
    assert "exited 127" in detail and "No such file or directory" in detail


@pytest.fixture
def rig(monkeypatch):
    """Run doctor's per-host loop with every other probe stood in for."""
    import importlib.metadata
    import shutil
    import sys
    import types

    from dgx_monarch import TORCHMONARCH_PIN

    torch = types.ModuleType("torch")
    torch.__version__ = "2.test"
    torch.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: TORCHMONARCH_PIN)
    monkeypatch.setattr(doctor, "_frontend_skew_row",
                        lambda: {"status": probe_certainty.OK, "name": "frontend", "detail": "stub"})
    monkeypatch.setattr(doctor, "_spark_health_rows", lambda _config: [])
    monkeypatch.setattr(doctor, "_mesh_health_row",
                        lambda: {"status": probe_certainty.OK, "name": "mesh health", "detail": "stub"})
    monkeypatch.setattr(doctor.socket, "gethostname", lambda: "driver")
    monkeypatch.setattr(doctor.socket, "gethostbyname", lambda _name: "10.0.0.1")
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/rsync")
    monkeypatch.setattr(doctor, "_tcp_probe", lambda *_a, **_kw: False)
    monkeypatch.setattr(doctor, "passive_worker_health",
                        lambda *_a, **_kw: {"running": True, "listening": True, "healthy": True})
    return monkeypatch


def test_doctor_reports_the_row_even_when_the_venv_probe_fails(rig, capsys):
    """The helpers need system Python most when the Worker venv is broken."""
    scripts: list[str] = []

    def run(_config, _host, script, **_kwargs):
        scripts.append(script)
        if script == doctor_probes.SYSTEM_PYTHON_PROBE:
            return subprocess.CompletedProcess([], 1, "", "Fatal Python error: init_import_site\n")
        return subprocess.CompletedProcess([], 1, "", "venv python: No such file or directory\n")

    rig.setattr(doctor, "run_on_host", run)
    rows = doctor._doctor_rows(CONFIG)
    output = capsys.readouterr().out
    row = next(item for item in rows if item["name"] == "h1 system python")
    assert row["status"] == probe_certainty.FAIL
    assert "exited 1: Fatal Python error: init_import_site" in row["detail"]
    assert "[FAIL] h1 system python:" in output
    assert scripts.count(doctor_probes.SYSTEM_PYTHON_PROBE) == 1
    names = [item["name"] for item in rows]
    assert names.index("h1 system python") < names.index("h1 env")
