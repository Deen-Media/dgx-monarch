"""Require one named doctor row per health probe, including failed probes.

Missing or broken tools must produce a row so the reported check count does
not silently omit unobserved health checks.
"""
from __future__ import annotations

import subprocess
import sys
import types

from dgx_monarch.cli import doctor


def _row(rows, name):
    return next(r for r in rows if r["name"] == name)


def _stub_no_cuda_torch(monkeypatch):
    """Stub torch with no CUDA device, so the GPU rows take their no-device branch."""
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)


def test_sageattention_row_present_when_torch_itself_is_unimportable(monkeypatch):
    # sys.modules[name] = None is the standard way to make `import name`
    # raise ImportError without touching builtins.__import__.
    monkeypatch.setitem(sys.modules, "torch", None)
    rows = doctor._spark_health_rows(None)
    row = _row(rows, "sageattention")
    assert row["status"] == doctor._WARN
    assert "probe unavailable" in row["detail"]


def test_sageattention_row_present_when_no_cuda_device(monkeypatch):
    _stub_no_cuda_torch(monkeypatch)
    rows = doctor._spark_health_rows(None)
    row = _row(rows, "sageattention")
    assert row["status"] == doctor._OK
    assert "no CUDA device" in row["detail"]


def test_default_doctor_never_dispatches_cuda_or_sage_work(monkeypatch):
    calls: list[str] = []

    def forbidden(name):
        def trap(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(f"passive doctor dispatched {name}")

        return trap

    torch_stub = types.ModuleType("torch")
    torch_stub.randn = forbidden("torch.randn")
    torch_stub.matmul = forbidden("torch.matmul")
    torch_stub.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=forbidden("torch.cuda.get_device_name"),
        synchronize=forbidden("torch.cuda.synchronize"),
    )
    sage_stub = types.ModuleType("sageattention")
    sage_stub.sageattn = forbidden("sageattention.sageattn")
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "sageattention", sage_stub)
    # A blind busy check (gpu_busy returns None) keeps the kernel probe passive.
    from dgx_monarch.cli import doctor_probes

    monkeypatch.setattr(doctor_probes, "gpu_busy", lambda: None)
    monkeypatch.setattr(doctor_probes, "run_kernel_probe", forbidden("kernel subprocess"))

    rows = doctor._spark_health_rows(None)

    assert calls == []
    sage = [row for row in rows if row["name"] == "sageattention"]
    clock = [row for row in rows if row["name"] == "GPU clock under load"]
    assert len(sage) == len(clock) == 1
    assert sage[0]["status"] == doctor._WARN
    assert clock[0]["status"] == doctor._WARN
    assert "NOT RUN" in sage[0]["detail"]
    assert "NOT RUN" in clock[0]["detail"]


def test_python_dev_headers_row_present_when_sysconfig_fails(monkeypatch):
    _stub_no_cuda_torch(monkeypatch)
    import sysconfig

    def boom():
        raise RuntimeError("no sysconfig paths on this build")

    monkeypatch.setattr(sysconfig, "get_paths", boom)
    rows = doctor._spark_health_rows(None)
    row = _row(rows, "python dev headers")
    assert row["status"] == doctor._WARN
    assert "probe unavailable" in row["detail"]


def test_board_hotspots_row_present_when_the_verdict_raises(monkeypatch):
    _stub_no_cuda_torch(monkeypatch)

    def raiser():
        raise RuntimeError("hwmon reader blew up")

    monkeypatch.setattr(doctor, "board_hotspot_verdict", raiser)
    rows = doctor._spark_health_rows(None)
    named = [r for r in rows if r["name"] == "board hotspots"]
    assert len(named) == 1
    assert named[0]["status"] == doctor._WARN
    assert "probe unavailable" in named[0]["detail"]


def test_board_hotspots_row_is_not_a_warn_when_the_box_exposes_no_sensors(monkeypatch):
    _stub_no_cuda_torch(monkeypatch)
    detail = ("no temperature sensors on this box: nothing readable under "
              "/sys/class/hwmon and `sensors` added none.")
    monkeypatch.setattr(doctor, "board_hotspot_verdict", lambda: (False, detail))
    row = _row(doctor._spark_health_rows(None), "board hotspots")
    assert row["status"] == doctor._OK
    assert row["detail"] == detail


def test_board_hotspots_row_warns_when_a_sensor_is_at_the_hazard_threshold(monkeypatch):
    _stub_no_cuda_torch(monkeypatch)
    detail = "1 of 15 sensors at or above 90 C: hwmon0/acpitz/temp6 90.0 C"
    monkeypatch.setattr(doctor, "board_hotspot_verdict", lambda: (True, detail))
    row = _row(doctor._spark_health_rows(None), "board hotspots")
    assert row["status"] == doctor._WARN
    assert row["detail"] == detail


def test_board_hotspots_row_needs_no_sensors_binary(monkeypatch):
    # The unstubbed verdict runs against this box's real /sys/class/hwmon.
    # No count is asserted: a container with no hwmon tree lands on the
    # information row, which also carries no `probe unavailable`.
    _stub_no_cuda_torch(monkeypatch)
    real_run = subprocess.run

    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == "sensors":
            raise FileNotFoundError("sensors: command not found")
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    row = _row(doctor._spark_health_rows(None), "board hotspots")
    assert row["status"] in (doctor._OK, doctor._WARN)
    assert "probe unavailable" not in row["detail"]


def _stub_orphan_report(monkeypatch, result):
    from dgx_monarch.cli import actor_reaper

    def report(**_kwargs):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(actor_reaper, "orphan_report", report)


def _orphan(pid=117707, gib=49.1, provenance="orphan", established=0):
    from dgx_monarch.cli.actor_reaper import ActorProc

    return ActorProc(
        pid=pid, starttime=900, ppid=1, gib=gib, rss_gib=gib,
        established=established, provenance=provenance, age_s=22320.0,
        orphan_s=None, argv_tail="-m monarch._src.actor.bootstrap_main",
    )


def test_orphaned_actor_row_present_when_the_probe_raises(monkeypatch):
    _stub_no_cuda_torch(monkeypatch)
    _stub_orphan_report(monkeypatch, PermissionError("procfs unreadable"))
    row = _row(doctor._spark_health_rows(None), "orphaned actor procs")
    assert row["status"] == doctor._WARN
    assert "unobserved" in row["detail"]
    assert "PermissionError" not in row["detail"]


def test_orphaned_actor_row_warns_and_names_the_pid_and_size(monkeypatch):
    from dgx_monarch.cli.actor_reaper import OrphanReport

    _stub_no_cuda_torch(monkeypatch)
    _stub_orphan_report(monkeypatch, OrphanReport(
        tracked=2, attached=1, orphans=(_orphan(),)))
    row = _row(doctor._spark_health_rows(None), "orphaned actor procs")
    assert row["status"] == doctor._WARN
    assert "117707" in row["detail"]
    assert "49.1 GiB" in row["detail"]
    assert "age 6h12m" in row["detail"]
    assert "dgxm restart" in row["detail"] and "dgxm reap" in row["detail"]


def test_orphaned_actor_row_is_ok_when_every_actor_has_a_client(monkeypatch):
    from dgx_monarch.cli.actor_reaper import OrphanReport

    _stub_no_cuda_torch(monkeypatch)
    _stub_orphan_report(monkeypatch, OrphanReport(tracked=2, attached=2, orphans=()))
    row = _row(doctor._spark_health_rows(None), "orphaned actor procs")
    assert row["status"] == doctor._OK
    assert "2 tracked, 2 attached" in row["detail"]


def test_orphaned_actor_probe_is_read_only_against_the_real_proc(monkeypatch):
    # The unstubbed probe runs in these tests and in test_cli's real `doctor`
    # invocation, so it must never shell out or signal anything.
    _stub_no_cuda_torch(monkeypatch)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the orphan probe must not signal a process")

    monkeypatch.setattr("os.kill", forbidden)
    row = _row(doctor._spark_health_rows(None), "orphaned actor procs")
    assert row["status"] in (doctor._OK, doctor._WARN)


def _break_command(monkeypatch, name):
    """Make one shelled-out probe fail while the rest of the run proceeds."""
    real_run = subprocess.run

    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == name:
            raise FileNotFoundError(f"{name}: command not found")
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_unexpected_shutdown_row_warns_when_the_boot_records_are_unreadable(monkeypatch):
    _stub_no_cuda_torch(monkeypatch)
    _break_command(monkeypatch, "last")
    row = _row(doctor._spark_health_rows(None), "unexpected shutdowns")
    assert row["status"] == doctor._WARN
    assert "unobserved" in row["detail"]
    assert "FileNotFoundError" not in row["detail"]


def test_co_resident_row_warns_when_nvidia_smi_is_missing(monkeypatch):
    _stub_no_cuda_torch(monkeypatch)
    _break_command(monkeypatch, "nvidia-smi")
    row = _row(doctor._spark_health_rows(None), "co-resident GPU procs")
    assert row["status"] == doctor._WARN
    assert "unobserved" in row["detail"]
    assert "FileNotFoundError" not in row["detail"]


def test_gpu_clock_row_warns_when_torch_itself_is_unimportable(monkeypatch):
    """A box that cannot import torch reads WARN, and one that can reads WARN
    too (the passive doctor runs no synthetic load). An OK here would make the
    broken box look healthier than the working one."""
    monkeypatch.setitem(sys.modules, "torch", None)
    row = _row(doctor._spark_health_rows(None), "GPU clock under load")
    assert row["status"] == doctor._WARN
    assert "unobserved" in row["detail"]
    assert "ImportError" not in row["detail"]


def test_gpu_clock_row_is_ok_only_when_the_absent_device_was_observed(monkeypatch):
    _stub_no_cuda_torch(monkeypatch)
    row = _row(doctor._spark_health_rows(None), "GPU clock under load")
    assert row["status"] == doctor._OK
    assert "no CUDA device" in row["detail"]
