"""Check doctor's advisory listener-generation row for each worker loop.

A listening socket alone cannot distinguish an old loop from its replacement
(docs/TROUBLESHOOTING.md #101). The row identifies the listening process and
its age. Unobserved readings do not affect the exit code; the Worker service
row already reports a missing listener.

The probe runs against a local test socket, never a worker protocol socket.
"""
from __future__ import annotations

import contextlib
import inspect
import io
import socket
from types import SimpleNamespace

import pytest

from dgx_monarch.cli import doctor, listener_generation
from dgx_monarch.cli.probe_certainty import OK, UNOBSERVED, WARN, probe_row

HOST = "h1"
ADDRESS = "tcp://10.0.0.1:26600"


def _reading(**overrides) -> dict[str, str]:
    """One host's parsed line, spelled the way the snippet prints it."""
    fields = {
        "gen": "5c4ab03463a9", "loop_pid": "4242", "age_s": "612",
        "listening": "true", "unit_state": "active", "unit_restarts": "0",
        "unit_started": "Mon_2026-09-07_10:11:12_UTC", "marker": "match",
    }
    fields.update(overrides)
    return fields


def _run(address: str) -> str:
    """Run the snippet in this process and return the line it printed."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(listener_generation.probe_source(address),  # noqa: S102
                     "<listener-generation-probe>", "exec"), {})
    return buffer.getvalue().strip()


def test_an_observed_loop_reads_ok_and_carries_every_field():
    status, name, detail, reason, critical = listener_generation.doctor_row(
        HOST, _reading())
    assert (status, name) == (OK, "h1 listener generation")
    assert "gen=5c4ab03463a9" in detail
    assert "pid=4242" in detail
    assert "age_s=612" in detail
    assert "restarts=0" in detail
    assert "started=Mon_2026-09-07_10:11:12_UTC" in detail
    assert "marker=match" in detail
    assert (reason, critical) == ("", False)


def test_the_real_row_prints_and_never_gates_the_run():
    row = probe_row(*listener_generation.doctor_row(HOST, _reading()))
    assert row["status"] == OK
    assert "critical" not in row


def test_a_probe_that_printed_nothing_reads_unobserved_not_failed():
    status, _name, detail, reason, critical = listener_generation.doctor_row(
        HOST, {})
    assert status == WARN
    assert detail.startswith("unobserved:")
    assert reason == UNOBSERVED
    assert critical is False
    assert listener_generation.ENTRY in detail


def test_a_probe_that_named_its_fault_puts_it_in_the_row():
    _status, _name, detail, reason, _critical = listener_generation.doctor_row(
        HOST, {"gen": "unknown", "error": "TimeoutExpired"})
    assert "TimeoutExpired" in detail
    assert reason == UNOBSERVED


def test_a_loop_with_no_process_on_its_socket_warns_and_stays_advisory():
    status, _name, detail, reason, critical = listener_generation.doctor_row(
        HOST, _reading(gen="none", loop_pid="none", listening="false"))
    assert status == WARN
    assert "no process holds the worker LISTEN socket" in detail
    assert (reason, critical) == ("", False)


def test_the_snippet_reads_the_endpoint_the_configured_address_names():
    source = listener_generation.probe_source("tcp://127.0.0.1:26600")
    assert "'/proc/net/tcp'" in source
    assert "'0100007F:67E8'" in source          # 26600 == 0x67E8, little-endian ip
    assert "'dgxm-worker.service'" in source


def test_an_address_with_nothing_listening_reads_as_no_listener():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    fields = listener_generation.parse(_run(f"tcp://127.0.0.1:{free}"))
    assert fields["gen"] == "none"
    assert fields["listening"] == "false"


def test_a_real_listener_is_read_back_as_listening():
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        port = held.getsockname()[1]
        fields = listener_generation.parse(_run(f"tcp://127.0.0.1:{port}"))
    assert fields["listening"] == "true"
    assert set(fields) >= {"gen", "loop_pid", "age_s", "marker", "unit_restarts"}


def test_the_line_is_picked_out_of_the_several_the_host_script_prints():
    lines = ["torchmonarch=0.6.0 torch=2.9.0", "sol=True tvm=True",
             listener_generation.LINE_PREFIX + "abc123 loop_pid=7",
             "dgxm_swap=0.0"]
    assert listener_generation.select(lines)["gen"] == "abc123"
    assert listener_generation.select(lines[:2]) == {}


def test_a_line_that_is_not_this_probes_line_parses_as_nothing():
    assert listener_generation.parse("sol=True tvm=True") == {}


class _Runner:
    """Records the script it was handed; contacts nothing."""

    def __init__(self, result):
        self.result = result
        self.scripts: list[str] = []

    def __call__(self, _config, host, script, timeout=None):
        self.scripts.append(script)
        self.timeout = timeout
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def _config(*addresses: str):
    from dgx_monarch.config import ClusterConfig, HostConfig

    return ClusterConfig(
        hosts=tuple(HostConfig(name=f"h{index}", address=address)
                    for index, address in enumerate(addresses, 1)),
        client_bind="tcp://10.0.0.1:0", transport_security="trusted_fabric")


def test_a_hosts_reading_comes_back_parsed_and_bounded():
    config = _config(ADDRESS)
    runner = _Runner(SimpleNamespace(
        stdout="noise\n" + listener_generation.LINE_PREFIX + "abc123 age_s=5\n",
        stderr="", returncode=0))
    fields = listener_generation.probe(
        config, config.hosts[0], runner=runner, timeout=3.0)
    assert fields == {"gen": "abc123", "age_s": "5"}
    assert runner.timeout == 3.0
    assert "0A" in runner.scripts[0]            # the passive LISTEN read


@pytest.mark.parametrize(("result", "expected"), [
    (SimpleNamespace(stdout="", stderr="", returncode=255), "exit_255"),
    (SimpleNamespace(stdout="nothing useful", stderr="", returncode=0), "no_line"),
    (OSError("no route"), "OSError"),
])
def test_a_probe_that_could_not_read_reports_it_and_never_raises(result, expected):
    config = _config(ADDRESS)
    fields = listener_generation.probe(
        config, config.hosts[0], runner=_Runner(result))
    assert fields == {"gen": "unknown", "error": expected}


def test_the_fleet_reading_is_keyed_by_worker_address():
    config = _config(ADDRESS, "tcp://10.0.0.2:26600")
    runner = _Runner(SimpleNamespace(
        stdout=listener_generation.LINE_PREFIX + "abc123", stderr="", returncode=0))
    assert set(listener_generation.fleet(config, runner=runner)) == {
        ADDRESS, "tcp://10.0.0.2:26600"}


def test_doctor_sends_the_snippet_and_prints_the_row():
    """Pinned on the source: the per-host heredoc is built inside the loop,
    and an unstubbed doctor run would probe this box."""
    source = inspect.getsource(doctor._doctor_rows)
    assert "listener_generation.probe_source(host.address)" in source
    assert "listener_generation.doctor_row(" in source
    assert "listener_generation.select(lines)" in source


@pytest.fixture
def cluster_rig(monkeypatch):
    """`_doctor_rows` with every probe stood in for: no torch, no socket off
    this box, no ssh.

    The row under test lands inside doctor's per-host loop, and the only way
    to read it as an operator does is to run that loop.
    """
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
                        lambda: {"status": OK, "name": "frontend", "detail": "stub"})
    monkeypatch.setattr(doctor, "_spark_health_rows", lambda _config: [])
    monkeypatch.setattr(doctor, "_mesh_health_row",
                        lambda: {"status": OK, "name": "mesh health", "detail": "stub"})
    monkeypatch.setattr(doctor.socket, "gethostname", lambda: "driver")
    monkeypatch.setattr(doctor.socket, "gethostbyname", lambda _name: "10.0.0.1")
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/rsync")
    monkeypatch.setattr(doctor, "_tcp_probe", lambda *_a, **_kw: False)
    monkeypatch.setattr(
        doctor, "passive_worker_health",
        lambda *_a, **_kw: {"running": True, "listening": True, "healthy": True})
    return monkeypatch


def _generation_row(rows: list[dict]) -> dict:
    return next(row for row in rows
                if row["name"].endswith(listener_generation.NAME_SUFFIX))


def _same_exit_without_it(rows: list[dict], row: dict) -> bool:
    """Whether dropping this row would have moved doctor's exit code."""
    from dgx_monarch.cli import probe_certainty

    rest = [item for item in rows if item is not row]
    return (probe_certainty.exit_code(rows, len(doctor._failures(rows)))
            == probe_certainty.exit_code(rest, len(doctor._failures(rest))))


def test_a_host_whose_remote_check_exited_nonzero_still_gets_its_row(
        cluster_rig, capsys):
    """The row matters most on a host that is unwell, so a combined probe that
    exits nonzero must not drop it."""
    cluster_rig.setattr(
        doctor, "run_on_host",
        lambda *_a, **_kw: SimpleNamespace(returncode=255, stdout="",
                                           stderr="ssh: no route to host"))
    rows = doctor._doctor_rows(_config(ADDRESS))
    capsys.readouterr()
    row = _generation_row(rows)
    assert row["status"] == WARN
    assert row["detail"].startswith("unobserved: remote check exited 255")
    assert row.get("critical") is not True
    assert _same_exit_without_it(rows, row)


def test_a_host_whose_ssh_raised_still_gets_its_row(cluster_rig, capsys):
    def refuse(*_args, **_kwargs):
        raise OSError("no route")

    cluster_rig.setattr(doctor, "run_on_host", refuse)
    rows = doctor._doctor_rows(_config(ADDRESS))
    capsys.readouterr()
    row = _generation_row(rows)
    assert row["status"] == WARN
    assert "ssh failed: OSError" in row["detail"]
    assert row["reason"] == UNOBSERVED
    assert row.get("critical") is not True
    assert _same_exit_without_it(rows, row)


def test_the_row_a_failed_host_owes_reads_like_any_other_unobserved_one():
    status, name, detail, reason, critical = listener_generation.unobserved_row(
        HOST, "ssh failed: OSError")
    assert (status, name) == (WARN, "h1 listener generation")
    assert detail.startswith("unobserved: ssh failed: OSError")
    assert listener_generation.ENTRY in detail
    assert (reason, critical) == (UNOBSERVED, False)
