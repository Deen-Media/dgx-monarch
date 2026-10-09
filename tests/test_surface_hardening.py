"""Regression tests for config, operator, TUI, and browser hardening."""

from __future__ import annotations

import asyncio
import importlib.metadata
import subprocess
import sys
import types
from pathlib import Path

import pytest

from dgx_monarch import TORCHMONARCH_PIN
from dgx_monarch.cli import doctor, lifecycle
from dgx_monarch.config import ClusterConfig, HostConfig
from dgx_monarch.tui.data import Poller


def _stub_driver_checks(monkeypatch, health=()):
    torch = types.ModuleType("torch")
    torch.__version__ = "2.test"
    torch.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: TORCHMONARCH_PIN)
    monkeypatch.setattr(
        doctor,
        "_frontend_skew_row",
        lambda: {"status": doctor._OK, "name": "frontend", "detail": "stub"},
    )
    monkeypatch.setattr(doctor, "_spark_health_rows", lambda _config: list(health))
    # The mesh row reaches the driver over HTTP; stubbed like the frontend row
    # so a doctor test never depends on what is listening on this box.
    monkeypatch.setattr(
        doctor,
        "_mesh_health_row",
        lambda: {"status": doctor._OK, "name": "mesh health", "detail": "stub"},
    )
    monkeypatch.setattr(doctor.socket, "gethostname", lambda: "driver")
    monkeypatch.setattr(doctor.socket, "gethostbyname", lambda _name: "10.0.0.1")


def test_doctor_normalizes_nccl_proto_lists():
    assert doctor._normalize_nccl_proto(" simple, ll128 ") == "SIMPLE,LL128"
    assert doctor._uses_nccl_ll(" simple, Ll ")
    assert not doctor._uses_nccl_ll("LL128,SIMPLE")


def test_doctor_requires_explicit_trusted_fabric(capsys):
    host = HostConfig(name="worker", address="tcp://10.0.0.2:26600")
    row = doctor._transport_security_row(ClusterConfig(hosts=(host,)))
    assert row["status"] == doctor._FAIL
    assert "no peer authentication" in capsys.readouterr().out

    row = doctor._transport_security_row(ClusterConfig(hosts=(host,), transport_security="trusted_fabric"))
    assert row["status"] == doctor._OK


def test_doctor_transport_rows_scope_the_boundary_to_the_whole_interface():
    """SECURITY.md rules out the port-scoped reading: 26600 is only the Worker
    service listener and spawned actors allocate secondary dynamic ports, so a
    rule for that port isolates nothing. Every transport row must ask for the
    interface, and must say where the requirement is written."""
    host = HostConfig(name="worker", address="tcp://10.0.0.2:26600")
    public = HostConfig(name="public", address="tcp://8.8.8.8:26600")
    rows = [
        doctor._transport_security_row(ClusterConfig(hosts=(host,))),
        doctor._transport_security_row(ClusterConfig(
            hosts=(public,), transport_security="trusted_fabric")),
        doctor._transport_security_row(ClusterConfig(
            hosts=(host,), transport_security="trusted_fabric")),
    ]
    assert {row["status"] for row in rows} == {doctor._FAIL, doctor._WARN, doctor._OK}
    for row in rows:
        detail = row["detail"].lower()
        assert "interface" in detail
        assert "source-restrict" in detail
        assert "security.md" in detail


def test_doctor_closed_port_and_remote_nonzero_are_failures(monkeypatch, capsys):
    _stub_driver_checks(monkeypatch)
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/rsync")
    raw_probes = []
    monkeypatch.setattr(
        doctor,
        "_tcp_probe",
        lambda address, **_kwargs: raw_probes.append(address) or False,
    )
    monkeypatch.setattr(
        doctor,
        "passive_worker_health",
        lambda *_args, **_kwargs: {
            "running": False,
            "listening": False,
            "healthy": False,
        },
    )
    monkeypatch.setattr(
        doctor,
        "run_on_host",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 23, "", "ssh failed"),
    )
    config = ClusterConfig(
        hosts=(HostConfig(name="worker", address="tcp://10.0.0.2:26600"),),
        client_bind="tcp://10.0.0.1:0",
        transport_security="trusted_fabric",
        source="test.toml",
    )
    assert not doctor.run_doctor(config)
    out = capsys.readouterr().out
    assert "worker worker service: tcp://10.0.0.2:26600 process=stopped" in out
    assert "listener=not listening" in out
    assert "remote check exited 23" in out
    assert raw_probes == [f"tcp://10.0.0.2:{config.nccl_master_port}"]


def test_doctor_rejects_open_port_without_worker_process(monkeypatch, capsys):
    _stub_driver_checks(monkeypatch)
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/rsync")
    monkeypatch.setattr(
        doctor,
        "_tcp_probe",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        doctor,
        "passive_worker_health",
        lambda *_args, **_kwargs: {
            "running": False,
            "listening": True,
            "healthy": False,
        },
    )
    output = f"torchmonarch={TORCHMONARCH_PIN} torch=2.test comfy=True nccl_proto=unset\nloop=stopped\n"
    monkeypatch.setattr(
        doctor,
        "run_on_host",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, output, ""),
    )
    config = ClusterConfig(
        hosts=(HostConfig(name="worker", address="tcp://10.0.0.2:26600"),),
        client_bind="tcp://10.0.0.1:0",
        transport_security="trusted_fabric",
        source="test.toml",
    )
    assert not doctor.run_doctor(config)
    assert "process=stopped listener=listening" in capsys.readouterr().out


def test_doctor_runs_portable_health_checks_in_local_mode(monkeypatch):
    health = [{"status": doctor._FAIL, "name": "portable", "detail": "broken"}]
    _stub_driver_checks(monkeypatch, health=health)
    assert not doctor.run_doctor(None)


def test_sync_uses_remote_user_managed_path(monkeypatch, tmp_path):
    host = HostConfig(name="worker", address="tcp://10.0.0.2:26600", ssh_user="alice")
    config = ClusterConfig(hosts=(host,))
    remote_scripts = []
    commands = []
    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)
    monkeypatch.setattr(lifecycle, "_pkg_src_dir", lambda: tmp_path / "checkout" / "src")

    def run_host(_config, _host, script, timeout=60):
        remote_scripts.append(script)
        return subprocess.CompletedProcess([], 0, "", "")

    def run_command(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(lifecycle, "run_on_host", run_host)
    monkeypatch.setattr(lifecycle.subprocess, "run", run_command)
    assert lifecycle.sync_package(config, host)
    assert len(remote_scripts) == 2 and "dgxm-setup-token" in remote_scripts[0]
    assert 'mkdir -p "$HOME/.local/share/dgx-monarch/src"' in remote_scripts[1]
    assert "--rsync-path" in commands[0] and "lifecycle.lock" in commands[0][commands[0].index("--rsync-path") + 1]
    assert commands[0][-1] == "alice@worker:~/.local/share/dgx-monarch/src/"


def test_sync_does_not_overwrite_a_setup_managed_release(monkeypatch):
    host = HostConfig(name="worker", address="tcp://10.0.0.2:26600")
    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)
    scripts = []
    monkeypatch.setattr(
        lifecycle,
        "run_on_host",
        lambda _config, _host, script, timeout=60: (
            scripts.append(script) or subprocess.CompletedProcess([], 0, "MANAGED\n", "")
        ),
    )
    monkeypatch.setattr(
        lifecycle.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("managed source must not be rsynced")),
    )
    assert lifecycle.sync_package(ClusterConfig(hosts=(host,)), host)
    assert len(scripts) == 1 and "dgxm-setup-token" in scripts[0]


def test_sync_stops_when_remote_mkdir_fails(monkeypatch):
    host = HostConfig(name="worker", address="tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,))
    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)
    monkeypatch.setattr(
        lifecycle,
        "run_on_host",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, "", "denied"),
    )
    monkeypatch.setattr(
        lifecycle.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("rsync must not run")),
    )
    assert lifecycle.sync_package(config, host) is False


def test_install_systemd_respects_sync_failure(monkeypatch):
    host = HostConfig(name="worker", address="tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    scripts = []
    monkeypatch.setattr(lifecycle, "sync_package", lambda *_args: False)
    monkeypatch.setattr(
        lifecycle,
        "run_on_host",
        lambda _config, _host, script, **_kwargs: scripts.append(script)
        or subprocess.CompletedProcess([], 0, "UNIT_ABSENT\n", ""),
    )
    assert lifecycle.install_systemd(config) is False
    assert len(scripts) == 1
    assert "enable --now" not in scripts[0]


def test_remote_systemd_unit_uses_managed_pythonpath(monkeypatch):
    host = HostConfig(name="worker", address="tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    scripts = []
    monkeypatch.setattr(lifecycle, "sync_package", lambda *_args: True)
    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)

    responses = iter((
        subprocess.CompletedProcess([], 0, "UNIT_ABSENT\n", ""),
        subprocess.CompletedProcess([], 0, "STATE=active\nLINGER=yes\n", ""),
    ))

    def run_host(_config, _host, script, timeout=60):
        scripts.append(script)
        return next(responses)

    monkeypatch.setattr(lifecycle, "run_on_host", run_host)
    assert lifecycle.install_systemd(config)
    assert 'Environment="PYTHONPATH=%h/.local/share/dgx-monarch/src"' in scripts[1]
    assert '"PYTHONDONTWRITEBYTECODE=1"' in scripts[1]
    assert '"PYTHONPYCACHEPREFIX=/proc/self/fd/2147483647"' in scripts[1]
    # A worker started before the fabric NIC is up crash-loops with no diagnosis.
    # After= only orders the unit against network-online.target; Wants= also
    # pulls that target in.
    assert "Wants=network-online.target" in scripts[1]
    assert "After=network-online.target" in scripts[1]
    assert "UMask=0077" in scripts[1]


def test_tcp_probe_supports_ipv6(monkeypatch):
    seen = []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        lifecycle.socket,
        "create_connection",
        lambda endpoint, timeout: seen.append((endpoint, timeout)) or Connection(),
    )
    assert lifecycle._tcp_probe("tcp://[fd00::2]:26600", timeout=0.2)
    assert seen == [(("fd00::2", 26600), 0.2)]


def test_tcp_probe_reports_a_timeout_as_unknown(monkeypatch):
    """A connect that timed out observed nothing, so it returns None, not False.

    Doctor prints False as `free`, so a firewalled or wedged host would read as a
    clean NCCL master port.
    """
    def refuse(_endpoint, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(lifecycle.socket, "create_connection", refuse)
    assert lifecycle._tcp_probe("tcp://10.0.0.2:29500", timeout=0.2) is None

    def refused(_endpoint, timeout):
        raise ConnectionRefusedError("refused")

    monkeypatch.setattr(lifecycle.socket, "create_connection", refused)
    assert lifecycle._tcp_probe("tcp://10.0.0.2:29500", timeout=0.2) is False


def test_ipv6_loopback_host_is_local():
    assert lifecycle._is_local(HostConfig(name="::1", address="tcp://[::1]:26600"))


def test_poller_loads_loop_endpoints_from_cluster_config(monkeypatch, tmp_path):
    config = tmp_path / "cluster.toml"
    config.write_text(
        '[cluster]\ntransport_security = "trusted_fabric"\n\n'
        '[[hosts]]\nname = "worker"\naddress = "tcp://[fd00::2]:26600"\n'
    )
    monkeypatch.setenv("DGXM_CLUSTER_TOML", str(config))
    poller = Poller()
    assert poller.loops == ("tcp://[fd00::2]:26600",)


def test_poller_has_no_lab_specific_loop_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("DGXM_CLUSTER_TOML", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert Poller().loops == ()


def test_textual_polling_offloads_blocking_tick(monkeypatch):
    rich = types.ModuleType("rich")
    rich_text = types.ModuleType("rich.text")
    textual = types.ModuleType("textual")
    textual_app = types.ModuleType("textual.app")
    textual_binding = types.ModuleType("textual.binding")
    textual_widgets = types.ModuleType("textual.widgets")

    class Text:
        pass

    class App:
        pass

    class Binding:
        def __init__(self, *_args, **_kwargs):
            pass

    class Static:
        pass

    rich.text = rich_text
    rich_text.Text = Text
    textual_app.App = App
    textual_binding.Binding = Binding
    textual_widgets.Static = Static
    monkeypatch.setitem(sys.modules, "rich", rich)
    monkeypatch.setitem(sys.modules, "rich.text", rich_text)
    monkeypatch.setitem(sys.modules, "textual", textual)
    monkeypatch.setitem(sys.modules, "textual.app", textual_app)
    monkeypatch.setitem(sys.modules, "textual.binding", textual_binding)
    monkeypatch.setitem(sys.modules, "textual.widgets", textual_widgets)

    from dgx_monarch.tui import app as tui_app

    assert tui_app._loop_indicator(None, {"good": "green", "bad": "red"}) == ("?", "dim")
    assert tui_app._loop_indicator(False, {"good": "green", "bad": "red"}) == ("●", "red")
    application = tui_app.DgxmTopApp()
    application.poller = types.SimpleNamespace(
        tick=lambda: {
            "t": 1.0,
            "events": [],
            "hosts": {},
            "loops": [],
            "driver_up": False,
        }
    )
    calls = []

    async def to_thread(function, *args):
        calls.append(function)
        return function(*args)

    monkeypatch.setattr(tui_app.asyncio, "to_thread", to_thread)
    asyncio.run(application.refresh_data())
    assert calls == [application.poller.tick]
    assert len(application.ring.ticks) == 1


def test_browser_panel_avoids_html_interpolation_and_accounts_slab():
    web = Path(__file__).parents[1] / "web" / "js"
    panel = (web / "dgx_monarch_panel.js").read_text()
    segments = (web / "dgx_monarch_segments.js").read_text()
    assert "innerHTML" not in panel
    assert 'from "./dgx_monarch_segments.js"' in panel
    assert "slab_gib" in segments
    assert "slab ${slab.toFixed(1)}G" in panel


def test_setup_script_is_fail_fast_and_python_version_independent():
    script = Path(__file__).parents[1] / "scripts/setup_env.sh"
    source = script.read_text()
    assert "set -euo pipefail" in source
    assert "sysconfig.get_paths()" in source
    assert "/lib/python3.12/site-packages" not in source
    result = subprocess.run(["bash", str(script), "--invalid"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "unknown argument" in result.stderr


@pytest.mark.parametrize(
    ("failure", "expected", "copy"),
    [
        (subprocess.TimeoutExpired("rsync", 120), None, "UNKNOWN (package sync unobserved"),
        (OSError("rsync: command not found"), False, "FAILED (package sync"),
    ],
)
def test_a_package_sync_nobody_watched_end_is_unknown_not_failed(
    monkeypatch, capsys, failure, expected, copy
):
    """The 120 s timeout kills the local rsync; the remote one is unobserved.

    So `up` prints UNKNOWN, not FAILED (cli/lifecycle_host.sync_outcome), as it
    does for a lost transport on the start command. An OSError, such as a
    missing rsync, is a definite failure.
    """
    host = HostConfig(name="worker", address="tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)
    monkeypatch.setattr(
        lifecycle, "run_on_host",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(
        lifecycle.subprocess, "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure))

    assert lifecycle.sync_package(config, host) is expected
    assert lifecycle.up(config) is expected
    out = capsys.readouterr().out
    assert copy in out
    assert "timed out" not in out


def test_a_transport_that_died_during_the_remote_mkdir_is_unknown(monkeypatch, capsys):
    """SSH exit 255 is the transport, not the command; 1 is the command."""
    host = HostConfig(name="worker", address="tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)
    monkeypatch.setattr(
        lifecycle.subprocess, "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("rsync must not run")))
    for returncode, expected in ((255, None), (1, False)):
        monkeypatch.setattr(
            lifecycle, "run_on_host",
            lambda *_args, _rc=returncode, **_kwargs: subprocess.CompletedProcess(
                [], _rc, "", "lost"))
        assert lifecycle.sync_package(config, host) is expected
    capsys.readouterr()


def test_one_unobserved_sync_dominates_a_definitely_failed_one(monkeypatch, capsys):
    """One unobserved host makes the fleet result unknown, even beside a definite failure."""
    hosts = (
        HostConfig(name="w1", address="tcp://10.0.0.2:26600"),
        HostConfig(name="w2", address="tcp://10.0.0.3:26600"),
    )
    config = ClusterConfig(hosts=hosts, transport_security="trusted_fabric")
    outcomes = {"w1": False, "w2": None}
    monkeypatch.setattr(
        lifecycle, "sync_package", lambda _config, host: outcomes[host.name])

    assert lifecycle.up(config) is None
    out = capsys.readouterr().out
    assert "w1: FAILED (package sync" in out
    assert "w2: UNKNOWN (package sync unobserved" in out
