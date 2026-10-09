"""doctor's `hostname resolution` row asks one question: will the client
transport advertise loopback? A configured `cluster.client_bind` answers it:
`mesh_attach.attach_cluster` hands that bind to `enable_transport` and the
hostname never enters the decision, so the row reads the loaded config instead
of warning on every loopback hostname."""
from __future__ import annotations

import importlib.metadata
import subprocess
import sys
import types

from dgx_monarch import TORCHMONARCH_PIN
from dgx_monarch.cli import doctor
from dgx_monarch.cli.advertised_address import (
    hostname_row,
    pinned_fabric_ip,
    unresolved_row,
)
from dgx_monarch.config import ClusterConfig, HostConfig

FABRIC = ClusterConfig(client_bind="tcp://10.0.0.1:0")
_REMOTE_OK = (
    f"torchmonarch={TORCHMONARCH_PIN} torch=2.test comfy=True nccl_proto=unset\n"
    "fabric_iface=up rdma_active=1\n"
)


def _stub_driver(monkeypatch, resolved):
    """Mirror tests/test_surface_hardening.py::_stub_driver_checks, with the
    resolved address as the one value this module varies."""
    torch = types.ModuleType("torch")
    torch.__version__ = "2.test"
    torch.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: TORCHMONARCH_PIN)
    monkeypatch.setattr(
        doctor, "_frontend_skew_row",
        lambda: {"status": doctor._OK, "name": "frontend", "detail": "stub"},
    )
    monkeypatch.setattr(doctor, "_spark_health_rows", lambda _config: [])
    monkeypatch.setattr(
        doctor, "_mesh_health_row",
        lambda: {"status": doctor._OK, "name": "mesh health", "detail": "stub"},
    )
    monkeypatch.setattr(doctor.socket, "gethostname", lambda: "driver")
    monkeypatch.setattr(doctor.socket, "gethostbyname", lambda _name: resolved)


def test_configured_fabric_bind_clears_the_loopback_warning():
    ok, detail = hostname_row("driver", "127.0.1.1", FABRIC)
    assert ok
    assert "client_bind pins 10.0.0.1" in detail
    assert "never resolves the hostname" in detail
    assert "127.0.1.1 (loopback)" in detail


def test_loopback_hostname_without_a_bind_still_warns():
    for config in (None, ClusterConfig(), ClusterConfig(client_bind="")):
        ok, detail = hostname_row("driver", "127.0.1.1", config)
        assert not ok, config
        assert "Multi-host needs cluster.client_bind on the fabric IP" in detail


def test_a_loopback_client_bind_does_not_count_as_the_remediation():
    config = ClusterConfig(client_bind="tcp://127.0.0.1:0")
    assert pinned_fabric_ip(config) == ""
    ok, detail = hostname_row("driver", "127.0.1.1", config)
    assert not ok
    assert "Multi-host needs cluster.client_bind on the fabric IP" in detail


def test_ipv4_mapped_binds_never_count_as_the_fabric_pin():
    # These are the IPv4-mapped unspecified, loopback and multicast addresses.
    # None may count as the fabric pin this row looks for.
    for host in ("::ffff:0.0.0.0", "::ffff:127.0.0.1", "::ffff:224.0.0.1"):
        config = ClusterConfig(client_bind=f"tcp://[{host}]:0")
        assert pinned_fabric_ip(config) == "", host
        ok, detail = hostname_row("driver", "127.0.1.1", config)
        assert not ok, host
        assert "Multi-host needs cluster.client_bind on the fabric IP" in detail


def test_a_mapped_loopback_resolution_reads_as_loopback():
    ok, detail = hostname_row("driver", "::ffff:127.0.0.1", None)
    assert not ok
    assert "(loopback)" in detail


def test_a_plain_fabric_bind_is_still_the_remediation():
    assert pinned_fabric_ip(FABRIC) == "10.0.0.1"
    assert pinned_fabric_ip(ClusterConfig(client_bind="tcp://[2001:db8::2]:0")) == "2001:db8::2"


def test_an_unparseable_bind_does_not_count_as_the_remediation():
    # tcp_endpoint raises ValueError; the guard in doctor._doctor_rows catches only OSError.
    assert pinned_fabric_ip(ClusterConfig(client_bind="not-a-url")) == ""
    assert not hostname_row("driver", "127.0.1.1", ClusterConfig(client_bind="not-a-url"))[0]


def test_a_routable_hostname_keeps_the_plain_ok_row():
    for config in (None, FABRIC):
        assert hostname_row("driver", "10.0.0.1", config) == (True, "driver -> 10.0.0.1")


def test_ipv6_loopback_is_classified_as_loopback():
    assert not hostname_row("driver", "::1", None)[0]


def test_doctor_reports_ok_when_client_bind_covers_a_loopback_hostname(monkeypatch, capsys):
    _stub_driver(monkeypatch, "127.0.1.1")
    doctor.run_doctor(FABRIC)
    output = capsys.readouterr().out
    assert "  [ ok ] hostname resolution: driver -> 127.0.1.1 (loopback)" in output
    assert "  [WARN] hostname resolution" not in output


def test_doctor_still_warns_in_an_unconfigured_local_install(monkeypatch, capsys):
    _stub_driver(monkeypatch, "127.0.1.1")
    doctor.run_doctor(None)
    output = capsys.readouterr().out
    assert "  [WARN] hostname resolution" in output
    assert "cluster checks: skipped" in output


def test_a_failed_lookup_is_cleared_by_the_same_evidence():
    # Minimal images keep the box's own hostname out of /etc/hosts and DNS.
    # A pinned bind makes that as harmless as a loopback answer.
    ok, detail = unresolved_row(FABRIC, OSError("no such host"))
    assert ok
    assert "hostname does not resolve (OSError('no such host'))" in detail
    assert "client_bind pins 10.0.0.1" in detail


def test_a_failed_lookup_without_a_bind_still_warns():
    for config in (None, ClusterConfig(), ClusterConfig(client_bind="tcp://127.0.0.1:0")):
        ok, detail = unresolved_row(config, OSError("no such host"))
        assert not ok, config
        assert "hostname does not resolve" in detail
        assert "Multi-host needs cluster.client_bind on the fabric IP" in detail


def test_doctor_clears_the_row_when_the_lookup_raises_under_a_bind(monkeypatch, capsys):
    _stub_driver(monkeypatch, "10.0.0.1")

    def _boom(_name):
        raise OSError("no such host")

    monkeypatch.setattr(doctor.socket, "gethostbyname", _boom)
    doctor.run_doctor(FABRIC)
    output = capsys.readouterr().out
    assert output.count("hostname resolution") == 1
    assert "  [ ok ] hostname resolution: hostname does not resolve" in output
    assert "  [WARN] hostname resolution" not in output


def test_doctor_still_warns_when_the_lookup_raises_unconfigured(monkeypatch, capsys):
    _stub_driver(monkeypatch, "10.0.0.1")

    def _boom(_name):
        raise OSError("no such host")

    monkeypatch.setattr(doctor.socket, "gethostbyname", _boom)
    doctor.run_doctor(None)
    output = capsys.readouterr().out
    assert "  [WARN] hostname resolution: hostname does not resolve" in output


def test_one_report_never_contradicts_itself_about_the_bind(monkeypatch, capsys):
    """The reported symptom was one report carrying both `[WARN] ... needs
    client_bind` and `[ ok ] client_bind: tcp://...`. Render the cluster
    section so both rows print, and hold them to the same answer."""
    _stub_driver(monkeypatch, "127.0.1.1")
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/rsync")
    monkeypatch.setattr(doctor, "_tcp_probe", lambda *_a, **_kw: False)
    monkeypatch.setattr(
        doctor, "passive_worker_health",
        lambda *_a, **_kw: {"running": True, "listening": True, "healthy": True},
    )
    monkeypatch.setattr(
        doctor, "run_on_host",
        lambda *_a, **_kw: subprocess.CompletedProcess([], 0, _REMOTE_OK, ""),
    )
    doctor.run_doctor(ClusterConfig(
        hosts=(HostConfig(name="worker", address="tcp://10.0.0.2:26600"),),
        client_bind="tcp://10.0.0.1:0",
        fabric_profile="generic-roce",
        transport_security="trusted_fabric",
        source="test.toml",
    ))
    output = capsys.readouterr().out
    assert "  [ ok ] hostname resolution: driver -> 127.0.1.1 (loopback)" in output
    assert "  [ ok ] client_bind: tcp://10.0.0.1:0" in output
    assert "  [WARN] hostname resolution" not in output


def test_cross_host_torch_version_skew_fails_doctor(monkeypatch, capsys):
    _stub_driver(monkeypatch, "10.0.0.1")
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/rsync")
    monkeypatch.setattr(doctor, "_tcp_probe", lambda *_a, **_kw: False)
    monkeypatch.setattr(
        doctor, "passive_worker_health",
        lambda *_a, **_kw: {"running": True, "listening": True, "healthy": True},
    )
    remote = _REMOTE_OK.replace("torch=2.test", "torch=2.worker")
    monkeypatch.setattr(
        doctor, "run_on_host",
        lambda *_a, **_kw: subprocess.CompletedProcess([], 0, remote, ""),
    )
    config = ClusterConfig(
        hosts=(HostConfig(name="worker", address="tcp://10.0.0.2:26600"),),
        client_bind="tcp://10.0.0.1:0",
        fabric_profile="generic-roce",
        transport_security="trusted_fabric",
        source="test.toml",
    )

    assert doctor.run_doctor(config) is False
    output = capsys.readouterr().out
    assert "[FAIL] torch version match" in output
    assert "driver=2.test, worker=2.worker" in output
