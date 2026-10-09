"""Worker-loop health checks that never open the Monarch socket."""
from __future__ import annotations

import socket
import subprocess

import pytest

from dgx_monarch.cli import lifecycle
from dgx_monarch.cli.worker_health import (
    _proc_listener_target,
    passive_worker_health,
    worker_service_row,
)
from dgx_monarch.config import ClusterConfig, HostConfig


def _config(address: str = "tcp://10.0.0.2:26600") -> tuple[ClusterConfig, HostConfig]:
    host = HostConfig(name="worker", address=address)
    return ClusterConfig(hosts=(host,)), host


def test_proc_listener_target_encodes_ipv4_kernel_format():
    assert _proc_listener_target("tcp://10.0.0.2:26600") == (
        "/proc/net/tcp",
        "0200000A:67E8",
    )


def test_proc_listener_target_encodes_ipv6_kernel_format():
    assert _proc_listener_target("tcp://[fd00::2]:26600") == (
        "/proc/net/tcp6",
        "000000FD000000000000000002000000:67E8",
    )


@pytest.mark.parametrize(
    ("process", "listener", "expected"),
    [
        ("running", "listening", True),
        ("running", "not_listening", False),
        ("stopped", "listening", False),
        ("stopped", "not_listening", False),
        # A proc table this host could not read is not a definite answer: one
        # observed half up and the other unknown leaves the pair unknown.
        ("running", "unknown", None),
    ],
)
def test_passive_health_requires_process_listener_agreement(
    process: str,
    listener: str,
    expected: bool | None,
):
    config, host = _config()

    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            [], 0, f"MODE=nohup\nPROCESS={process}\nLISTENER={listener}\n", ""
        )

    health = passive_worker_health(config, host, runner=runner)
    assert health["healthy"] is expected


def test_passive_health_never_opens_the_monarch_socket(monkeypatch):
    config, host = _config("tcp://[fd00::2]:26600")
    scripts = []
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Monarch worker sockets are not readiness probes")
        ),
    )

    def runner(_config, _host, script, timeout=30):
        scripts.append((script, timeout))
        return subprocess.CompletedProcess(
            [], 0, "MODE=systemd\nPROCESS=running\nLISTENER=listening\n", ""
        )

    assert passive_worker_health(config, host, runner=runner)["healthy"]
    assert "/proc/net/tcp6" in scripts[0][0]
    assert "pgrep -f --" in scripts[0][0]
    assert scripts[0][0].count("echo PROCESS=running") == 1
    assert "elif [ $? -eq 1 ]; then" in scripts[0][0]
    assert "echo PROCESS=unknown" in scripts[0][0]
    assert "socket" not in scripts[0][0]


def test_passive_health_reports_failed_host_execution_as_unknown():
    config, host = _config()

    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 255, "", "ssh failed")

    assert passive_worker_health(config, host, runner=runner) == {
        "running": None,
        "listening": None,
        "healthy": None,
        "mode": "unknown",
        "error": "host command exited 255",
    }


def _fleet(count: int = 3) -> tuple[ClusterConfig, tuple[HostConfig, ...]]:
    hosts = tuple(
        HostConfig(name=f"worker-{index}", address=f"tcp://10.0.0.{index}:26600")
        for index in range(1, count + 1)
    )
    return ClusterConfig(hosts=hosts), hosts


def test_a_timed_out_fleet_pass_reports_nothing_dead():
    """Timeouts and the unprobed remainder are unknown, never unhealthy.

    A host in `dead` authorizes a Worker service restart during attach, so a
    fleet-wide SSH stall must not read as a dead fleet.
    """
    from dgx_monarch.cli.worker_health import passive_unhealthy_workers

    config, hosts = _fleet()
    now = [0.0]
    timeouts: list[float] = []

    def runner(_config, _host, _script, timeout):
        timeouts.append(timeout)
        now[0] += timeout
        raise subprocess.TimeoutExpired("worker health", timeout)

    health = passive_unhealthy_workers(
        config,
        runner=runner,
        deadline=5.0,
        per_host_timeout=3.0,
        clock=lambda: now[0],
    )

    # One fleet-wide budget: two probes fit and the third never runs.
    assert timeouts == [3.0, 2.0]
    assert now[0] == 5.0
    assert health.dead == ()
    assert health.unobserved == tuple(host.address for host in hosts)


def test_an_observed_stopped_loop_is_the_only_thing_that_lands_in_dead():
    """Definite-dead and unobserved reach the caller as separate collections."""
    from dgx_monarch.cli.worker_health import passive_unhealthy_workers

    config, hosts = _fleet()
    answers = {
        hosts[0].address: "MODE=nohup\nPROCESS=stopped\nLISTENER=not_listening\n",
        hosts[1].address: "MODE=systemd\nPROCESS=running\nLISTENER=listening\n",
    }

    def runner(_config, host, _script, timeout=30):
        stdout = answers.get(host.address)
        if stdout is None:
            return subprocess.CompletedProcess([], 255, "", "ssh failed")
        return subprocess.CompletedProcess([], 0, stdout, "")

    health = passive_unhealthy_workers(
        config, runner=runner, deadline=5.0, clock=lambda: 0.0)

    assert health.dead == (hosts[0].address,)
    assert health.unobserved == (hosts[2].address,)


def test_lifecycle_status_does_not_raw_connect_to_worker(monkeypatch, capsys):
    config, _host = _config()
    monkeypatch.setattr(
        lifecycle,
        "_tcp_probe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("status must use passive worker health")
        ),
    )
    monkeypatch.setattr(
        lifecycle,
        "passive_worker_health",
        lambda *_args, **_kwargs: {
            "running": False,
            "listening": True,
            "healthy": False,
            "mode": "nohup",
        },
    )

    assert lifecycle.status(config) == [{
        "host": "worker",
        "loop": "stopped",
        "port": "open",
        "address": "tcp://10.0.0.2:26600",
        "mode": "nohup",
    }]
    assert "worker_service=STOPPED listener=open" in capsys.readouterr().out


def test_a_blind_host_probe_warns_instead_of_failing_the_doctor_row():
    """Unknown evidence must not read as a definite worker failure.

    Doctor's exit code is a verified update's authority to restore the prior
    release. A flaky SSH hop that never reached the host proves nothing about
    the loop, so it must not trigger that rollback.
    """
    config, host = _config()

    def timed_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("ssh", 30)

    blind = passive_worker_health(config, host, runner=timed_out)
    status, name, detail, reason, critical = worker_service_row(
        "w1", host.address, blind)
    # Critical: a host this run could not reach makes the whole run
    # inconclusive, unlike an advisory local probe that could not run.
    assert (status, reason, critical) == ("WARN", "unobserved", True)
    assert name == "w1 worker service"
    assert "process=unknown listener=unknown" in detail

    def exited_255(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 255, "", "ssh failed")

    assert worker_service_row("w1", host.address, passive_worker_health(
        config, host, runner=exited_255))[::3] == ("WARN", "unobserved")


def test_an_observed_stopped_loop_is_still_a_definite_failure():
    """The tri-state must not soften a loop the host reported as down."""
    config, host = _config()

    def stopped(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            [], 0, "MODE=nohup\nPROCESS=stopped\nLISTENER=not_listening\n", "")

    health = passive_worker_health(config, host, runner=stopped)
    status, _name, detail, reason, critical = worker_service_row(
        "w1", host.address, health)
    assert (status, reason, critical) == ("FAIL", "", False)
    assert "process=stopped listener=not listening" in detail

    def healthy(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            [], 0, "MODE=systemd\nPROCESS=running\nLISTENER=listening\n", "")

    assert worker_service_row("w1", host.address, passive_worker_health(
        config, host, runner=healthy))[::3] == ("ok", "")


def _emitted_script(config, host) -> str:
    captured: list[str] = []

    def runner(_config, _host, script, timeout=30):
        captured.append(script)
        return subprocess.CompletedProcess([], 0, "MODE=nohup\n", "")

    passive_worker_health(config, host, runner=runner)
    return captured[0]


@pytest.mark.parametrize(
    ("pgrep_exit", "expected"),
    [(0, "PROCESS=running"), (1, "PROCESS=stopped"), (2, "PROCESS=unknown"),
     (3, "PROCESS=unknown"), (127, "PROCESS=unknown")],
)
def test_the_remote_script_maps_a_pgrep_fault_to_unknown(
    tmp_path, pgrep_exit: int, expected: str,
):
    """Run the real script text against a pgrep that exits each way.

    `pgrep` exits 1 for no match and 2 or 3 for a fault. Never write
    `&& running || stopped`: it reports every fault as a stopped loop, which is
    the evidence an auto-heal restart reads.
    """
    config, host = _config()
    stub = tmp_path / "pgrep"
    stub.write_text(f"#!/bin/sh\nexit {pgrep_exit}\n")
    stub.chmod(0o755)
    result = subprocess.run(
        ["/bin/sh", "-c", _emitted_script(config, host)],
        capture_output=True, text=True, timeout=30,
        env={"PATH": f"{tmp_path}:/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert expected in result.stdout, result.stdout


def test_an_unknown_process_line_parses_as_no_answer():
    config, host = _config()

    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            [], 0, "MODE=nohup\nPROCESS=unknown\nLISTENER=listening\n", "")

    health = passive_worker_health(config, host, runner=runner)
    assert health["running"] is None
    assert health["healthy"] is not True
