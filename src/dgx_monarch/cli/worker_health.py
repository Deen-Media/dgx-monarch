"""Protocol-safe worker-loop health checks.

Monarch owns the worker socket protocol: a connect followed by disconnect is
not a harmless TCP readiness probe.  Inspect the exact loop process and the
host's passive LISTEN table instead, using the lifecycle command runner that
the caller already trusts for local/SSH execution.
"""
from __future__ import annotations

import ipaddress
import re
import shlex
import subprocess
import time
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

from ..config import ClusterConfig, HostConfig, tcp_endpoint
from . import probe_certainty

_UNIT_NAME = "dgxm-worker.service"


class FleetHealth(NamedTuple):
    """Separate confirmed-down workers from workers whose state was not observed.

    Only confirmed-down workers authorize a restart.
    """

    dead: tuple[str, ...]
    unobserved: tuple[str, ...]


def _health_verdict(running: bool | None, listening: bool | None) -> bool | None:
    """Combine tri-state process and LISTEN observations.

    Both must be True for health; either False establishes failure. Otherwise
    return None, which cannot authorize a restart.
    """
    if running is True and listening is True:
        return True
    if running is False or listening is False:
        return False
    return None


def loop_pattern(host: HostConfig) -> str:
    """Stable worker argv tail, independent of the expanded Python path."""
    return f"dgx_monarch.cli.worker_loop --address {host.address}"


def loop_regex(host: HostConfig) -> str:
    """Anchored ERE for procps, including literal IPv6 brackets."""
    return re.escape(loop_pattern(host)) + r"$"


def _proc_listener_target(address: str) -> tuple[str, str]:
    """Return the proc table and encoded local endpoint for a TCP URL.

    ``/proc/net/tcp{,6}`` prints 32-bit address words in host byte order; this
    assumes a little-endian host (one reversed word for IPv4, four for IPv6).
    """
    host, port = tcp_endpoint(address)
    ip = ipaddress.ip_address(host)
    if ip.version == 4:
        proc_path = "/proc/net/tcp"
        encoded = ip.packed[::-1]
    else:
        proc_path = "/proc/net/tcp6"
        encoded = b"".join(
            ip.packed[offset:offset + 4][::-1] for offset in range(0, 16, 4)
        )
    return proc_path, f"{encoded.hex().upper()}:{port:04X}"


def passive_worker_health(
    config: ClusterConfig,
    host: HostConfig,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Inspect one loop without opening its protocol socket.

    ``running``, ``listening`` and ``healthy`` are all tri-state: ``None``
    means the host command or proc table could not provide an answer.  Only an
    exact process plus exact LISTEN endpoint is healthy, and only a host that
    reported one of them down is definitely not. ``pgrep`` exits 1 for "no
    match" and 2 or 3 for a fault, so only the 1 is read as a stopped loop: a
    broken or missing ``pgrep`` says nothing about the process.
    """
    proc_path, endpoint = _proc_listener_target(host.address)
    pattern = shlex.quote(loop_regex(host))
    script = f"""
UNIT="$HOME/.config/systemd/user/{_UNIT_NAME}"
if [ -f "$UNIT" ]; then
  echo MODE=systemd
  if ! systemctl --user is-active --quiet {_UNIT_NAME} && \
      pgrep -f -- {pattern} >/dev/null; then
    echo STRAY_NOHUP=1
  fi
else
  echo MODE=nohup
fi
if pgrep -f -- {pattern} >/dev/null; then
  echo PROCESS=running
elif [ $? -eq 1 ]; then
  echo PROCESS=stopped
else
  echo PROCESS=unknown
fi
if [ -r {shlex.quote(proc_path)} ]; then
  if awk -v endpoint={shlex.quote(endpoint)} \
      '$2 == endpoint && $4 == "0A" {{ found=1 }} END {{ exit !found }}' \
      {shlex.quote(proc_path)}; then
    echo LISTENER=listening
  else
    echo LISTENER=not_listening
  fi
else
  echo LISTENER=unknown
fi
"""
    try:
        result = runner(config, host, script, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "running": None,
            "listening": None,
            "healthy": None,
            "mode": "unknown",
            "error": type(exc).__name__,
        }
    if result.returncode != 0:
        return {
            "running": None,
            "listening": None,
            "healthy": None,
            "mode": "unknown",
            "error": f"host command exited {result.returncode}",
        }

    fields: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.strip().partition("=")
        if separator:
            fields[key] = value
    running = {"running": True, "stopped": False}.get(fields.get("PROCESS", ""))
    listening = {
        "listening": True,
        "not_listening": False,
        "unknown": None,
    }.get(fields.get("LISTENER", ""))
    mode = fields.get("MODE", "unknown")
    if fields.get("STRAY_NOHUP") == "1":
        mode = "systemd, STRAY nohup loop; run dgxm down then up"
    return {
        "running": running,
        "listening": listening,
        "healthy": _health_verdict(running, listening),
        "mode": mode,
        "error": None,
    }


def worker_service_row(
    host: str, address: str, health: Mapping[str, Any]
) -> tuple[str, str, str, str, bool]:
    """Return ``(status, name, detail, reason, critical)`` for a Worker service.

    A timeout or command that never ran warns with reason ``unobserved``; treating
    it as a definite failure could authorize an unsafe update rollback. An observed
    healthy loop is OK. An unhealthy loop or a reported unknown state is FAIL.
    """
    process = {True: "running", False: "stopped"}.get(health["running"], "unknown")
    listener = {True: "listening", False: "not listening"}.get(
        health["listening"], "unknown")
    status, reason = (
        (probe_certainty.OK, "") if health["healthy"] is True
        else probe_certainty.blind_row(probe_certainty.FAIL)
        if health.get("error")
        else (probe_certainty.FAIL, "")
    )
    return (
        status,
        f"{host} worker service",
        f"{address} process={process} listener={listener}",
        reason,
        bool(reason),  # An unreachable host leaves the overall result unknown.
    )


def passive_unhealthy_workers(
    config: ClusterConfig,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
    deadline: float,
    per_host_timeout: float = 3.0,
    clock: Callable[[], float] = time.monotonic,
) -> FleetHealth:
    """Split the fleet into definite-dead and never-observed, on one deadline.

    A host whose probe reached no verdict and a host the deadline left
    unprobed are the same answer, so they share the second collection. Neither
    is evidence that a loop is down.
    """
    dead: list[str] = []
    unobserved: list[str] = []
    hosts = tuple(config.hosts)
    for index, host in enumerate(hosts):
        remaining = deadline - clock()
        if remaining <= 0:
            unobserved.extend(item.address for item in hosts[index:])
            break
        health = passive_worker_health(
            config,
            host,
            runner=runner,
            timeout=min(per_host_timeout, remaining),
        )
        healthy = health["healthy"]
        if healthy is False:
            dead.append(host.address)
        elif healthy is not True:
            unobserved.append(host.address)
    return FleetHealth(tuple(dead), tuple(unobserved))
