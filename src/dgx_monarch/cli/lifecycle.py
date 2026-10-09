"""Worker-loop lifecycle: up/down/status/restart and systemd installation."""

from __future__ import annotations

import ipaddress
import shlex
import socket
import subprocess
import time
from pathlib import Path

from ..config import ClusterConfig, HostConfig, tcp_endpoint
from ..log import get_logger
from . import actor_sweep
from .lifecycle_generation import invalidate_generation_command_shell
from .lifecycle_host import LifecycleResult, owned_listener_probe_shell, start_preflight, sync_outcome, sync_verdict
from .lifecycle_host import combine_result as _combine_result
from .lifecycle_host import is_local as _is_local
from .lifecycle_host import terminal_marker_result as _terminal_marker_result
from .lifecycle_host import transport_acknowledged as _transport_acknowledged
from .lifecycle_host import validate_stop_arguments as _validate_stop_arguments
from .lifecycle_host import validated_ssh_identity as _validated_ssh_identity
from .lifecycle_lock import locked_rsync_path, locked_script
from .lifecycle_managed import prepare_source_sync
from .lifecycle_scripts import build_start_script, build_stop_script
from .lifecycle_systemd import MANAGED_SRC_REL as _MANAGED_SRC_REL
from .lifecycle_systemd import UNIT_NAME as _UNIT_NAME
from .lifecycle_systemd import UNIT_OWNERSHIP_MARKER
from .lifecycle_systemd import package_src_dir as _pkg_src_dir
from .lifecycle_systemd import systemd_worker_unit as _systemd_worker_unit
from .lifecycle_systemd import unit_ownership_probe as _unit_ownership_probe
from .lifecycle_systemd import unit_publish_script as _unit_publish_script
from .lifecycle_systemd import unit_refusal as _unit_refusal
from .lifecycle_systemd import unit_removal_leftovers as _unit_removal_leftovers
from .lifecycle_systemd import unit_remove_script as _unit_remove_script
from .lifecycle_systemd import validate_unit_arguments as _validate_unit_arguments
from .worker_health import loop_regex as _loop_regex
from .worker_health import passive_worker_health

log = get_logger(__name__)
_listener_probe_shell = owned_listener_probe_shell

_LOG_PATH = "$HOME/.local/state/dgx-monarch/worker-loop.log"
_UNIT_OWNERSHIP_MARKER = UNIT_OWNERSHIP_MARKER

def _ssh_base(config: ClusterConfig, host: HostConfig) -> list[str]:
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "LogLevel=ERROR"]
    if config.ssh_key:
        cmd += ["-i", str(Path(config.ssh_key).expanduser())]
    name, user = _validated_ssh_identity(host)
    target = f"{user}@{name}" if user else name
    return [*cmd, "--", target]


def _rsync_target(host: HostConfig) -> str:
    """Rsync's remote-shell syntax needs brackets around literal IPv6 hosts."""
    name, user = _validated_ssh_identity(host)
    try:
        if ipaddress.ip_address(name).version == 6:
            name = f"[{name}]"
    except ValueError:
        pass
    return f"{user}@{name}" if user else name


def _rsync_remote_shell(config: ClusterConfig, host: HostConfig) -> str:
    """Rsync adds the login options and host itself; drop the trailing `--` and target."""
    return shlex.join(_ssh_base(config, host)[:-2])


def run_on_host(
    config: ClusterConfig, host: HostConfig, script: str, timeout: int = 60,
    *, require_known_locality: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a stdin-delivered shell script locally or through SSH."""
    locality = _is_local(host)
    if locality is None and require_known_locality:
        raise OSError("Worker host locality is unknown")
    if locality:
        return subprocess.run(["bash", "-s"], input=script, capture_output=True, text=True, timeout=timeout)
    return subprocess.run(
        [*_ssh_base(config, host), "bash", "-s"],
        input=script,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _loop_args(host: HostConfig) -> str:
    """Shell-safe args after the Python binary; also the process-match tail."""
    return f"-m dgx_monarch.cli.worker_loop --address {shlex.quote(host.address)}"


def _pybin_shell(python_bin: str) -> str:
    return (
        f"PYBIN={shlex.quote(python_bin)}\n"
        'case "$PYBIN" in "~/"*) PYBIN="$HOME/${PYBIN#\\~/}";; "~") PYBIN="$HOME";; esac'
    )


def sync_package(config: ClusterConfig, host: HostConfig) -> LifecycleResult:
    """Sync the package; return None if transport loss leaves the outcome unknown."""
    _validated_ssh_identity(host)  # the refusal needs no lookup, so it comes first
    locality = _is_local(host)
    if locality is None:
        return None
    try:
        needed, prepared = prepare_source_sync(config, host, locality, run_on_host)
        if not needed or prepared is not True:
            return prepared
    except (OSError, subprocess.TimeoutExpired) as exc:
        return sync_outcome(exc, host.name, "mkdir/probe", log)
    src = _pkg_src_dir()
    ssh_cmd = _rsync_remote_shell(config, host)
    target = _rsync_target(host)
    dest = f"{target}:~/{_MANAGED_SRC_REL}/"
    try:
        result = subprocess.run(
            [
                "rsync",
                "-a",
                "--delete",
                "--delete-excluded",
                "--exclude=__pycache__/",
                "--exclude=*.pyc",
                "--exclude=*.pyo",
                "--include=/dgx_monarch/",
                "--include=/dgx_monarch/***",
                "--exclude=*",
                "-e",
                ssh_cmd,
                "--rsync-path",
                locked_rsync_path(config.python_bin, _MANAGED_SRC_REL),
                "--",
                f"{src}/",
                dest,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return sync_outcome(exc, host.name, "rsync", log)
    if result.returncode != 0:
        log.warning("rsync to %s failed: %s", host.name, result.stderr.strip())
        return None if result.returncode == 255 else False
    return True


def up(
    config: ClusterConfig, sync: bool = True, *, generation: str | None = None
) -> LifecycleResult:
    """Start or restart every worker loop, including loops already running.

    A loop can still have a live process after a failed attach makes it unusable.
    """
    if not start_preflight(config, generation):
        return False
    ok: LifecycleResult = True
    for host in config.hosts:
        synced = sync_package(config, host) if sync else True
        if synced is not True:
            ok = _combine_result(ok, synced)
            print(f"  {host.name}: {sync_verdict(synced, 'not started')}")
            continue
        pattern = _loop_regex(host)
        pythonpath = shlex.quote(str(_pkg_src_dir())) if _is_local(host) else f'"$HOME/{_MANAGED_SRC_REL}"'
        script = build_start_script(
            config,
            host,
            pattern=pattern,
            pythonpath=pythonpath,
            generation=generation,
            unit_name=_UNIT_NAME,
            log_path=_LOG_PATH,
            pybin_shell=_pybin_shell(config.python_bin),
            loop_args=_loop_args(host),
        )
        try:
            result = run_on_host(config, host, locked_script(config.python_bin, script), timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            ok = _combine_result(ok, None)
            print(f"  {host.name}: UNKNOWN (lifecycle transport unavailable)")
            continue
        if result.returncode == 255:
            ok = _combine_result(ok, None)
            print(f"  {host.name}: UNKNOWN (lifecycle transport exited 255)")
            continue
        terminal = _terminal_marker_result(
            result.stdout,
            ("STARTED_SYSTEMD", "STARTED_NOHUP"),
            (
                "FAILED_SOURCE_LAYOUT",
                "FAILED_SYSTEMD",
                "FAILED_NOHUP",
                "FAILED_GENERATION_INVALIDATION",
                "FAILED_GENERATION_RECORD",
            ),
        )
        if terminal is None:
            ok = _combine_result(ok, None)
            print(f"  {host.name}: UNKNOWN (no terminal lifecycle marker)")
            continue
        service = "STARTED_SYSTEMD" in result.stdout.splitlines()
        started = terminal
        mode = "systemd" if service else "nohup"
        sweep_ok = actor_sweep.report_result(host, result)
        host_ok = started and sweep_ok
        ok = _combine_result(ok, host_ok)
        verdict = "up" if host_ok else (
            "FAILED (actor sweep unsettled)" if started else "FAILED"
        )
        print(f"  {host.name}: {verdict} ({host.address}, {mode})")
        if not started:
            print(f"    stdout: {result.stdout.strip()}\n    stderr: {result.stderr.strip()}")
    return ok


def down(config: ClusterConfig) -> LifecycleResult:
    return _down(config, required_generation=None)


def down_if_generation(config: ClusterConfig, generation: str) -> LifecycleResult:
    """Stop only Workers whose recorded generation still matches this update."""
    return _down(config, required_generation=generation)


def _down(
    config: ClusterConfig, *, required_generation: str | None
) -> LifecycleResult:
    _validate_stop_arguments(config.hosts, required_generation)
    ok: LifecycleResult = True
    for host in config.hosts:
        pattern = _loop_regex(host)
        try:
            script = build_stop_script(
                config,
                host,
                pattern=pattern,
                required_generation=required_generation,
                unit_name=_UNIT_NAME,
            )
            result = run_on_host(
                config,
                host,
                locked_script(config.python_bin, script, generation_fence=True),
                timeout=75,
            )
        except (OSError, subprocess.TimeoutExpired):
            ok = _combine_result(ok, None)
            print(f"  {host.name}: UNKNOWN (lifecycle transport unavailable)")
            continue
        if result.returncode == 255:
            ok = _combine_result(ok, None)
            print(f"  {host.name}: UNKNOWN (lifecycle transport exited 255)")
            continue
        if "UNKNOWN_GENERATION_FENCE" in result.stdout.splitlines():
            ok = _combine_result(ok, None)
            print(f"  {host.name}: UNKNOWN (worker generation fence unavailable)")
            continue
        if "REFUSED_GENERATION" in result.stdout.splitlines():
            ok = _combine_result(ok, None)
            print(f"  {host.name}: UNKNOWN (worker generation changed)")
            continue
        done = _terminal_marker_result(
            result.stdout,
            ("DONE_SYSTEMD", "DONE_NOHUP"),
            (
                "FAILED_SYSTEMD_STOP",
                "FAILED_SYSTEMD_ACTIVE",
                "FAILED_STRAY_ACTIVE",
                "FAILED_NOHUP_ACTIVE",
                "FAILED_GENERATION_INVALIDATION",
            )
        )
        if done is None:
            ok = _combine_result(ok, None)
            print(f"  {host.name}: UNKNOWN (no terminal lifecycle marker)")
            continue
        sweep_ok = actor_sweep.report_result(host, result) if done else False
        host_ok = done and sweep_ok
        ok = _combine_result(ok, host_ok)
        verdict = "down" if host_ok else (
            "FAILED (actor sweep unsettled)" if done else "FAILED (host reported a stop failure)"
        )
        print(f"  {host.name}: {verdict}")
    return ok


def status(config: ClusterConfig) -> list[dict[str, str]]:
    rows = []
    for host in config.hosts:
        health = passive_worker_health(config, host, runner=run_on_host)
        loop = {True: "running", False: "stopped"}.get(health["running"], "unknown")
        port = {True: "open", False: "closed"}.get(health["listening"], "unknown")
        rows.append({"host": host.name, "loop": loop, "port": port, "address": host.address, "mode": health["mode"]})
        print(
            f"  {host.name}: worker_service={loop if loop == 'running' else loop.upper()} "
            f"listener={port if port == 'open' else port.upper()} "
            f"({host.address}, {health['mode']})"
        )
    return rows


def restart(config: ClusterConfig, sync: bool = True) -> LifecycleResult:
    # A start refusal that needs no host comes before the stop, so a refused
    # restart leaves the workers as they were instead of stopped.
    if not start_preflight(config):
        return False
    stopped = down(config)
    if stopped is not True:
        return stopped
    time.sleep(1)
    return up(config, sync=sync)


def ensure_torchmonarch_pin(config: ClusterConfig, pin: str) -> bool:
    """Install the exact runtime pin in every configured worker interpreter.

    ``dgxm update`` installs the node package with ``--no-deps``, so torchmonarch,
    whose exact version is part of the worker wire and API contract, is
    installed here. ``--no-deps`` keeps this install off ComfyUI's torch build.
    """
    ok = True
    expected = shlex.quote(pin)
    requirement = shlex.quote(f"torchmonarch=={pin}")
    for host in config.hosts:
        script = f"""
set -eu
{_pybin_shell(config.python_bin)}
INSTALLED=$("$PYBIN" -c 'import importlib.metadata as m; print(m.version("torchmonarch"))' 2>/dev/null || true)
if [ "$INSTALLED" != {expected} ]; then
  "$PYBIN" -m pip install --no-deps -q {requirement}
fi
"$PYBIN" -c 'import importlib.metadata as m, sys; sys.exit(0 if m.version("torchmonarch") == sys.argv[1] else 1)' {expected}
echo PIN_OK
"""
        try:
            result = run_on_host(config, host, locked_script(config.python_bin, script), timeout=180)
        except (OSError, subprocess.TimeoutExpired) as exc:
            ok = False
            print(f"  {host.name}: FAILED installing torchmonarch {pin} ({exc})")
            continue
        pinned = result.returncode == 0 and "PIN_OK" in result.stdout
        ok &= pinned
        print(f"  {host.name}: torchmonarch {pin} {'ready' if pinned else 'FAILED'}")
        if not pinned:
            detail = (result.stderr or result.stdout or "no output").strip()
            print(f"    {detail[-500:]}")
    return ok


def _tcp_probe(address: str, timeout: float = 3.0) -> bool | None:
    # True means connected, None means timed out, False means another error.
    # A timeout cannot establish that the port is free.
    try:
        host, port = tcp_endpoint(address)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except TimeoutError:
        return None
    except (OSError, ValueError):
        return False


def install_systemd(config: ClusterConfig) -> bool:
    """Install + enable a user systemd unit for the worker loop on each host."""
    if not _transport_acknowledged(config, "install/start worker services"):
        return False
    _validate_unit_arguments(config)
    ok = True
    for host in config.hosts:
        unit = _systemd_worker_unit(config, host)
        legacy_unit = _systemd_worker_unit(config, host, marked=False)
        # A worker process/listener is not evidence that this unit is ours.
        # Refuse a foreign unit before synchronizing any adjacent source.
        try:
            probe = run_on_host(
                config,
                host,
                locked_script(
                    config.python_bin, "set -eu\n" + _unit_ownership_probe(legacy_unit),
                ),
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            ok = False
            print(f"  {host.name}: FAILED (unit ownership probe: {exc})")
            continue
        if probe.returncode != 0 or probe.stdout.strip() not in {"UNIT_ABSENT", "UNIT_OWNED"}:
            ok = False
            print(f"  {host.name}: {_unit_refusal(probe.stdout)}")
            continue
        # A unit install is a mutation: fail-closed on an unobserved sync.
        synced = sync_package(config, host)
        if synced is not True:
            ok = False
            print(f"  {host.name}: {sync_verdict(synced, 'unit not installed')}")
            continue
        pattern = _loop_regex(host)
        script = f"""
set -eu
{_unit_ownership_probe(legacy_unit)}
{invalidate_generation_command_shell()} || exit $?
{_unit_publish_script(unit)}
# A nohup loop from `dgxm up` would hold the bind address and crash-loop the
# Clear the unit first to avoid EADDRINUSE (idempotent install).
pkill -f -- {shlex.quote(pattern)} 2>/dev/null || true
sleep 0.5
systemctl --user daemon-reload
systemctl --user enable --now {_UNIT_NAME}
sleep 1
echo "STATE=$(systemctl --user is-active {_UNIT_NAME})"
echo "LINGER=$(loginctl show-user "$USER" --property=Linger --value 2>/dev/null || echo unknown)"
"""
        try:
            result = run_on_host(config, host, locked_script(config.python_bin, script), timeout=60)
        except (OSError, subprocess.TimeoutExpired) as exc:
            ok = False
            print(f"  {host.name}: FAILED (lifecycle command: {exc})")
            continue
        state = linger = ""
        for line in result.stdout.splitlines():
            if line.startswith("STATE="):
                state = line.removeprefix("STATE=").strip()
            if line.startswith("LINGER="):
                linger = line.removeprefix("LINGER=").strip()
        active = result.returncode == 0 and state == "active"
        if active and linger != "yes":
            ok = False
            print(
                f"  {host.name}: unit started, but lingering is not confirmed; without it the unit stops when your "
                "last session on that host ends. Enable it there, then retry: sudo loginctl enable-linger $USER"
            )
            continue
        ok &= active
        print(f"  {host.name}: systemd unit {'active' if active else 'FAILED'}")
        if not active:
            print(f"    {result.stdout.strip()} {result.stderr.strip()}")
            print("    (a user unit also needs lingering to survive logout: sudo loginctl enable-linger $USER)")
    return ok


def units_are_removable(config: ClusterConfig) -> bool:
    """Check every host's unit ownership before stopping or removing anything.

    Run this before ``down`` so a foreign-unit refusal leaves Workers and their
    generation records intact. ``uninstall_systemd`` checks ownership again before
    removal.
    """
    _validate_unit_arguments(config)
    ok = True
    for host in config.hosts:
        legacy_unit = _systemd_worker_unit(config, host, marked=False)
        probe_script = "set -eu\n" + _unit_ownership_probe(legacy_unit, allow_setup_token=True)
        try:
            probe = run_on_host(
                config, host, locked_script(config.python_bin, probe_script), timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            ok = False
            print(f"  {host.name}: FAILED (unit ownership probe: {exc})")
            continue
        if probe.returncode != 0 or probe.stdout.strip() not in {"UNIT_ABSENT", "UNIT_OWNED"}:
            ok = False
            print(f"  {host.name}: {_unit_refusal(probe.stdout)}")
    return ok


def uninstall_systemd(config: ClusterConfig) -> bool:
    _validate_unit_arguments(config)
    ok = True
    for host in config.hosts:
        legacy_unit = _systemd_worker_unit(config, host, marked=False)
        try:
            result = run_on_host(
                config,
                host,
                locked_script(
                    config.python_bin, _unit_remove_script(legacy_unit), generation_fence=True),
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            ok = False
            print(f"  {host.name}: FAILED removing unit ({exc})")
            continue
        removed = result.returncode == 0 and "DONE" in result.stdout
        ok &= removed
        if removed:
            print(f"  {host.name}: unit removed")
            leftovers = _unit_removal_leftovers(result.stdout)
            if leftovers:
                print(f"    left behind: {leftovers}")
        elif "REFUSED" in result.stdout:
            print(f"  {host.name}: {_unit_refusal(result.stdout)}")
        else:
            print(f"  {host.name}: FAILED removing unit")
    return ok
