"""Host-locality checks, operation outcomes and listener ownership checks."""
from __future__ import annotations

import ipaddress
import os
import pwd
import shlex
import socket
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path

from ..config import ClusterConfig, HostConfig
from ..config_schema import validate_ssh_host, validate_ssh_user
from .lifecycle_generation import validate_generation
from .worker_health import _proc_listener_target

LifecycleResult = bool | None


def combine_result(
    current: LifecycleResult, value: LifecycleResult
) -> LifecycleResult:
    """Fold host outcomes while preserving any transport uncertainty."""
    if current is None or value is None:
        return None
    return current and value


def sync_outcome(
    error: BaseException, host: str, stage: str, log: object
) -> LifecycleResult:
    """Return False on definite sync failure and None on an unknown outcome.

    The 120-second timeout kills only local rsync. The remote ``rsync --server``
    may still be applying ``--delete --delete-excluded`` to the managed tree, so a
    timeout cannot establish failure. Omit exception details from operator logs.
    """
    if isinstance(error, subprocess.TimeoutExpired):
        log.warning("%s on %s was not observed to finish", stage, host)  # type: ignore[attr-defined]
        return None
    log.warning("%s on %s failed: %s", stage, host, error)  # type: ignore[attr-defined]
    return False


def sync_verdict(result: LifecycleResult, consequence: str) -> str:
    """The operator line for a package sync that did not plainly succeed."""
    if result is None:
        return f"UNKNOWN (package sync unobserved; {consequence})"
    return f"FAILED (package sync; {consequence}; see the warning above)"


def transport_acknowledged(config: ClusterConfig, operation: str) -> bool:
    """Refuse a networked start on an unacknowledged fabric, from the config alone."""
    if config.hosts and config.transport_security != "trusted_fabric":
        print(
            f"refusing to {operation}: peer authentication is unavailable at the "
            "torchmonarch attach API dgx-monarch calls. Source-restrict the dedicated "
            'fabric, then set cluster.transport_security = "trusted_fabric" '
            "(SECURITY.md)."
        )
        return False
    return True


def validated_ssh_identity(host: HostConfig) -> tuple[str, str]:
    """Revalidate programmatic host objects at each destination sink."""
    source = Path("programmatic-cluster-config")
    name = validate_ssh_host(host.name, "hosts.name", source)
    user = validate_ssh_user(host.ssh_user, "hosts.ssh_user", source) if host.ssh_user else ""
    return name, user


def validate_start_arguments(hosts: Iterable[HostConfig], generation: str | None) -> None:
    """Validate all Worker start arguments before any host operation.

    Check the generation, listener address, SSH name and SSH user for every host
    before resolving, syncing or restarting any host. Renderers and SSH sinks
    repeat these checks at use. Invalid SSH identities cannot be treated as local,
    even for programmatically built configs.
    """
    if generation is not None:
        validate_generation(generation)
    for host in hosts:
        _proc_listener_target(host.address)
        validated_ssh_identity(host)


def validate_stop_arguments(hosts: Iterable[HostConfig], generation: str | None) -> None:
    """Validate Worker stop arguments before any host operation.

    Check the generation token even with an empty host list, when no per-host
    stop script would validate it.
    """
    if generation is not None:
        validate_generation(generation)
    for host in hosts:
        validated_ssh_identity(host)


def start_preflight(config: ClusterConfig, generation: str | None = None) -> bool:
    """Run host-independent start checks; return False after printing a refusal.

    Transport acknowledgement runs first; argument checks then raise on failure.
    Run both before restart's stop phase and before any torchmonarch pin probe.
    """
    if not transport_acknowledged(config, "start the worker service"):
        return False
    validate_start_arguments(config.hosts, generation)
    return True


def remote_colocation_error(
    hosts: Iterable[HostConfig], probe: Callable[[HostConfig], LifecycleResult]
) -> str | None:
    """Return the fail-closed verified-update layout refusal, if any."""
    hosts = tuple(hosts)
    localities = tuple(probe(host) for host in hosts)
    if not localities:
        return "verified cluster update needs at least one worker host"
    if any(value is None for value in localities):
        return "verified release slots require definite worker/driver colocation evidence"
    if sum(value is True for value in localities) > 1:
        return "verified update refuses duplicate aliases for the local Worker"
    identities: set[str] = set()
    current_user = pwd.getpwuid(os.geteuid()).pw_name
    for host, local in zip(hosts, localities, strict=True):
        name, user = validated_ssh_identity(host)
        if local and user and user != current_user:
            return "verified update refuses a local Worker with a different SSH user"
        aliases = {name.lower(), host.address.rsplit(":", 1)[0]}
        try:
            aliases.update(str(item[4][0]) for item in socket.getaddrinfo(
                name, None, socket.AF_UNSPEC, socket.SOCK_STREAM))
        except (OSError, ValueError):
            pass  # The locality probe already established this host's location.
        if identities.intersection(aliases):
            return "verified update refuses duplicate Worker host aliases"
        identities.update(aliases)
    return None


def terminal_marker_result(
    stdout: str, successes: tuple[str, ...], failures: tuple[str, ...]
) -> LifecycleResult:
    """Classify only an explicit terminal lifecycle marker."""
    lines = {line.strip() for line in stdout.splitlines()}
    if any(marker in lines for marker in successes):
        return True
    if any(marker in lines for marker in failures):
        return False
    return None


def is_local(host: HostConfig) -> bool | None:
    """Return local/remote, or ``None`` when colocation cannot be resolved."""
    names = {socket.gethostname(), socket.getfqdn(), "localhost", "127.0.0.1"}
    if host.name in names:
        return True
    try:
        literal = ipaddress.ip_address(host.name)
    except ValueError:
        literal = None
    if literal is not None:
        return True if literal.is_loopback else _ip_is_local(literal.compressed)
    try:
        addresses = {
            str(item[4][0])
            for item in socket.getaddrinfo(
                host.name, None, socket.AF_UNSPEC, socket.SOCK_STREAM
            )
        }
    except (OSError, ValueError):
        return None
    unknown = False
    for address in addresses:
        try:
            if ipaddress.ip_address(address).is_loopback:
                return True
        except ValueError:
            unknown = True
            continue
        local = _ip_is_local(address)
        if local is True:
            return True
        unknown |= local is None
    return None if unknown else False


def _ip_is_local(ip: str) -> bool | None:
    try:
        result = subprocess.run(
            ["ip", "-o", "addr"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return f" {ip}/" in result.stdout or f" {ip} " in result.stdout


def owned_listener_probe_shell(
    host: HostConfig, pattern: str, *, proc_root: str = "/proc"
) -> str:
    """Prove an exact loop PID owns the configured LISTEN socket inode."""
    proc_path, endpoint = _proc_listener_target(host.address)
    proc_path = f"{proc_root.rstrip('/')}{proc_path.removeprefix('/proc')}"
    return f"""
DGXM_PROC_ROOT={shlex.quote(proc_root)}
dgxm_owned_listener_ready() {{
  DGXM_OWNED_LISTENER_PID=
  DGXM_OWNED_LISTENER_START=
  [ -r {shlex.quote(proc_path)} ] || return 1
  DGXM_LISTENER_INODES=$(awk -v endpoint={shlex.quote(endpoint)} \
    '$2 == endpoint && $4 == "0A" {{ print $10 }}' \
    {shlex.quote(proc_path)})
  [ -n "$DGXM_LISTENER_INODES" ] || return 1
  for DGXM_LOOP_PID in $(pgrep -f -- {shlex.quote(pattern)}); do
    [ -d "$DGXM_PROC_ROOT/$DGXM_LOOP_PID/fd" ] || continue
    DGXM_LOOP_STAT=$(cat "$DGXM_PROC_ROOT/$DGXM_LOOP_PID/stat" 2>/dev/null || true)
    DGXM_LOOP_TAIL=${{DGXM_LOOP_STAT##*) }}
    set -- $DGXM_LOOP_TAIL
    [ "$#" -ge 20 ] || continue
    DGXM_LOOP_START=${{20}}
    for DGXM_LOOP_FD in "$DGXM_PROC_ROOT/$DGXM_LOOP_PID/fd/"*; do
      DGXM_SOCKET=$(readlink "$DGXM_LOOP_FD" 2>/dev/null || true)
      case "$DGXM_SOCKET" in
        socket:\\[*\\])
          DGXM_SOCKET_INODE=${{DGXM_SOCKET#socket:\\[}}
          DGXM_SOCKET_INODE=${{DGXM_SOCKET_INODE%\\]}}
          for DGXM_LISTENER_INODE in $DGXM_LISTENER_INODES; do
            if [ "$DGXM_SOCKET_INODE" = "$DGXM_LISTENER_INODE" ]; then
              DGXM_LOOP_STAT=$(cat "$DGXM_PROC_ROOT/$DGXM_LOOP_PID/stat" 2>/dev/null || true)
              DGXM_LOOP_TAIL=${{DGXM_LOOP_STAT##*) }}
              set -- $DGXM_LOOP_TAIL
              [ "$#" -ge 20 ] && [ "${{20}}" = "$DGXM_LOOP_START" ] || continue
              DGXM_OWNED_LISTENER_PID=$DGXM_LOOP_PID
              DGXM_OWNED_LISTENER_START=$DGXM_LOOP_START
              return 0
            fi
          done
          ;;
      esac
    done
  done
  return 1
}}
"""
