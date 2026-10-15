"""Bounded Doctor categories for updater stderr; never disclose probe details."""
from __future__ import annotations

import json
import sys
from collections import Counter

_CHECKS = {
    "config permissions": "config_permissions",
    "hostname resolution": "hostname_resolution",
    "torchmonarch": "monarch_pin",
    "torch CUDA": "torch_cuda",
    "torch version match": "torch_version_match",
    "NCCL_PROTO": "nccl_protocol",
    "NCCL library": "nccl_library",
    "fabric NCCL_PROTO": "fabric_nccl_protocol",
    "NCCL_P2P_DISABLE": "nccl_peer_access",
    "client_bind": "client_bind",
    "rsync": "rsync",
    "mesh health": "mesh_health",
    "RDMA rails": "rdma_rails",
    "rdma rails": "rdma_rails",
    "fabric ifaces": "fabric_interfaces",
    "safetensors pread": "safetensors_pread",
    "python dev headers": "python_headers",
    "orphaned actor procs": "orphaned_actors",
}
_HOST_CHECKS = {
    "env": "worker_environment",
    "worker service": "worker_service",
    "fabric": "worker_fabric",
    "NCCL_PROTO": "worker_nccl_protocol",
    "listener generation": "worker_listener_generation",
    "swap": "worker_swap",
    "worker policy": "worker_policy",
    "worker release version": "worker_release_version",
}
_UNAVAILABLE = "doctor diagnostics: unavailable"


def _lines(stdout: object, hosts: tuple[str, ...]) -> list[str]:
    if not isinstance(stdout, str) or len(stdout) > 128 * 1024:
        return [_UNAVAILABLE]
    try:
        payload = json.loads(stdout)
    except (ValueError, RecursionError):
        return [_UNAVAILABLE]
    checks = payload.get("checks") if isinstance(payload, dict) else None
    if not isinstance(checks, list) or len(checks) > 256:
        return [_UNAVAILABLE]
    categories = dict(_CHECKS)
    categories.update((f"{host} {suffix}", category)
                      for host in hosts for suffix, category in _HOST_CHECKS.items())
    counts: Counter[tuple[str, str]] = Counter()
    for row in checks:
        if not isinstance(row, dict) or not isinstance(row.get("check"), str):
            return [_UNAVAILABLE]
        state = row.get("state")
        if state == "ok":
            continue
        if state not in ("fail", "warn", "unknown"):
            return [_UNAVAILABLE]
        counts[categories.get(row["check"], "other"), state] += 1
    if not counts:
        return [_UNAVAILABLE]
    ordered = sorted(counts.items(), key=lambda item: (item[0][1] != "fail", item[0]))
    result = [f"doctor diagnostics: {category} {state} ({count})"
              for (category, state), count in ordered[:20]]
    if len(ordered) > 20:
        result.append("doctor diagnostics: additional categories omitted")
    return result


def report_doctor_result(returncode: int, stdout: object, hosts: tuple[str, ...]) -> None:
    """Report fixed categories on nonzero exit without changing operation certainty."""
    if returncode == 0:
        return
    for line in _lines(stdout, hosts):
        try:
            print(line, file=sys.stderr)
        except (OSError, ValueError):
            # Lost diagnostic output cannot change failure into unknown, or vice versa.
            return
