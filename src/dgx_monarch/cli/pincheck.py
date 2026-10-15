"""Warn when Worker hosts drift from the checkout's torchmonarch pin.

Mixed pins can break the worker wire/API contract. Recovery commands probe and
warn without blocking startup. Plain ``dgxm update`` installs the required pin
on a plain host; only ``dgxm update --verify`` changes a managed release.
"""
from __future__ import annotations

import subprocess
import sys

from ..config import ClusterConfig
from .doctor_worker_context import POLICY_LINE, worker_context
from .lifecycle import _pybin_shell, run_on_host
from .release_slot import TROUBLESHOOTING
from .release_slot import parse as parse_release


def warn_on_pin_mismatch(config: ClusterConfig, pin: str) -> None:
    """Probe each host's Worker torchmonarch version; warn to stderr on drift.

    A managed host is read through its Worker release, as Doctor reads it, so a
    stale base interpreter behind a current release does not warn. Probe
    failures, an unverified release layout included, stay silent because
    recovery commands report reachability and layout separately. This check
    never raises or blocks startup.
    """
    for host in config.hosts:
        script = f"""
{_pybin_shell(config.python_bin)}
{worker_context(config, host)}
"$PYBIN" -c 'import importlib.metadata as m; print(m.version("torchmonarch"))' 2>/dev/null || echo MISSING
"""
        try:
            result = run_on_host(config, host, script, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode != 0:
            continue
        lines = result.stdout.strip().splitlines()
        installed = lines[-1].strip() if lines else ""
        if not installed or installed == "MISSING" or installed == pin:
            continue
        # A Worker in a release slot imports its release, and plain `dgxm update`
        # never changes a slot, so only `dgxm update --verify` fixes it. A managed
        # link outside releases/ (guided setup's colocated Worker) follows the
        # pull and reads torchmonarch from the base interpreter, as a plain host.
        managed = any(line.startswith(POLICY_LINE) and not line.endswith("PLAIN") for line in lines)
        release = parse_release(lines)
        in_slot = managed and (release is None or release[0] == "SLOT")
        found = (f"the managed Worker release on {host.name} carries torchmonarch {installed}"
                 if in_slot else f"{host.name} has torchmonarch {installed} installed")
        fix = (
            "Fix: run `dgxm update --verify`, which installs a release with the pin on every host; "
            "a plain `dgxm update` never changes a Worker release slot. If the driver already "
            f"moved without its Workers, repair that first ({TROUBLESHOOTING})"
        ) if in_slot else (
            "Fix: run `dgxm update`, which pulls the repo and installs the pin on every host, "
            f"or on {host.name}, with the python path from cluster.toml's [cluster] table, run: "
            f"<python> -m pip install --no-deps torchmonarch=={pin}")
        print(
            f"WARNING: {found}, but this "
            f"checkout pins {pin}. Different torchmonarch versions on the driver and workers can break "
            f"the worker wire protocol. {fix}",
            file=sys.stderr,
        )
