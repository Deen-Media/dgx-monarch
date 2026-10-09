"""Warn when Worker hosts drift from the checkout's torchmonarch pin.

Mixed pins can break the worker wire/API contract. Recovery commands probe and
warn without blocking startup; only ``dgxm update`` installs the required pin.
"""
from __future__ import annotations

import subprocess
import sys

from ..config import ClusterConfig
from .lifecycle import _pybin_shell, run_on_host


def warn_on_pin_mismatch(config: ClusterConfig, pin: str) -> None:
    """Probe each host's installed torchmonarch version; warn to stderr on drift.

    Probe failures stay silent because recovery commands report reachability
    separately. This check never raises or blocks startup.
    """
    for host in config.hosts:
        script = f"""
{_pybin_shell(config.python_bin)}
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
        print(
            f"WARNING: {host.name} has torchmonarch {installed} installed, but this "
            f"checkout pins {pin}. Different torchmonarch versions on the driver and workers can break "
            f"the worker wire protocol. Fix: run `dgxm update`, which pulls the repo and installs the pin on "
            f"every host, or on {host.name}, with the python_bin from cluster.toml, run: "
            f"<python_bin> -m pip install --no-deps torchmonarch=={pin}",
            file=sys.stderr,
        )
