"""Activity counts, dependency checks and the staging root for the verified update."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from ..config import ClusterConfig
from . import comfy_ports, lifecycle
from .update_release import CommandRunner

_DEPENDENCY_CHECK = r"""
import importlib.metadata as md, json, sys
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.version import Version
for raw in json.loads(sys.argv[1]):
    req = Requirement(raw)
    if req.name.lower() == "torchmonarch": continue
    try: installed = Version(md.version(req.name))
    except Exception: raise SystemExit(2)
    if req.specifier and installed not in req.specifier: raise SystemExit(3)
"""


def runtime_env(site: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(site)
    return env


def comfy_candidate_count() -> int | None:
    """Compatibility wrapper: count ComfyUI-shaped processes via comfy_ports; None when the scan is incomplete."""
    return comfy_ports.comfy_candidate_count()


def marked_actor_count(config: ClusterConfig) -> int | None:
    """Count every marked actor; any incomplete host report is ambiguity."""
    count = 0
    addresses = " ".join(
        f"--loop-address {shlex.quote(host.address)}" for host in config.hosts
    )
    for host in config.hosts:
        python = shlex.quote(config.python_bin)
        script = f"""
set -eu
PYBIN={python}
case "$PYBIN" in "~/"*) PYBIN="$HOME/${{PYBIN#\\~/}}";; "~") PYBIN="$HOME";; esac
export PYTHONPATH="$HOME/.local/share/dgx-monarch/src"
"$PYBIN" -m dgx_monarch.cli.actor_reaper --report {addresses}
"""
        try:
            result = lifecycle.run_on_host(config, host, script, timeout=60)
            payload = json.loads(result.stdout.splitlines()[-1])
        except (IndexError, json.JSONDecodeError, OSError, subprocess.TimeoutExpired):
            return None
        candidates = payload.get("candidates") if isinstance(payload, dict) else None
        if result.returncode != 0 or not isinstance(candidates, list):
            return None
        count += len(candidates)
    return count


def dependencies_available(
    config: ClusterConfig, dependencies: Sequence[str], run: CommandRunner
) -> bool:
    """Verify ambient target dependencies without mutating any interpreter."""
    payload = json.dumps(dependencies, separators=(",", ":"))
    local = run(
        [sys.executable, "-c", _DEPENDENCY_CHECK, payload],
        capture_output=True, text=True, timeout=60,
    )
    if local.returncode != 0:
        return False
    script = f"""
set -eu
PYBIN={shlex.quote(config.python_bin)}
case "$PYBIN" in "~/"*) PYBIN="$HOME/${{PYBIN#\\~/}}";; "~") PYBIN="$HOME";; esac
"$PYBIN" -c {shlex.quote(_DEPENDENCY_CHECK)} {shlex.quote(payload)}
"""
    for host in config.hosts:
        try:
            result = lifecycle.run_on_host(config, host, script, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return False
        if result.returncode != 0:
            return False
    return True


def private_root(path: Path) -> None:
    if not path.is_absolute():
        raise RuntimeError("update staging root must be absolute")
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not path.is_dir()
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
    ):
        raise RuntimeError("update staging root is not a real directory owned by the current user")
    path.chmod(0o700)
