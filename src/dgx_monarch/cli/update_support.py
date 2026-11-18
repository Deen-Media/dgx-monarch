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
from .update_dependency_contract import probe_source
from .update_dependency_diagnostics import report_result
from .update_release import CommandRunner

# Prints the unmet requirements as a JSON list in report_result's bounded
# forms, and exits 1 when there are any.
_DEPENDENCY_CHECK = r"""
import importlib.metadata as md, json, sys
errors = []
try:
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name
    from pip._vendor.packaging.version import Version
    for raw in json.loads(sys.argv[1]):
        req = Requirement(raw)
        if req.marker and not req.marker.evaluate({"extra": ""}): continue
        name = canonicalize_name(req.name)
        if name == "torchmonarch": continue
        try: installed = Version(md.version(req.name))
        except Exception:
            errors.append(name + ": missing")
            continue
        if req.specifier and installed not in req.specifier:
            errors.append(f"{name}: installed {installed}, requires {req.specifier}")
except Exception:
    errors = ["dependency metadata could not be verified"]
print(json.dumps(errors))
raise SystemExit(1 if errors else 0)
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
    """Verify ambient target dependencies without mutating any interpreter.

    Checks every interpreter and prints one line per unmet requirement to
    stderr, labelled ``driver`` or ``host_<n>`` in cluster.toml order.
    """
    return _every_interpreter(config, _DEPENDENCY_CHECK,
                              json.dumps(dependencies, separators=(",", ":")), run)


def candidate_dependencies_available(
    config: ClusterConfig, dependencies: Sequence[str], contract: dict, run: CommandRunner,
) -> bool:
    """Inspect the exact candidate's dependency closure on all target interpreters.

    Like ``dependencies_available``, it reads every interpreter before it
    answers, so its lines name each one that fails.
    """
    payload = json.dumps({"contract": contract, "roots": list(dependencies)}, separators=(",", ":"))
    return _every_interpreter(config, probe_source(), payload, run)


def _every_interpreter(config: ClusterConfig, source: str, payload: str, run: CommandRunner) -> bool:
    """Run one check on the driver's Python and each host's; report each, then answer."""
    try:
        satisfied = report_result(run(
            [sys.executable, "-c", source, payload],
            capture_output=True, text=True, timeout=60,
        ), "driver")
    except (OSError, subprocess.TimeoutExpired):
        print("  driver: dependency inspection failed", file=sys.stderr)
        satisfied = False
    script = f"""
set -eu
PYBIN={shlex.quote(config.python_bin)}
case "$PYBIN" in "~/"*) PYBIN="$HOME/${{PYBIN#\\~/}}";; "~") PYBIN="$HOME";; esac
"$PYBIN" -c {shlex.quote(source)} {shlex.quote(payload)}
"""
    for ordinal, host in enumerate(config.hosts, 1):
        try:
            satisfied &= report_result(lifecycle.run_on_host(config, host, script, timeout=60),
                                       f"host_{ordinal}")
        except (OSError, subprocess.TimeoutExpired):
            print(f"  host_{ordinal}: dependency inspection failed", file=sys.stderr)
            satisfied = False
    return satisfied


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
