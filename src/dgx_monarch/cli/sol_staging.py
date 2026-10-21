"""Remove private Sol staging that a Studio review created and no longer needs.

`sol_install_payload.plan_host` stages each plan under
~/.local/state/dgx-monarch/sol-install/<token> on every host: a pip resolution
and the wheels it built. A Studio review mints a token when the owner gives
none, and nothing else removes that staging when the review is abandoned.
This module removes the named token directories on one host under the shared
lifecycle lock. A directory that holds an apply readback is kept as evidence,
and the removal never follows a symbolic link.
"""
from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Collection

from ..config import ClusterConfig, HostConfig
from . import lifecycle
from .lifecycle_lock import lifecycle_lock_source

MARKER = "DGXM_SOL_STAGING="
DISCARD_LIMIT = 16
OUTCOMES = frozenset({"removed", "absent", "kept", "unsafe"})
_TOKEN = re.compile(r"[a-f0-9]{32}")

# Runs on the host with `python -I -B`, so it may import only the standard library.
_HOST_SOURCE = r'''
import json, os, shutil, stat
from pathlib import Path


def discard(tokens):
    root = Path.home()
    for part in (".local", "state", "dgx-monarch", "sol-install"):
        root = root / part
        try:
            info = root.lstat()
        except FileNotFoundError:
            return {token: "absent" for token in tokens}
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o022:
            return {token: "unsafe" for token in tokens}
    if not shutil.rmtree.avoids_symlink_attacks:
        return {token: "unsafe" for token in tokens}
    outcome = {}
    for token in tokens:
        path = root / token
        try:
            info = path.lstat()
        except FileNotFoundError:
            outcome[token] = "absent"
            continue
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            outcome[token] = "unsafe"
        elif os.path.lexists(path / "readback.json"):
            outcome[token] = "kept"
        else:
            shutil.rmtree(path)
            outcome[token] = "removed"
    return outcome
'''


def discard_script(config: ClusterConfig, tokens: Collection[str]) -> str:
    names = sorted(set(tokens))
    if not names or len(names) > DISCARD_LIMIT or any(not _TOKEN.fullmatch(token) for token in names):
        raise ValueError("Sol staging tokens must be 1 to 16 plan tokens of 32 lowercase hex characters")
    source = lifecycle_lock_source() + _HOST_SOURCE + f'''
request = {names!r}
try:
    lock_fd = acquire_lifecycle_lock(Path.home(), True)
    print({MARKER!r} + json.dumps({{"status": "ok", "outcome": discard(request)}}))
except Exception:
    print({MARKER!r} + json.dumps({{"status": "failed"}}))
    raise SystemExit(1)
'''
    return (f"set -eu\n{lifecycle._pybin_shell(config.python_bin)}\n"
            f"\"$PYBIN\" -I -B - <<'DGXM_SOL_STAGING'\n{source}\nDGXM_SOL_STAGING\n")


def discard(config: ClusterConfig, host: HostConfig, tokens: Collection[str]) -> dict[str, str] | None:
    """Each token's outcome on this host, or None when the outcome is unknown."""
    names = sorted(set(tokens))
    try:
        result = lifecycle.run_on_host(config, host, discard_script(config, names), timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    rows = [line[len(MARKER):] for line in result.stdout.splitlines() if line.startswith(MARKER)]
    if len(rows) != 1 or result.returncode != 0:
        return None
    try:
        value = json.loads(rows[0])
    except ValueError:
        return None
    outcome = value.get("outcome") if isinstance(value, dict) and value.get("status") == "ok" else None
    if not isinstance(outcome, dict) or sorted(outcome) != names or any(state not in OUTCOMES for state in outcome.values()):
        return None
    return {str(token): str(state) for token, state in outcome.items()}
