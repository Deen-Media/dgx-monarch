"""Read-only activity and opt-in privileged marker checks for verified updates."""
from __future__ import annotations

import json
import math
import shlex
import subprocess
import sys
import time
from typing import Any

from ..config import ClusterConfig
from . import lifecycle
from .setup_process_inspection import privileged_process_source
from .update_types import ActivitySnapshot


def activity_snapshot(driver_host: str | None, actors: int | None,
                      comfy_ports: Any, candidate_count: Any) -> ActivitySnapshot:
    ports = comfy_ports.find_comfy_ports()
    running, observed = comfy_ports.driver_probe(driver_host, ports)
    candidates = candidate_count()
    candidate_running = type(candidates) is int and candidates > 0
    ambiguous_driver = running is None and (
        not observed or driver_host is not None or candidates is None)
    return ActivitySnapshot(
        comfy_running=running is not None or candidate_running,
        render_active=None if ambiguous_driver or running is not None or candidate_running else False,
        active_leases=None if ambiguous_driver or running is not None or candidate_running else 0,
        marked_actors=actors,
    )


def _unavailable() -> None:
    print("verified update: privileged process inspection is unavailable or incomplete; "
          "keep the reviewed read-only inspectors running on every configured host "
          "and retry only after their reports are complete (docs/INSTALL.md)", file=sys.stderr)
    return None


def privileged_actor_count(config: ClusterConfig) -> int | None:
    """Require a new authenticated report on every host; never start a helper.

    The existing client authenticates socket ownership, root peer credentials,
    UID, boot, exact helper source and request-bounded freshness. A complete
    report may count actors; any unavailable or unknown report blocks update.
    """
    if not config.hosts:
        _unavailable()
        return None
    source = privileged_process_source(True) + '''
try:
    report = _request_process_report(DGXM_INSPECTOR_SHA)
    if report["status"] != "ok" or report["unknown_count"] != 0:
        raise ValueError("incomplete process report")
    print(json.dumps({"actors": len(report["actors"])}))
except (OSError, ValueError, TypeError, KeyError, IndexError, UnicodeError, struct.error):
    raise SystemExit(2)
'''
    script = (
        f'PYBIN={shlex.quote(config.python_bin)}\n'
        'case "$PYBIN" in "~/"*) PYBIN="$HOME/${PYBIN#\\~/}";; "~") PYBIN="$HOME";; esac\n'
        f'"$PYBIN" -I -S -B -c {shlex.quote(source)}\n')
    deadline = time.monotonic() + 45.0
    count = 0
    for host in config.hosts:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _unavailable()
            return None
        try:
            result = lifecycle.run_on_host(
                config, host, script, timeout=max(1, math.ceil(min(15.0, remaining))))
            if result.returncode != 0 or len(result.stdout) > 256:
                _unavailable()
                return None
            payload = json.loads(result.stdout)
            if not isinstance(payload, dict) or set(payload) != {"actors"}:
                _unavailable()
                return None
            value = payload["actors"]
            if type(value) is not int or not 0 <= value <= 8192:
                _unavailable()
                return None
        except (OSError, ValueError, subprocess.TimeoutExpired):
            _unavailable()
            return None
        count += value
    return count
