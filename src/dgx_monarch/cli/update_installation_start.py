"""Generation-bound restart of an exactly restored legacy Worker installation."""
from __future__ import annotations

import base64
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def start_restored(request: dict[str, Any], installation: dict[str, Any], generation: dict[str, Any]) -> None:
    """Run in the isolated host script while holding the lifecycle and generation locks."""
    base = Path.home() / ".local/share/dgx-monarch"
    unit = Path.home() / ".config/systemd/user/dgxm-worker.service"
    journal = base / "transactions" / request["token"]
    prior = request["prior"]
    read, stopped = installation["read"], installation["stopped"]
    if read(journal / "compensated") != b"COMPENSATED\n":
        raise ValueError("prior installation was not compensated")
    recorded = json.loads(read(journal / "installation.json"))
    if recorded["prior"] != prior or recorded["token"] != request["token"]:
        raise ValueError("prior journal differs")
    stopped(unit)
    if read(unit) != base64.b64decode(prior["unit"], validate=True) or installation["link_state"](base / "src") != prior["live"]:
        raise ValueError("restored installation changed")
    pattern = request["pattern"]
    if generation["matching"](pattern) != [] or not generation["remove_marker"]():
        raise ValueError("prior Worker is not stopped")
    # Prestart clears the old marker under the generation lock. Release that
    # lock for its subprocess while retaining the lifecycle lock.
    fenced_unit = any(line.startswith("ExecStartPre=") for line in base64.b64decode(prior["unit"], validate=True).decode().splitlines())
    if fenced_unit:
        fcntl.flock(generation["fencefd"], fcntl.LOCK_UN)
        os.close(generation["fencefd"])
        generation["fencefd"] = -1
    started = subprocess.run(["systemctl", "--user", "start", unit.name], capture_output=True, timeout=40)
    if fenced_unit:
        generation["fencefd"] = generation["acquire_generation_lock"](Path.home(), True)
    if started.returncode:
        raise ValueError("prior Worker start did not complete")
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            current = installation["snapshot"](unit, base / "src", request["python_bin"], request["address"])
        except (OSError, ValueError, subprocess.SubprocessError):
            time.sleep(0.2)
            continue
        if {key: value for key, value in current.items() if key not in {"pid", "birth"}} != {key: value for key, value in prior.items() if key not in {"pid", "birth"}}:
            raise ValueError("started installation differs from prior")
        identity = generation["active_identity"](pattern, request["proc_table"], request["endpoint"], current["pid"], current["birth"])
        if identity is None:
            time.sleep(0.2)
            continue
        installation["durable"](journal / "prior-start.json", json.dumps(current, sort_keys=True).encode())
        if not generation["write_marker"](request["generation"], *identity):
            raise ValueError("prior generation could not be recorded")
        if generation["active_identity"](pattern, request["proc_table"], request["endpoint"], *identity) != identity:
            generation["remove_marker"]()
            raise ValueError("prior Worker changed after generation recording")
        print("STARTED_PRIOR")
        return
    raise ValueError("prior Worker readiness was not proved")
