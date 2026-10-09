"""Recognize and journal supported Worker installation layouts."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from ..config import ClusterConfig, HostConfig
from . import lifecycle, update_installation_remote, update_installation_start
from .lifecycle_generation import _GENERATION_HELPER, systemd_generation_invalidator, validate_generation
from .lifecycle_lock import locked_script
from .lifecycle_systemd import hardened_worker_unit, systemd_worker_unit
from .systemd_unit import quote
from .update_release_types import HostRunner
from .update_types import Certainty
from .worker_health import _proc_listener_target


def installation_script(request: dict[str, Any]) -> str:
    source = Path(update_installation_remote.__file__).read_text(encoding="utf-8")
    encoded = base64.b64encode(json.dumps(request).encode()).decode()
    return "# DGXM_INSTALLATION_OPERATION=" + str(request["operation"]) + "\n/usr/bin/python3 -I -S -B - <<'DGXM_INSTALLATION'\n" + source + "\nmain(json.loads(base64.b64decode(" + repr(encoded) + ")))\nDGXM_INSTALLATION\n"


def recognized_unit(config: ClusterConfig, host: HostConfig, snapshot: dict[str, Any]) -> bool:
    """Match complete supported unit bytes, never an ownership comment alone."""
    try:
        unit = base64.b64decode(snapshot["unit"], validate=True).decode()
        site, home = snapshot["site"], snapshot["home"]
        if not all(isinstance(value, str) and value.startswith("/") and not any(ord(char) < 32 for char in value) for value in (site, home)):
            return False
        if hashlib.sha256(unit.encode()).hexdigest() != snapshot["sha256"]:
            return False
        local = lifecycle._is_local(host)
        if local is None:
            return False
        managed = home + "/.local/share/dgx-monarch/src"
        if not local and site != managed and not re.fullmatch(re.escape(home) + r"/\.local/share/dgx-monarch/releases/s-[0-9a-f]{16}-[1-9][0-9]*/site", site):
            return False
        if snapshot.get("site_packages"):
            legacy = hardened_worker_unit(config, host, source=site, site_packages=snapshot["site_packages"], home=home, user=snapshot["user"])
            candidates = [legacy]
            if not local and site == managed:
                candidates.append(hardened_worker_unit(config, host, source="%h/.local/share/dgx-monarch/src", site_packages=snapshot["site_packages"], home=home, user=snapshot["user"]))
            if any(unit == candidate + "\n" * extra for candidate in candidates for extra in range(3)):
                return True
        canonical = systemd_worker_unit(config, host, managed_source=True)
        environment = 'Environment="PYTHONPATH=%h/.local/share/dgx-monarch/src"'
        rendered_site = site if local else site.replace(home + "/", "%h/", 1)
        normal = canonical.replace(environment, "Environment=" + quote("PYTHONPATH=" + rendered_site, preserve_home_specifier=not local))
        if unit in (normal, normal.removeprefix("# dgxm-managed-unit-v1\n")):
            return True
        match = re.match(r"# dgxm-update-token=(u-[0-9a-f]{12}-[0-9a-f]{16})\n", unit)
        if match and site == managed:
            target = snapshot["live"]
            expected = home + "/.local/share/dgx-monarch/releases/" + match[1] + "/site"
            actual = str((Path(managed).parent / target["target"]).resolve()) if target["kind"] == "symlink" else ""
            return actual == expected and unit == systemd_worker_unit(config, host, managed_source=True, update_token=match[1], site_packages=snapshot.get("site_packages") or None)
        setup = re.match(r"# dgxm-setup-token=(s-[0-9a-f]{16}) ordinal=([1-9][0-9]*) source=([0-9a-f]{64})\n", unit)
        if setup:
            body = normal.removeprefix("# dgxm-managed-unit-v1\n")
            invalidator = systemd_generation_invalidator() + "\n"
            body = body.replace(invalidator, "").replace("ExecStart=", invalidator + "ExecStart=")
            body = body.replace("RestartSec=3\n", "RestartSec=3\nTimeoutStopSec=20\n")
            return unit == setup[0] + body
        return False
    except (KeyError, TypeError, ValueError, UnicodeError):
        return False


def capture_installation(config: ClusterConfig, host: HostConfig, run: HostRunner) -> dict[str, Any] | None:
    try:
        result = run(config, host, installation_script({"operation": "snapshot", "python_bin": config.python_bin, "address": host.address}), 30)
        if result.returncode:
            return None
        snapshot = json.loads(result.stdout)
        return snapshot if isinstance(snapshot, dict) and recognized_unit(config, host, snapshot) else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def change_installation(config: ClusterConfig, host: HostConfig, token: str, snapshot: dict[str, Any], run: HostRunner, *, compensate: bool = False, finalize: bool = False, preflight_script: str = "") -> Certainty:
    request = {"operation": "finalize" if finalize else "compensate" if compensate else "activate", "token": token, "prior": snapshot, "unit": base64.b64encode(systemd_worker_unit(config, host, managed_source=True, update_token=token, site_packages=snapshot.get("site_packages") or None).encode()).decode()}
    try:
        result = run(config, host, locked_script(config.python_bin, preflight_script + "\n" + installation_script(request), generation_fence=not finalize), 60)
    except (OSError, subprocess.TimeoutExpired):
        return Certainty.UNKNOWN
    lines = result.stdout.splitlines()
    if result.returncode == 0 and ("FINALIZED" if finalize else "COMPENSATED" if compensate else "ACTIVATED") in lines:
        return Certainty.SUCCEEDED
    if result.returncode == 0 and "ACTIVATION_PROBE_PRIOR" in lines:
        return Certainty.FAILED
    return Certainty.UNKNOWN


def stop_installation(config: ClusterConfig, host: HostConfig, snapshot: dict[str, Any], run: HostRunner, preflight_script: str) -> Certainty:
    request = {"operation": "stop", "prior": snapshot, "python_bin": config.python_bin, "address": host.address}
    script = locked_script(config.python_bin, preflight_script + "\n" + installation_script(request), generation_fence=True)
    try:
        result = run(config, host, script, 90)
    except (OSError, subprocess.TimeoutExpired):
        return Certainty.UNKNOWN
    return Certainty.SUCCEEDED if result.returncode == 0 and "STOPPED" in result.stdout.splitlines() else Certainty.UNKNOWN


def start_prior_installation(config: ClusterConfig, host: HostConfig, token: str, snapshot: dict[str, Any], run: HostRunner, preflight_script: str, generation: str) -> Certainty:
    validate_generation(generation)
    proc_table, endpoint = _proc_listener_target(host.address)
    request = {"token": token, "prior": snapshot, "python_bin": config.python_bin, "address": host.address, "generation": generation, "pattern": lifecycle._loop_regex(host), "proc_table": proc_table, "endpoint": endpoint}
    source = Path(update_installation_remote.__file__).read_text(encoding="utf-8")
    starter = Path(update_installation_start.__file__).read_text(encoding="utf-8")
    generation_source = _GENERATION_HELPER.split("operation=sys.argv[1]", 1)[0]
    payload = base64.b64encode(json.dumps(request).encode()).decode()
    script = source + "\ninstallation_api = dict(globals())\ngeneration_api = {}\nexec(" + repr(generation_source) + ", generation_api)\nexec(" + repr(starter) + ", globals())\nstart_restored(json.loads(base64.b64decode(" + repr(payload) + ")), installation_api, generation_api)\n"
    shell = preflight_script + "\n# DGXM_INSTALLATION_OPERATION=start_prior\n/usr/bin/python3 -I -S -B - <<'DGXM_PRIOR_START'\n" + script + "DGXM_PRIOR_START\n"
    try:
        result = run(config, host, locked_script(config.python_bin, shell), 90)
    except (OSError, subprocess.TimeoutExpired):
        return Certainty.UNKNOWN
    return Certainty.SUCCEEDED if result.returncode == 0 and "STARTED_PRIOR" in result.stdout.splitlines() else Certainty.UNKNOWN
