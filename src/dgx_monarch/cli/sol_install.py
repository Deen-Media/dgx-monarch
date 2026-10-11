"""Plan-first optional Sol installation across configured interpreters."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import uuid
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import TypedDict, cast

from ..config import ClusterConfig, HostConfig
from . import lifecycle
from .lifecycle_lock import lifecycle_lock_source
from .probe_certainty import UNKNOWN_EXIT
from .sol_install_guard import PROBE_SOURCE
from .sol_install_payload import canonical, protected


class PublicChange(TypedDict):
    name: str
    old: str | None
    new: str


class PublicPlan(TypedDict):
    digest: str
    allowed: list[str]
    changes: list[PublicChange]


class HostResult(TypedDict, total=False):
    status: str
    plan: PublicPlan
    reason: str


MARKER = "DGXM_SOL_INSTALL="


def host_script(config: ClusterConfig, action: str, token: str, allowed: Collection[str], expected: str = "") -> str:
    payload = Path(__file__).with_name("sol_install_payload.py").read_text()
    payload = payload.replace("from .sol_install_guard import require_idle\n", "")
    payload = payload.replace("from .sol_install_requirements import validate_wheel_requirements\n", "")
    requirements = Path(__file__).with_name("sol_install_requirements.py").read_text()
    requirements = requirements.replace("from __future__ import annotations\n", "")
    request = repr({"action": action, "token": token, "allowed": sorted(allowed), "expected": expected})
    source = payload + "\n" + requirements + "\n" + PROBE_SOURCE + "\n" + lifecycle_lock_source() + f'''
request = {request}
try:
    lock_fd = acquire_lifecycle_lock(Path.home(), True)
    if request['action'] == 'plan':
        require_idle()
        value = plan_host(request['token'], request['allowed'])
    elif request['action'] == 'read':
        value = read_plan(request['token'])
    else:
        require_idle()
        value = apply_host(request['token'], request['expected'], request['allowed'])
    public = {{'digest': value['digest'], 'allowed': value['allowed'],
               'changes': [{{key: row[key] for key in ('name', 'old', 'new')}} for row in value['changes']]}}
    print({MARKER!r} + json.dumps({{'status': 'ok', 'plan': public}}))
except Exception as exc:
    # Only our bounded messages are public; pip/transport logs can carry secrets.
    reason = str(exc) if type(exc) in (SolInstallError, IdleMaintenanceError, WheelRequirementsError) else "observation_or_operation_failed"
    print({MARKER!r} + json.dumps({{'status': 'failed', 'reason': reason}}))
    raise SystemExit(1)
'''
    shell = f"set -eu\n{lifecycle._pybin_shell(config.python_bin)}\n\"$PYBIN\" -I -B - <<'DGXM_SOL_PAYLOAD'\n{source}\nDGXM_SOL_PAYLOAD\n"
    return shell


def _host(config: ClusterConfig, host: HostConfig, action: str, token: str,
          allowed: Collection[str], expected: str = "") -> HostResult | None:
    try:
        result = lifecycle.run_on_host(config, host, host_script(config, action, token, allowed, expected),
                                       timeout=1800)
        rows = [line[len(MARKER):] for line in result.stdout.splitlines() if line.startswith(MARKER)]
        if len(rows) != 1:
            return None
        value = json.loads(rows[0])
        if not isinstance(value, dict):
            return None
        if value.get("status") != "ok" or result.returncode != 0:
            return cast(HostResult, value) if value.get("status") == "failed" else None
        plan = value["plan"]
        if not re.fullmatch(r"[a-f0-9]{64}", plan["digest"]) or plan["allowed"] != sorted(allowed):
            return None
        for row in plan["changes"]:
            if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", row["name"]):
                return None
            for key in ("old", "new"):
                if row[key] is not None and (not isinstance(row[key], str) or len(row[key]) > 100
                                             or not re.fullmatch(r"[a-zA-Z0-9.+!_-]+", row[key])):
                    return None
        return cast(HostResult, value)
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError):
        return None


def fleet_digest(config: ClusterConfig, token: str, allowed: Collection[str], plans: Mapping[str, PublicPlan]) -> str:
    value = {"token": token, "python": config.python_bin, "allowed": sorted(allowed),
             "hosts": [(host.name, host.address, host.ssh_user, plans[host.name]["digest"]) for host in config.hosts]}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def refusal_reason(result: HostResult | None) -> str:
    reason = result.get("reason", "operation_failed") if result else "observation_unknown"
    if not isinstance(reason, str) or len(reason) > 256 or not re.fullmatch(r"[A-Za-z0-9 _;.-]+", reason):
        return "operation_failed"
    return reason


def run(config: ClusterConfig | None, *, apply: bool = False, plan: str | None = None,
        plan_digest: str | None = None, allow_change: Sequence[str] = (), idle_maintenance: bool = False) -> int:
    """Stage or apply one reviewed fleet plan; no lifecycle actions are implicit."""
    if not config or not config.hosts or len({host.name for host in config.hosts}) != len(config.hosts):
        print("Sol install requires unique configured Worker hosts.")
        return 1
    allowed = {canonical(name) for name in allow_change}
    if any(not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", name) or protected(name) for name in allowed):
        print("--allow-change cannot name protected runtime packages or invalid distribution names.")
        return 1
    if apply and (not idle_maintenance or not plan or not plan_digest):
        print("Apply requires --plan, --plan-digest and --idle-maintenance; stop ComfyUI and GPU clients first.")
        return 1
    token = plan or uuid.uuid4().hex
    if not re.fullmatch(r"[a-f0-9]{32}", token):
        print("Invalid plan token.")
        return 1
    print("Sol staging downloads/builds into private per-host directories; it does not install packages.")
    plans: dict[str, PublicPlan] = {}
    failed = False
    unknown = False
    for host in config.hosts:
        result = _host(config, host, "read" if plan else "plan", token, allowed)
        if result is None:
            print(f"{host.name}: planning/readback unknown.")
            unknown = True
        elif result["status"] != "ok":
            reason = refusal_reason(result)
            print(f"{host.name}: planning refused: {reason}.")
            failed = True
        else:
            plans[host.name] = result["plan"]
            for row in result["plan"]["changes"]:
                print(f"{host.name}: {row['name']} {row['old'] or '(absent)'} -> {row['new']}")
            if not result["plan"]["changes"]:
                print(f"{host.name}: requested packages already satisfy the pinned selection.")
    if failed or unknown:
        print(f"Incomplete plan retained as {token}; no package apply attempted.")
        return UNKNOWN_EXIT if unknown else 1
    actual_digest = fleet_digest(config, token, allowed, plans)
    print(f"Plan: {token}\nPlan digest: {actual_digest}")
    if not apply:
        print("Review every package change. Apply with this token/digest and the same --allow-change names plus --idle-maintenance.")
        return 0
    if plan_digest != actual_digest:
        print("Reviewed fleet plan/configuration changed; no apply attempted.")
        return 1
    completed: list[str] = []
    for host in config.hosts:
        result = _host(config, host, "apply", token, allowed, plans[host.name]["digest"])
        if result is None or result["status"] != "ok":
            print(f"{host.name}: apply/readback incomplete: {refusal_reason(result)}; "
                  "package state may be partial. No restart or rollback attempted.")
            print(f"Completed hosts: {', '.join(completed) or 'none'}. Remaining hosts were not changed.")
            return UNKNOWN_EXIT
        completed.append(host.name)
        print(f"{host.name}: exact package readback and protected runtime file hashes match.")
    print("Sol installed. Kernel execution remains untested; use the separate explicit sol-smoke action while idle.")
    return 0
