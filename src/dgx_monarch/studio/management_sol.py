"""Explicit Sol maintenance through the existing staged-plan and idle guards.

A Sol install review stages a pip resolution and wheel build on every host.
Studio reuses the staging of a live review of the same plan rather than
building another, and removes staging that only its own abandoned reviews
used: reviews that expired or that a newer review of the same plan replaced,
with no apply ever started from them.
"""
from __future__ import annotations

import json
import re
import subprocess
import uuid
from typing import Any

from ..cli import sol_install, sol_smoke, sol_staging
from ..cli.sol_install_payload import canonical, protected
from ..config import ClusterConfig
from .contracts import StudioError, digest, timestamp
from .store import Store

REVIEW_SCAN = 256


def allowed_names(values: list[str]) -> list[str]:
    names = sorted({canonical(name) for name in values})
    if any(not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", name) or protected(name) for name in names):
        raise StudioError("sol_packages", "Protected runtime packages and invalid names cannot be changed.")
    return names


def read_plans(config: ClusterConfig, token: str, allowed: list[str], action: str) -> dict[str, Any]:
    plans = {}
    for host in config.hosts:
        result = sol_install._host(config, host, action, token, allowed)
        if result is None or result.get("status") != "ok":
            raise StudioError("sol_plan", "Sol plan is incomplete; private staging is retained without package apply.", 409)
        plans[host.name] = result["plan"]
    return plans


def fleet_identity(config: ClusterConfig) -> str:
    """The hosts and interpreter a staged plan belongs to."""
    return digest([config.python_bin, [(host.name, host.address, host.ssh_user) for host in config.hosts]])


def _stage(config: ClusterConfig, token: str, allowed: list[str], action: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plans, rows = {}, []
    for index, host in enumerate(config.hosts):
        result = sol_install._host(config, host, action, token, allowed)
        if result is not None and result.get("status") == "ok":
            plans[host.name] = result["plan"]
            rows.append({"ordinal": index, "plan": result["plan"]})
        else:
            rows.append({"ordinal": index, "error": sol_install.refusal_reason(result)})
    return rows, plans


def preview(action: str, config: ClusterConfig, parameters: dict[str, Any], *, reuse: str | None = None) -> dict[str, Any]:
    if not config.hosts or len({host.name for host in config.hosts}) != len(config.hosts):
        raise StudioError("sol_hosts", "Sol maintenance requires unique configured hosts.")
    if action == "sol-smoke":
        if any(host.gpus != 1 for host in config.hosts):
            raise StudioError("sol_hosts", "Sol smoke requires single-GPU hosts.")
        return {"hosts": len(config.hosts), "scope": "Explicit BF16 dense kernel control; no model fidelity evidence.",
                "timeout_s": parameters["timeout_s"]}
    allowed = allowed_names(parameters["allow_change"])
    supplied = parameters.get("plan")
    reuse = None if supplied else reuse
    token = supplied or reuse or uuid.uuid4().hex
    if not re.fullmatch(r"[a-f0-9]{32}", token):
        raise StudioError("sol_plan", "Existing plan token must be 32 lowercase hex characters.")
    rows, plans = _stage(config, token, allowed, "read" if supplied or reuse else "plan")
    if reuse and len(plans) != len(config.hosts):
        # The reused staging no longer reads back on every host; stage afresh.
        token, reuse = uuid.uuid4().hex, None
        rows, plans = _stage(config, token, allowed, "plan")
    staging = {"staging_owner": "operator" if supplied else "studio", "staging_reused": reuse is not None,
               "fleet_identity": fleet_identity(config)}
    if len(plans) != len(config.hosts):
        recovery = ("Incomplete private staging retained; no package apply attempted." if supplied else
                    "No package apply attempted. The next Sol review removes this incomplete staging.")
        return {"token": token, "ready": False, "allowed": allowed, "hosts": rows, **staging, "recovery": recovery}
    fleet_digest = sol_install.fleet_digest(config, token, allowed, plans)
    if parameters.get("plan_digest") and parameters["plan_digest"] != fleet_digest:
        raise StudioError("sol_plan", "Existing fleet plan digest does not match.", 409)
    return {"token": token, "fleet_digest": fleet_digest,
            "ready": True, "allowed": allowed, "hosts": rows, **staging}


def sol_reviews(store: Store) -> list[dict[str, Any]]:
    """Recent Sol install reviews and the operations applied from them."""
    with store.lock:
        rows = store.db.execute(
            "SELECT document FROM objects WHERE kind='operations' AND json_extract(document,'$.action')='sol-install' "
            "ORDER BY updated DESC LIMIT ?", (REVIEW_SCAN,)).fetchall()
    return [json.loads(row[0]) for row in rows]


def _plan(document: dict[str, Any]) -> dict[str, Any]:
    value = document.get("adapter_plan")
    return value if isinstance(value, dict) else {}


def _same_plan(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return first.get("allowed") == second.get("allowed") and first.get("fleet_identity") == second.get("fleet_identity")


def _reviews(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [document for document in documents if document.get("state") == "review" and not document.get("review_id")
            and _plan(document).get("staging_owner") == "studio" and isinstance(_plan(document).get("token"), str)]


def _live(review: dict[str, Any], reviews: list[dict[str, Any]], now: float) -> bool:
    """A review that can still be applied and that no newer review of its plan replaced."""
    plan = _plan(review)
    if review.get("consumed_by") or not review.get("expires_at", 0) >= now or plan.get("ready") is not True:
        return False
    return not any(other is not review and _plan(other).get("ready") is True and _same_plan(_plan(other), plan)
                   and other.get("created_at", 0) > review.get("created_at", 0) and _plan(other)["token"] != plan["token"]
                   for other in reviews)


def reusable_token(documents: list[dict[str, Any]], allowed: list[str], identity: str, now: float) -> str | None:
    """The token of the newest live review of this exact plan."""
    reviews = _reviews(documents)
    wanted = {"allowed": allowed, "fleet_identity": identity}
    for review in sorted(reviews, key=lambda row: row.get("created_at", 0), reverse=True):
        if _same_plan(_plan(review), wanted) and _live(review, reviews, now):
            return str(_plan(review)["token"])
    return None


def abandoned_tokens(documents: list[dict[str, Any]], now: float) -> list[str]:
    """Studio-minted tokens that no live review, apply or earlier removal claims."""
    applied = {_plan(document).get("token") for document in documents if document.get("review_id")}
    reviews = _reviews(documents)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for review in reviews:
        grouped.setdefault(_plan(review)["token"], []).append(review)
    tokens = [token for token, group in grouped.items()
              if token not in applied and not any(review.get("consumed_by") or review.get("staging_discard") for review in group)
              and not any(_live(review, reviews, now) for review in group)]
    return sorted(tokens)[:sol_staging.DISCARD_LIMIT]


def discard_abandoned(store: Store, config: ClusterConfig, documents: list[dict[str, Any]], now: float) -> list[dict[str, Any]]:
    """Remove abandoned staging on every host; an unknown outcome is retried at the next review."""
    tokens = abandoned_tokens(documents, now)
    if not tokens:
        return []
    outcomes: list[dict[str, str] | None] = []
    for host in config.hosts:
        try:
            outcomes.append(sol_staging.discard(config, host, tokens))
        except (OSError, ValueError, subprocess.SubprocessError):
            outcomes.append(None)
    rows = []
    for token in tokens:
        hosts = [None if outcome is None else outcome.get(token) for outcome in outcomes]
        state = ("unknown" if None in hosts else
                 "kept" if any(value in {"kept", "unsafe"} for value in hosts) else "removed")
        if state != "unknown":
            for review in _reviews(documents):
                if _plan(review)["token"] == token:
                    store.update("operations", review["id"], {"staging_discard": {"state": state, "at": timestamp()}})
        rows.append({"token": token, "state": state,
                     "hosts": [{"ordinal": index, "outcome": value or "unknown"} for index, value in enumerate(hosts)]})
    return rows


def review_install(store: Store, config: ClusterConfig, parameters: dict[str, Any]) -> dict[str, Any]:
    """A Sol install review that reuses live staging and clears abandoned staging first."""
    documents, now = sol_reviews(store), timestamp()
    cleanup = discard_abandoned(store, config, documents, now)
    reuse = reusable_token(documents, allowed_names(parameters["allow_change"]), fleet_identity(config), now)
    plan = preview("sol-install", config, parameters, reuse=reuse)
    return {**plan, **({"staging_cleanup": cleanup} if cleanup else {})}


def execute(action: str, config: ClusterConfig, parameters: dict[str, Any], reviewed: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if parameters["idle_maintenance"] is not True:
        raise StudioError("idle_maintenance", "Explicit idle-maintenance authority is required; close ComfyUI and settle actors first.", 409)
    if reviewed.get("ready") is False:
        raise StudioError("sol_plan", "The reviewed plan is incomplete and cannot be applied.", 409)
    rows: list[dict[str, Any]] = []
    if action == "sol-smoke":
        if preview(action, config, parameters) != reviewed:
            raise StudioError("operation_changed", "Sol smoke configuration changed.", 409)
        script = sol_smoke.smoke_script(config.python_bin)
        for index, host in enumerate(config.hosts):
            try:
                result = sol_smoke.run_on_host(config, host, script, timeout=parameters["timeout_s"], require_known_locality=True)
                if result.returncode:
                    return "unknown", {"source": "sol_smoke", "hosts": rows, "incomplete_host": index,
                                       "reason": "Kernel guard or execution failed; remaining hosts were not tested."}
                rows.append({"ordinal": index, "result": sol_smoke._parse_result(result.stdout)})
            except (OSError, subprocess.TimeoutExpired, ValueError):
                return "unknown", {"source": "sol_smoke", "hosts": rows, "incomplete_host": index}
        return "succeeded", {"source": "sol_smoke", "hosts": rows, "scope": reviewed["scope"]}
    token, allowed = reviewed["token"], allowed_names(parameters["allow_change"])
    plans = read_plans(config, token, allowed, "read")
    if sol_install.fleet_digest(config, token, allowed, plans) != reviewed["fleet_digest"]:
        raise StudioError("operation_changed", "The exact staged Sol plan changed after review.", 409)
    for index, host in enumerate(config.hosts):
        applied = sol_install._host(config, host, "apply", token, allowed, plans[host.name]["digest"])
        if applied is None or applied.get("status") != "ok":
            return "unknown", {"source": "sol_install", "token": token, "hosts": rows,
                               "incomplete_host": index, "recovery": "No restart or rollback attempted; staging is retained."}
        if applied["plan"] != plans[host.name]:
            return "unknown", {"source": "sol_install", "token": token, "hosts": rows, "incomplete_host": index}
        rows.append({"ordinal": index, "status": "verified"})
    return "succeeded", {"source": "sol_install", "token": token, "fleet_digest": reviewed["fleet_digest"],
                         "hosts": rows, "kernel_execution": "not_run"}
