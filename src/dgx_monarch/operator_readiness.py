"""Canonical readiness data in a stdlib-only leaf for every operator surface.

Actions and commands are inert. Details copy only explicit sanitized overrides.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from . import operator_actions
from .mesh_evidence import mesh_block_dirty
from .mesh_evidence import strict_count as _strict_count

SCHEMA_VERSION: Final = 1
OVERALL_STATES: Final = ("ready", "degraded", "blocked", "unknown")
LAYER_NAMES: Final = ("worker_service", "attached_mesh", "render_session")
LAYER_STATES: Final = ("ready", "idle", "active", "blocked", "unknown")
ACTION_SAFETY: Final = ("automatic", "manual", "destructive")
_MISSING = object()
# How a layer state votes in the overall verdict; anything else is ready.
_LAYER_OVERALL = {"blocked": "blocked", "unknown": "unknown", "active": "degraded"}
def _layer(state: str, detail: str) -> dict[str, str]:
    return {"state": state, "detail": detail}

def _safe_override(details: object, name: str, fallback: str) -> str:
    """Use a bounded plain string without stringifying foreign objects."""
    if not isinstance(details, Mapping):
        return fallback
    value = details.get(name)
    if not isinstance(value, str):
        return fallback
    value = " ".join(value.split())
    return value[:400] if value else fallback

def _error_present(value: object) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True

def _workers_layer(workers: object) -> tuple[dict[str, str], list[dict[str, object]], str]:
    if not isinstance(workers, Sequence) or isinstance(workers, (str, bytes, bytearray)):
        return (
            _layer("unknown", "Worker service observations are missing or malformed."),
            [operator_actions.inspect_workers()],
            "unknown",
        )
    if not workers:
        return (
            _layer("unknown", "No worker service observation is available."),
            [operator_actions.inspect_workers()],
            "unknown",
        )

    unknown = False
    blocked = False
    for worker in workers:
        if not isinstance(worker, Mapping):
            unknown = True
            continue
        if any(_error_present(worker.get(key)) for key in ("status_error", "health_error", "error")):
            unknown = True
            continue

        cleanup_failed = worker.get("setup_cleanup_failed", _MISSING)
        if cleanup_failed is not _MISSING:
            if not isinstance(cleanup_failed, bool):
                unknown = True
            elif cleanup_failed:
                blocked = True

        observed = False
        for key in ("healthy", "up", "running", "listening"):
            value = worker.get(key, _MISSING)
            if value is _MISSING:
                continue
            observed = True
            if not isinstance(value, bool):
                unknown = True
            elif not value:
                blocked = True
        for key, good, bad in (("loop", "running", "stopped"), ("port", "open", "closed")):
            value = worker.get(key, _MISSING)
            if value is _MISSING:
                continue
            observed = True
            if not isinstance(value, str):
                unknown = True
            elif value.strip().lower() == bad:
                blocked = True
            elif value.strip().lower() != good:
                unknown = True

        # A rank/world pair is the actor status reply: it must be well formed, but
        # it does not observe the worker service, so it never sets `observed`.
        rank, world = worker.get("rank", _MISSING), worker.get("world", _MISSING)
        if rank is not _MISSING or world is not _MISSING:
            if (
                isinstance(rank, bool)
                or not isinstance(rank, int)
                or isinstance(world, bool)
                or not isinstance(world, int)
                or world < 1
                or not 0 <= rank < world
            ):
                unknown = True
        if not observed:
            unknown = True

    if blocked:
        return (
            _layer("blocked", "At least one worker service is not ready."),
            [operator_actions.inspect_workers()],
            "blocked",
        )
    if unknown:
        return (
            _layer("unknown", "At least one worker service reported an error or no usable state."),
            [operator_actions.inspect_workers()],
            "unknown",
        )
    count = len(workers)
    noun = "service" if count == 1 else "services"
    return _layer("ready", f"{count} worker {noun} ready."), [], "ready"


def _mesh_layer(
    mesh: object, sanitized_details: object
) -> tuple[dict[str, str], list[dict[str, object]], str]:
    if not isinstance(mesh, Mapping):
        return (
            _layer("unknown", "Attached mesh state is missing or malformed."),
            [operator_actions.unknown(
                "inspect-attached-mesh", "Inspect the attached mesh",
                "The driver did not provide a usable mesh state.")],
            "unknown",
        )

    state_value = mesh.get("state", _MISSING)
    state = state_value.strip().lower() if isinstance(state_value, str) else None
    malformed = state is None

    counts: dict[str, int] = {}
    unreported: set[str] = set()
    for key in ("active_leases", "abandoned_samples", "abandoned_leases"):
        value = mesh.get(key, _MISSING)
        if value is _MISSING:
            unreported.add(key)
            continue
        count = _strict_count(value)
        if count is None:
            malformed = True
        else:
            counts[key] = count

    replacement = mesh.get("replacement_blocked", False)
    replacement_valid = (
        replacement is None or replacement is False or replacement is True
        or isinstance(replacement, str)
    )
    if not replacement_valid:
        malformed = True
    poisoned = mesh.get("poisoned", False)
    if not isinstance(poisoned, bool):
        malformed = True
        poisoned = False
    verdict_value = mesh.get("verdict", "")
    verdict = verdict_value.strip().lower() if isinstance(verdict_value, str) else ""
    if not isinstance(verdict_value, str):
        malformed = True
    known_verdicts = {"none", "completed", "live", "blocked", "unresolved", "unknown"}
    if verdict not in known_verdicts:
        malformed = True

    busy_phase = mesh.get("busy_phase", "")
    if not isinstance(busy_phase, str):
        malformed = True
        busy_phase = ""
    if mesh_block_dirty(mesh):
        detail = _safe_override(
            sanitized_details, "attached_mesh",
            "The attached mesh is dirty, poisoned, or blocked from safe replacement.")
        return _layer("blocked", detail), [operator_actions.recover_mesh()], "blocked"
    # Accept only the combinations mesh_health.MeshHealth publishes, plus any block
    # whose state is unknown. A completed (stopped) handle can keep inert lease
    # counts, so idle with a completed verdict may report active leases; any other
    # contradiction reads as malformed.
    active_leases = counts.get("active_leases", 0)
    coherent = (
        state == "unknown"
        or (state == "idle" and verdict in {"none", "completed"}
            and not busy_phase.strip() and not (verdict == "none" and active_leases > 0))
        or (state == "ok" and verdict == "live" and not busy_phase.strip())
        or (state == "busy" and verdict == "unresolved" and bool(busy_phase.strip()))
    )
    if not coherent:
        malformed = True
    if malformed or state not in {"ok", "idle", "busy", "unknown"}:
        return (
            _layer("unknown", "Attached mesh state is missing or malformed."),
            [operator_actions.unknown(
                "inspect-attached-mesh", "Inspect the attached mesh",
                "The driver did not provide a usable mesh state.")],
            "unknown",
        )
    if state == "unknown":
        return (
            _layer("unknown", "The driver could not determine attached mesh state."),
            [operator_actions.unknown(
                "inspect-attached-mesh", "Inspect the attached mesh",
                "The driver could not determine whether the cached mesh is usable.")],
            "unknown",
        )
    # An omitted count is not an observed zero. Only the healthy outcomes refuse
    # on it: a dirty or busy mesh already outranks unknown. No producer emits
    # abandoned_leases; it stays an alias of abandoned_samples.
    blind_counts = (
        "active_leases" in unreported
        or {"abandoned_samples", "abandoned_leases"} <= unreported
    )
    if state == "idle" and not blind_counts:
        detail = _safe_override(
            sanitized_details, "attached_mesh",
            "No mesh is attached; the next render can attach one.")
        return _layer("idle", detail), [], "ready"
    if state == "busy" or active_leases > 0:
        detail = _safe_override(
            sanitized_details, "attached_mesh",
            "The attached mesh has active work in flight.")
        return _layer("active", detail), [operator_actions.wait()], "degraded"
    if blind_counts:
        return (
            _layer("unknown", "Attached mesh lease counts were not reported."),
            [operator_actions.unknown(
                "inspect-attached-mesh", "Inspect the attached mesh",
                "The driver did not report whether leases or samples are live.")],
            "unknown",
        )
    detail = _safe_override(
        sanitized_details, "attached_mesh", "The attached mesh is ready for work.")
    return _layer("ready", detail), [], "ready"


def _render_layer(
    render: object, sanitized_details: object
) -> tuple[dict[str, str], list[dict[str, object]], str]:
    if not isinstance(render, Mapping):
        return (
            _layer("unknown", "Render session state is missing or malformed."),
            [operator_actions.unknown(
                "inspect-render-session", "Inspect the render session",
                "The driver did not provide a usable render state.")],
            "unknown",
        )
    if any(_error_present(render.get(key)) for key in ("status_error", "error")):
        return (
            _layer("unknown", "Render session state could not be read."),
            [operator_actions.unknown(
                "inspect-render-session", "Inspect the render session",
                "The driver could not determine whether a render is active.")],
            "unknown",
        )
    active = render.get("active", _MISSING)
    count_value = render.get("active_renders", _MISSING)
    count = None if count_value is _MISSING else _strict_count(count_value)
    if count_value is not _MISSING and count is None:
        active = _MISSING
    if active is _MISSING and count is not None:
        active = count > 0
    # A supplied count is evidence, not a hint: one that contradicts `active`
    # makes the session unknown. Only a missing count takes the one-render
    # fallback below.
    if isinstance(active, bool) and count is not None and active != (count > 0):
        active = _MISSING
    if not isinstance(active, bool):
        return (
            _layer("unknown", "Render session state is missing or malformed."),
            [operator_actions.unknown(
                "inspect-render-session", "Inspect the render session",
                "The driver did not provide a usable render state.")],
            "unknown",
        )
    if active:
        if count in (None, 0):
            count = 1
        detail = _safe_override(
            sanitized_details, "render_session",
            f"{count} render session{' is' if count == 1 else 's are'} active.")
        return _layer("active", detail), [operator_actions.wait()], "degraded"
    detail = _safe_override(
        sanitized_details, "render_session", "No render session is active.")
    return _layer("idle", detail), [], "ready"


def _overall(states: Sequence[str]) -> str:
    for state in ("blocked", "unknown", "degraded"):
        if state in states:
            return state
    return "ready"


def readiness_from_telemetry(
    workers: object,
    mesh: object,
    render: object,
    telemetry_error: object = None,
    *,
    sanitized_details: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Normalize the four public telemetry observations.

    Missing or malformed observations never become healthy by default. A known
    blocked layer outranks an unrelated unknown observation.
    """
    worker_layer, worker_actions, worker_overall = _workers_layer(workers)
    mesh_layer, mesh_actions, mesh_overall = _mesh_layer(mesh, sanitized_details)
    render_layer, render_actions, render_overall = _render_layer(render, sanitized_details)
    actions = [*worker_actions, *mesh_actions, *render_actions]
    states = [worker_overall, mesh_overall, render_overall]
    if _error_present(telemetry_error):
        states.append("unknown")
        actions.append(operator_actions.unknown(
            "restore-telemetry", "Restore telemetry",
            "Telemetry refresh failed; the last observation cannot establish current readiness."))
    return {
        "schema_version": SCHEMA_VERSION,
        "overall": _overall(states),
        "lifecycle": {
            "worker_service": worker_layer,
            "attached_mesh": mesh_layer,
            "render_session": render_layer,
        },
        "actions": operator_actions.ordered(actions),
    }


def readiness_from_telemetry_payload(
    payload: object,
    *,
    sanitized_details: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Convenience wrapper for a complete ``/dgxm/telemetry`` object."""
    if not isinstance(payload, Mapping):
        return readiness_from_telemetry(
            None, None, None, "malformed", sanitized_details=sanitized_details)
    return readiness_from_telemetry(
        payload.get("workers"),
        payload.get("mesh"),
        payload.get("render"),
        payload.get("telemetry_error"),
        sanitized_details=sanitized_details,
    )


def readiness_from_doctor_rows(rows: object) -> dict[str, object]:
    """Normalize doctor ``{status, name, detail}`` rows without mutating them."""
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)) or not rows:
        return {
            "schema_version": SCHEMA_VERSION,
            "overall": "unknown",
            "lifecycle": {
                "worker_service": _layer("unknown", "Doctor supplied no usable worker checks."),
                "attached_mesh": _layer("unknown", "Doctor supplied no usable mesh check."),
                "render_session": _layer("unknown", "Doctor does not observe render sessions."),
            },
            "actions": [operator_actions.unknown(
                "rerun-doctor", "Run doctor again",
                "The doctor report was empty or malformed.")],
        }

    malformed = 0
    failures = 0
    warnings = 0
    permission_issue = False
    worker_states: list[str] = []
    mesh_states: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            malformed += 1
            continue
        status_value, name_value, detail_value = (
            row.get("status"), row.get("name"), row.get("detail"))
        if not isinstance(status_value, str) or not isinstance(name_value, str):
            malformed += 1
            continue
        status = status_value.strip().lower()
        if status not in {"ok", "warn", "fail"}:
            malformed += 1
            continue
        failures += status == "fail"
        warnings += status == "warn"
        name = name_value.strip().lower()
        if name.endswith(("worker service", "worker loop")):
            worker_states.append(status)
        if name == "mesh health":
            # Doctor reports "no driver reachable" as OK because a stopped ComfyUI
            # is not a failed install; that row is no evidence of a ready mesh.
            # Only the fixed details mesh_health_row emits set a positive mesh
            # state, and no foreign detail is copied into the report.
            detail = detail_value.strip().lower() if isinstance(detail_value, str) else ""
            if status == "fail":
                mesh_states.append("blocked")
            elif status == "warn":
                mesh_states.append("unknown")
            elif detail.startswith("attached mesh ready,"):
                mesh_states.append("ready")
            elif detail.startswith("no attached mesh;"):
                mesh_states.append("idle")
            elif detail.startswith("no driver reachable;"):
                mesh_states.append("unknown")
            elif detail.endswith(
                    " in flight; an attached mesh its own driver is working on "
                    "is not a dirty one"):
                mesh_states.append("active")
            else:
                mesh_states.append("unknown")
        if name == "config permissions" and status != "ok":
            permission_issue = True

    actions: list[dict[str, object]] = []
    if failures:
        actions.append(operator_actions.action(
            "resolve-doctor-failures", "Resolve failed doctor checks",
            f"Doctor found {failures} failed check{'s' if failures != 1 else ''}.",
            "manual", command="dgxm doctor", docs_ref="docs/TROUBLESHOOTING.md"))
    if warnings:
        actions.append(operator_actions.action(
            "review-doctor-warnings", "Review doctor warnings",
            f"Doctor found {warnings} warning{'s' if warnings != 1 else ''}.",
            "manual", command="dgxm doctor", docs_ref="docs/TROUBLESHOOTING.md"))
    if permission_issue:
        actions.append(operator_actions.tighten_config_permissions())
    if malformed:
        actions.append(operator_actions.unknown(
            "rerun-doctor", "Run doctor again",
            "At least one doctor row was malformed and was not trusted."))

    if "fail" in worker_states:
        worker_layer = _layer("blocked", "At least one worker service check failed.")
        actions.append(operator_actions.inspect_workers())
    elif "warn" in worker_states:
        worker_layer = _layer("unknown", "At least one worker service check was inconclusive.")
        actions.append(operator_actions.inspect_workers())
    elif worker_states:
        worker_layer = _layer("ready", "All observed worker services are ready.")
    else:
        worker_layer = _layer("idle", "No cluster worker service checks were required.")

    if "blocked" in mesh_states:
        mesh_layer = _layer("blocked", "The attached mesh health check failed.")
        actions.append(operator_actions.recover_mesh())
    elif "unknown" in mesh_states:
        mesh_layer = _layer("unknown", "The attached mesh health check was inconclusive.")
        actions.append(operator_actions.unknown(
            "inspect-attached-mesh", "Inspect the attached mesh",
            "Doctor could not establish whether the attached mesh is usable."))
    elif "active" in mesh_states:
        mesh_layer = _layer("active", "The attached mesh is changing or has work in flight.")
        actions.append(operator_actions.wait())
    elif "ready" in mesh_states:
        mesh_layer = _layer("ready", "The attached mesh health check passed.")
    elif mesh_states:
        mesh_layer = _layer("idle", "No mesh is attached; the next render can attach one.")
    else:
        mesh_layer = _layer("idle", "No cluster mesh check was required.")

    # Take the worst of the row counts and each layer's vote, so clean counts
    # never publish `ready` over an unknown mesh. render_session stays out: doctor
    # never observes one, so its permanent unknown would forbid a ready report.
    counted = "blocked" if failures else (
        "unknown" if malformed else ("degraded" if warnings else "ready"))
    aggregate = _overall([
        counted,
        _LAYER_OVERALL.get(worker_layer["state"], "ready"),
        _LAYER_OVERALL.get(mesh_layer["state"], "ready"),
    ])
    return {
        "schema_version": SCHEMA_VERSION,
        "overall": aggregate,
        "lifecycle": {
            "worker_service": worker_layer,
            "attached_mesh": mesh_layer,
            "render_session": _layer("unknown", "Doctor does not observe render sessions."),
        },
        "actions": operator_actions.ordered(actions),
    }
