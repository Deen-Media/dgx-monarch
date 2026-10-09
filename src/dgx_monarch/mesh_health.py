"""Read-only driver health for cached attached meshes.

Worker-service probes cannot see abandoned samples or unresolved teardown, so
only the driver can report those states. `_busy_phase` separates a lost outcome
from a healthy operation still in progress. This module never calls workers or
takes the handle lock; recycle holds that lock before the registry lock, and a
diagnostic reader must not reverse that order.
"""
from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from . import client_lease, mesh_safety, mesh_teardown
from .log import get_logger

log = get_logger(__name__)

# These verdicts are unusable after `_busy_phase` excludes active work.
_UNUSABLE_VERDICTS = ("blocked", "unresolved")


def _busy_phase(handle: Any) -> str:
    """The operation the driver is still waiting on, or empty.

    `unresolved` covers both active work and a lost outcome. Do not call
    `token_within_authority` here because it logs on every telemetry poll.
    Teardown publishes its outcome before it clears the token, so the outcome
    check keeps a decided stop from reading as busy while the token remains.
    """
    if getattr(handle, "defunct", False):
        # Supervision already failed this fleet.
        return ""
    if client_lease.dispatch_in_flight(handle):
        state = getattr(handle, "setup_cleanup_state", None)
        return str(getattr(state, "phase", "") or "a worker RPC")
    if (mesh_safety.token_present(id(handle))
            and not mesh_safety.teardown_outcome_published(handle)):
        return "worker teardown"
    return ""


def _lease_total(handle: Any, attribute: str) -> int:
    """Sum a mutating lease map from a C-level private copy."""
    try:
        counts = list(dict(getattr(handle, attribute, None) or {}).values())
    except (AttributeError, TypeError, ValueError):
        return 0
    total = 0
    for count in counts:
        try:
            total += int(count)
        except (TypeError, ValueError):
            continue
    return total


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


@dataclass(frozen=True)
class MeshHealth:
    """The driver's own view of its cached fleet, as facts plus one verdict."""

    cached: bool = False
    verdict: str = "none"
    active_leases: int = 0
    abandoned_samples: int = 0
    setup_generation: int = 0
    busy_phase: str = ""
    pending: bool = False
    # Which stop left the fleet blocked. Empty when nothing recorded one, which
    # is the only case whose remedy has to cover every cause at once.
    block_cause: str = ""
    # Why the driver's Monarch transport cannot be reused, or empty. This is
    # process state, not handle state: while it is set every get_mesh raises,
    # whatever the registries hold, so an empty registry must not read idle.
    poison: str = ""

    @property
    def poisoned(self) -> bool:
        return bool(self.poison)

    @property
    def dirty(self) -> bool:
        if self.poison:
            return True
        if self.verdict in ("completed", "unknown", "none"):
            return False
        if self.pending and not self.busy_phase:
            return True
        if self.abandoned_samples > 0:
            return True
        return self.verdict in _UNUSABLE_VERDICTS and not self.busy_phase

    @property
    def state(self) -> str:
        """`ok`, `busy`, `dirty`, `poisoned`, `idle` or `unknown`.

        Confirmed stop is idle because `get_mesh` replaces it. Unreadable or
        actively mutating handles are not reported as dirty. A poisoned
        transport outranks every other reading, including an empty registry:
        there is nothing cached and no attach can make one either.
        """
        if self.poison:
            return "poisoned"
        if self.verdict == "unknown":
            return "unknown"
        if not self.cached or self.verdict == "completed":
            return "idle"
        if self.dirty:
            return "dirty"
        return "busy" if self.busy_phase else "ok"

    @property
    def reason(self) -> str:
        """Explain an unusable fleet in operator terms, or return empty."""
        if not self.dirty:
            return ""
        parts = []
        if self.poison:
            parts.append(
                "the driver's Monarch transport cannot be reused "
                f"({self.poison})")
        if self.abandoned_samples:
            parts.append(
                f"{_plural(self.abandoned_samples, 'abandoned sample')} may still be running")
        if self.pending:
            parts.append(
                "a worker fleet was created but was never published to the attached-mesh cache"
            )
        elif self.verdict == "blocked":
            label = mesh_teardown.label_for_cause(self.block_cause or None)
            parts.append(
                "an earlier attached-mesh process stop failed, so replacement "
                "is blocked" + (f" ({label})" if label else "")
            )
        elif self.verdict == "unresolved" and not self.busy_phase:
            parts.append("the previous teardown outcome is unresolved")
        return "; ".join(parts)

    @property
    def remedy(self) -> str:
        """Return the recovery action for a dirty fleet, or empty."""
        if not self.dirty:
            return ""
        if self.poison:
            # The poison lives in the driver process, so only a new one attaches.
            return "restart ComfyUI; no attached-mesh reset clears a poisoned transport"
        if self.pending and not self.abandoned_samples:
            # The reset route reads only the published cache, so it answers
            # "no attached mesh to reset" for a handle that never reached it.
            # The next render settles this row instead: through
            # mesh_creation.admission_occupant it creates over an interrupted
            # fleet that is gone, and refuses with a typed error naming the
            # restart while that fleet is still live.
            return ("start the next render: if the interrupted fleet is gone, it drops "
                    "this creation and attaches a fresh fleet; if that fleet is still live or its state "
                    "cannot be read, it refuses with a typed error that says to restart ComfyUI. "
                    "The sidebar Reset attached mesh cannot reach a fleet the cache never held")
        if self.verdict in _UNUSABLE_VERDICTS and not self.busy_phase:
            base = ("reset the attached mesh; restart ComfyUI if that stop cannot "
                    "be confirmed")
            if not self.block_cause:
                return base
            return f"{base}. {mesh_teardown.sentence_for_cause(self.block_cause)}"
        # Reset accepts abandoned samples, so the panel can clear this state.
        return (
            "reset the attached mesh (sidebar Reset attached mesh); if setup still "
            "fails, restart the worker service with dgxm restart"
        )

    def as_dict(self) -> dict:
        """Return the canonical `/dgxm/telemetry` mesh block."""
        return {
            "state": self.state,
            "cached": self.cached,
            "verdict": self.verdict,
            "active_leases": self.active_leases,
            "abandoned_samples": self.abandoned_samples,
            "setup_generation": self.setup_generation,
            "busy_phase": self.busy_phase,
            "pending": self.pending,
            "poisoned": self.poisoned,
            "reason": self.reason,
            "remedy": self.remedy,
        }


def _handle_health(handle: Any) -> MeshHealth:
    try:
        verdict = mesh_safety.coherent_lifecycle_verdict(handle, RuntimeError)
    except (AttributeError, RuntimeError):
        # Bounded coherent-read failure stays unknown.
        verdict = "unknown"
    try:
        generation = int(getattr(handle, "setup_generation", 0) or 0)
    except (TypeError, ValueError):
        generation = 0
    return MeshHealth(
        cached=True,
        verdict=verdict,
        active_leases=_lease_total(handle, "sample_leases"),
        abandoned_samples=_lease_total(handle, "abandoned_sample_leases"),
        setup_generation=generation,
        # Only `unresolved` can still represent active work.
        busy_phase=_busy_phase(handle) if verdict == "unresolved" else "",
        block_cause=str(mesh_teardown.block_cause(handle) or ""),
    )


def health_of(handles: Iterable[Any | MeshHealth], *, poison: str = "") -> MeshHealth:
    """The worst health across the driver's cached handles.

    Dirty outranks all other states because it blocks the next render. Busy
    outranks quiet states so diagnostics name the active operation. ``poison``
    is the driver's process-wide transport state, which belongs to no handle
    and is carried onto whichever one is reported, or onto an empty report.
    """
    ranked = [
        handle if isinstance(handle, MeshHealth) else _handle_health(handle)
        for handle in handles
    ]
    if not ranked:
        return MeshHealth(poison=poison)
    worst = max(ranked, key=lambda health: (
        health.dirty, bool(health.busy_phase), health.abandoned_samples,
        health.active_leases))
    return replace(worst, poison=poison) if poison else worst


def _attempt_is_live(attempt: Any) -> bool:
    """Whether the get_mesh frame that owns this creation attempt is running.

    An attempt whose owner exited is abandoned, not busy: the next get_mesh
    reaps it before it attaches, so there is nothing for the operator to wait
    on. An attempt without the owner fields, or one whose owner cannot be read,
    reads as live, so the block reports it busy instead of turning unknown.
    """
    if (getattr(attempt, "owner_thread_id", None) is None
            or getattr(attempt, "owner_frame", None) is None):
        return True
    try:
        from . import mesh_creation

        return bool(mesh_creation.attempt_is_active(attempt))
    except Exception as exc:
        log.debug("mesh creation attempt liveness unreadable: %r", exc)
        return True


def _pending_health(handle: Any, creating: bool) -> MeshHealth:
    """Project a private creation handle without reading its lock-protected state."""
    if creating:
        return MeshHealth(
            cached=True, verdict="unresolved", busy_phase="mesh creation", pending=True)
    return MeshHealth(cached=True, verdict="unresolved", pending=True)


def mesh_health_snapshot() -> dict:
    """Return telemetry health without raising or importing Monarch eagerly."""
    try:
        mesh_module: Any = sys.modules.get(f"{__package__}.mesh")
        if mesh_module is None:
            from . import mesh as mesh_module
        _MESH_LOCK = mesh_module._MESH_LOCK
        _MESHES = mesh_module._MESHES

        with _MESH_LOCK:  # snapshot only; verdicts are computed unlocked
            # Process-wide transport state; see MeshHealth.poison.
            poison = str(getattr(mesh_module, "_TRANSPORT_POISON", None) or "")
            public = list(_MESHES.values())
            public_ids = {id(handle) for handle in public}
            pending = [
                handle for handle in getattr(mesh_module, "_MESH_PENDING", {}).values()
                if id(handle) not in public_ids
            ]
            # Only an attempt whose owner is still running is work in flight.
            # An abandoned one reported as busy prints "wait for it, nothing to
            # fix" about a creation nobody is driving, and it hides a stranded
            # pending handle behind that row instead of reporting it dirty.
            # The liveness read happens here, under the same lock reap uses.
            attempts = [
                attempt
                for attempt in getattr(mesh_module, "_MESH_CREATING", {}).values()
                if _attempt_is_live(attempt)
            ]
            pending_ids = {id(handle) for handle in pending}
            creating_pending_ids = {
                id(getattr(attempt, "handle", None)) for attempt in attempts
                if getattr(attempt, "handle", None) is not None
            }
            creation_without_handle = any(
                getattr(attempt, "handle", None) is None
                or id(getattr(attempt, "handle", None)) not in public_ids | pending_ids
                for attempt in attempts
            )
        # Never take a handle lock here; the module docstring gives the lock order.
        healths = [_handle_health(handle) for handle in public]
        healths.extend(
            _pending_health(handle, id(handle) in creating_pending_ids)
            for handle in pending
        )
        if creation_without_handle:
            healths.append(MeshHealth(
                cached=True, verdict="unresolved", busy_phase="mesh creation", pending=True))
        return health_of(healths, poison=poison).as_dict()
    except Exception as exc:
        # Polling failures stay at DEBUG to avoid log floods.
        log.debug("mesh health unreadable: %r", exc)
        return MeshHealth(verdict="unknown").as_dict()
