"""Renew client leases for every live worker fleet.

One daemon thread in the ComfyUI process broadcasts renewals. When the client
exits they stop with no cleanup, so each actor's in-process reaper can fire.
Handles with a teardown token or a completed teardown are skipped. An in-flight
mutation is renewed; other non-live handles are renewed only while sample or
read leases remain. A fleet latched after a worker death is not renewed for
abandoned leases.

Synchronous cast failures are bounded by retry backoff. Asynchronous delivery
faults remain visible to supervision, where the fault hook counts exact
repeats, and lifecycle skip rules stop further casts once the handle is
non-live without leases. This module also publishes the driver's
`(pid, starttime)` marker for the actor sweep.
"""
from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from typing import Any

from . import actor_lifetime, mesh_safety, mesh_setup_state
from .log import get_logger

RENEW_INTERVAL_S = actor_lifetime.RENEW_INTERVAL_S
DEFAULT_LEASE_S = actor_lifetime.REAP_GRACE_S
MIN_LEASE_S = actor_lifetime.MIN_LEASE_S
MAX_LEASE_S = actor_lifetime.MAX_LEASE_S
GRACE_ENV = "DGXM_REAP_GRACE_S"
DISABLE_ENV = "DGXM_DISABLE_ACTOR_REAPER"
DRIVER_ENV = "DGXM_DRIVER"
MAX_CONSECUTIVE_FAILURES = 3
# A raised cast never arrived, so retry inside the grace while bounding faults
# from a fleet that remains unreachable.
RETIRE_RETRY_PASSES = 4

log = get_logger(__name__)

_THREAD: Any | None = None
_THREAD_STARTING = False
_START_LOCK = threading.Lock()
_FAILURES: dict[int, int] = {}
# Handles whose dead-fleet stand-down has been announced, by identity.
_STOOD_DOWN: set[int] = set()


def lease_seconds(environ: Mapping[str, str] | None = None) -> float:
    """The lease this driver advertises, or 0.0 when reaping is disabled.

    Invalid or out-of-range configuration falls back to the safe default.
    """
    source = os.environ if environ is None else environ
    if str(source.get(DISABLE_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}:
        return 0.0
    raw = source.get(GRACE_ENV)
    if raw is None or str(raw).strip() == "":
        return DEFAULT_LEASE_S
    try:
        value = float(raw)
    except (TypeError, ValueError):
        log.debug("%s is not a number (%r); using %.0fs", GRACE_ENV, raw, DEFAULT_LEASE_S)
        return DEFAULT_LEASE_S
    if not math.isfinite(value) or value < MIN_LEASE_S or value > MAX_LEASE_S:
        log.warning(
            "%s=%r is outside [%.0f, %.0f] and was not applied; the lease stays %.0fs. "
            "The floor exists so a rank cannot exit while its stop can still resolve.",
            GRACE_ENV, raw, MIN_LEASE_S, MAX_LEASE_S, DEFAULT_LEASE_S)
        return DEFAULT_LEASE_S
    return value


def plant_driver_marker(environ: MutableMapping[str, str] | None = None) -> str:
    """Publish driver identity for inheritance by spawned worker processes.

    `/proc/<pid>/environ` exposes exec-time state, not this process's later
    writes. Children inherit the marker, allowing the sweep to distinguish a
    live `(pid, starttime)` owner from a dead one.
    """
    from .cli import proc_identity

    target = os.environ if environ is None else environ
    marker = proc_identity.driver_identity()
    if marker:
        target[DRIVER_ENV] = marker
    # Empty is fail-closed as `held`; an unusable pair could falsely identify
    # this live driver's fleet as abandoned.
    return marker


def outstanding_leases(handle: Any, *, abandoned: bool = True) -> int:
    """Sample and read leases still outstanding on a handle.

    ``abandoned=False`` leaves out the leases a failed render abandoned.
    """
    total = 0
    names = ("sample_leases", "abandoned_sample_leases") if abandoned else ("sample_leases",)
    for name in names:
        leases = getattr(handle, name, None)
        values = getattr(leases, "values", None)
        if values is None:
            continue
        try:
            total += sum(int(count) for count in values())
        except (TypeError, ValueError):
            continue
    return total


def lifecycle_verdict(handle: Any) -> str:
    """`mesh_safety`'s coherent lifecycle read of this handle, or ``live`` if it raises.

    Coherent reads include `blocked` and `unresolved`; raw fields could miss
    them and keep a stranded fleet renewed indefinitely.
    """
    try:
        return mesh_safety.coherent_lifecycle_verdict(handle, RuntimeError)
    except Exception:
        return "live"


def dispatch_in_flight(handle: Any) -> bool:
    """True while a mutation RPC this driver is still awaiting is outstanding.

    `call_all` publishes `IN_PROGRESS` before mutation and retains it until a
    confirmed outcome. That is positive evidence the driver is waiting, even
    when a long load holds no sample lease. Decided outcomes stop renewal. The
    RPC budget bounds this state, and driver death ends the daemon renewer.
    """
    state = getattr(handle, "setup_cleanup_state", None)
    return (state is not None and getattr(state, "outcome", None)
            is mesh_setup_state.SetupCleanupOutcome.IN_PROGRESS)


DEAD_FLEET_SKIP = "blocked after a worker death with only abandoned leases"


def dead_fleet_latch(handle: Any) -> bool:
    """Return whether a stop timed out after a supervised worker death.

    Only a liveness proof can clear this state, and it ignores abandoned leases.
    Renewing those leases would keep sending calls to an unusable fleet. Other
    blocked states retain renewals because their processes may still be healthy.
    """
    from . import mesh_teardown

    return (mesh_teardown.block_cause(handle)
            == mesh_teardown.TAG_STOP_TIMEOUT_AFTER_DEATH)


def skip_reason(handle: Any) -> str | None:
    """Why this handle must not be renewed right now, or None."""
    if getattr(handle, "_creation_phase", "ready") == "rollback":
        return "mesh creation rollback in progress"
    if mesh_safety.token_present(id(handle)):
        return "deliberate teardown in flight"
    verdict = lifecycle_verdict(handle)
    if verdict == "completed":
        return "teardown complete"
    if verdict == "live":
        return None
    dead = verdict == "blocked" and dead_fleet_latch(handle)
    if outstanding_leases(handle, abandoned=not dead):
        return None
    if verdict == "unresolved" and dispatch_in_flight(handle):
        return None
    if dead and outstanding_leases(handle):
        return DEAD_FLEET_SKIP
    return f"{verdict} with no outstanding leases"


def _retired_skip(tracked: dict[int, int], key: int) -> bool:
    """True when this pass falls inside a stood-down handle's backoff window."""
    count = tracked.get(key, 0)
    if count < MAX_CONSECUTIVE_FAILURES or count % RETIRE_RETRY_PASSES == 0:
        return False
    tracked[key] = count + 1
    return True


def renew_pass(
    handles: Iterable[Any],
    lease_s: float,
    *,
    failures: dict[int, int] | None = None,
) -> int:
    """Cast one renewal at every eligible handle. Returns the cast count."""
    tracked = _FAILURES if failures is None else failures
    live: set[int] = set()
    renewed = 0
    for handle in handles:
        key = id(handle)
        live.add(key)
        if _retired_skip(tracked, key):
            continue
        reason = skip_reason(handle)
        if reason is not None:
            if reason == DEAD_FLEET_SKIP and key not in _STOOD_DOWN:
                # Report the transition once when renewals stop.
                _STOOD_DOWN.add(key)
                fate = (f"exits {lease_s:.0f}s after the last renewal that reached it"
                        if lease_s else "stays until its Worker service restarts, "
                        "because actor reaping is disabled")
                log.warning(
                    "client lease renewals stopped for a fleet latched after a worker "
                    "death; an actor process that survived %s.", fate)
            log.debug("client lease skipped (%s)", reason)
            continue
        try:
            handle.workers.renew_client_lease.broadcast(float(lease_s))
        except BaseException as exc:
            # Never let a cast failure end the renewer; the module docstring
            # says how faults are bounded.
            tracked[key] = tracked.get(key, 0) + 1
            if tracked[key] == MAX_CONSECUTIVE_FAILURES:
                # Warn once because these actors exit when their grace expires.
                log.warning(
                    "client lease renewal to a worker fleet failed %d times in a row (%r); "
                    "renewing it once every %d passes from here. Unless actor reaping is disabled, its "
                    "actor processes exit %.0fs after the last renewal that reached them.",
                    MAX_CONSECUTIVE_FAILURES, exc, RETIRE_RETRY_PASSES, lease_s)
            else:
                log.debug("client lease renewal failed (%d): %r", tracked[key], exc)
            continue
        tracked.pop(key, None)
        renewed += 1
    for key in [key for key in tracked if key not in live]:
        tracked.pop(key, None)
    _STOOD_DOWN.intersection_update(live)
    return renewed


def registry_snapshot() -> Sequence[Any]:
    """Live handles, copied under the registry lock and cast to outside it.

    Hold `_MESH_LOCK` only for the copy; broadcasts must not serialize creation.
    """
    from . import mesh

    with mesh._MESH_LOCK:
        return list({
            id(handle): handle
            for handle in (*mesh._MESHES.values(), *mesh._MESH_PENDING.values())
        }.values())


def _run(
    snapshot: Callable[[], Sequence[Any]],
    lease_s: float,
    interval_s: float,
    sleeper: Callable[[float], None],
) -> None:
    while True:
        try:
            renew_pass(snapshot(), lease_s)
        except BaseException as exc:
            log.debug("client lease pass failed: %r", exc)
        sleeper(interval_s)


def start(
    snapshot: Callable[[], Sequence[Any]] | None = None,
    *,
    lease_s: float | None = None,
    interval_s: float = RENEW_INTERVAL_S,
    sleeper: Callable[[float], None] = time.sleep,
    thread_factory: Any = threading.Thread,
) -> bool:
    """Start the renewal thread once per process. True when it started here.

    The first pass runs at once, so a disabled reaper's 0.0 lease reaches the
    actors before any grace expires.
    """
    global _THREAD, _THREAD_STARTING
    with _START_LOCK:
        if _THREAD is not None:
            alive = _thread_liveness(_THREAD)
            if _THREAD_STARTING:
                if alive is True:
                    _THREAD_STARTING = False
                    return False
                raise RuntimeError(
                    "a prior client-lease thread start has an unresolved outcome; "
                    "restart ComfyUI before publishing another worker fleet")
            if alive is None:
                raise RuntimeError(
                    "the existing client-lease thread's liveness could not be "
                    "confirmed; restart ComfyUI before publishing another worker fleet")
            if alive:
                return False
        source = registry_snapshot if snapshot is None else snapshot
        lease = lease_seconds() if lease_s is None else float(lease_s)
        thread = thread_factory(
            target=_run,
            args=(source, lease, interval_s, sleeper),
            name="dgxm-client-lease",
            daemon=True,
        )
        _THREAD = thread
        _THREAD_STARTING = True
        try:
            thread.start()
            _THREAD_STARTING = False
        except BaseException as exc:
            alive = _thread_liveness(thread)
            if alive is True:
                _THREAD_STARTING = False
            elif isinstance(exc, Exception) and alive is False:
                _THREAD = None
                _THREAD_STARTING = False
            raise
        return True


def _thread_liveness(thread: Any) -> bool | None:
    try:
        return bool(thread.is_alive())
    except Exception:
        return None
