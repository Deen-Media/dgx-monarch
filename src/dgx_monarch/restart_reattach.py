"""Attach again inside the render when every Worker loop restarted under it.

docs/TROUBLESHOOTING.md #2. A Worker loop that restarts under an attached mesh
takes the mesh's actors with it, and a message to one of them comes back to
the driver's fault hook as undeliverable, never to the call that sent it. The
render that meets the old actors used to wait the capacity guard's whole 30 s
deadline, refuse, and leave the reset to the operator. Now:

1. The capacity price wait ends within one poll of a delivery failure that
   names a fleet host and the mesh it prices on (``delivery_faults.await_rank``,
   ``watched_mesh``) and raises ``WorkerDeliveryError`` (``refuse_undelivered``).
2. A sampler render that owns its render session, or the UNET Loader's eager
   load, calls ``heal``. It reads each loop's id with the listener-generation
   probe every attach already runs, and compares it with the id this driver
   dialed (``mesh_attach.last_dialed``).
3. When every loop serves a new id, the loops restarted and the actors are
   gone. ``heal`` publishes that death the way a supervision failure does, and
   stops the fleet through the deliberate teardown with a short bound: in CPU
   runs a stop addressed to the old ids was never acknowledged, and the old
   processes died with their loops. As after any stop that times out, the
   latch releases only when the liveness check finds every host's old
   processes gone (DESIGN.md section 5.1). ``mesh.ensure_live``, or the
   caller's ``respawn``, then attaches by the new ids through the ordinary
   attach, and the caller dispatches again on the new mesh.
4. Anything else refuses at once and names its remedy: some loops moved and
   some did not, none moved, an id the probe could not read, or old processes
   the liveness check did not find gone.

Bounds: ``GENERATION_PROBE_S`` per host for the reading, two ``STOP_BOUND_S``
waits for the stop, the liveness fleet budget, and the attach's own budgets.
The replacement attach reuses the one transport this process bound. After an
outage longer than Monarch's delivery timeout the channel to that address
stays closed, that attach fails at its config-push budget, and the session
asks for the documented driver relaunch, as before.

Every detour writes one ``attach_trace event=restart_reattach`` line, one
``restart_reattach outcome=...`` log line and one telemetry event.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NoReturn

from . import attach_trace, delivery_faults, worker_proc_id
from .delivery_faults import WorkerDeliveryError
from .log import get_logger
from .refusal import RefusalClass, refusal

log = get_logger(__name__)

EVENT = "restart_reattach"
# capacity_agreement.GUARD, spelled out because the refusal ledger reads the literal.
GUARD = "cross_rank_capacity"
TROUBLESHOOTING = 2
# Each of the two waits the deliberate teardown makes on a fleet whose loops
# restarted: the group teardown and the proc stop. Neither can be acknowledged.
STOP_BOUND_S = 2.0

ALL, PARTIAL, NONE, UNKNOWN = "all", "partial", "none", "unknown"
_RESTART_TOGETHER = (
    "run `dgxm restart` so every Worker loop restarts together, then queue again")
_ONE_SAMPLER_RENDER = (
    "run one render through KSampler (DGX Monarch), not KSampler Pipeline, "
    "Identity Gate or Fleet KSampler, with the loader and Init inputs unchanged: "
    "it attaches again by the new ids. Then queue this workflow again")


def watched_mesh(handle: Any) -> str | None:
    """The uid of the actor mesh a price waits on, or None when it cannot be read.

    With None the wait keeps its whole deadline, as before #47: a failure it
    cannot tie to this mesh never ends it.
    """
    from .mesh_runtime import get_off_loop

    try:
        name = get_off_loop(handle.workers._name, delivery_faults.NAME_READ_S)
    except Exception as exc:
        log.warning("the Worker mesh's name could not be read (%s); the capacity "
                    "price waits its whole deadline", type(exc).__name__)
        return None
    return delivery_faults.mesh_uid(name)


@dataclass(frozen=True)
class Moves:
    """Which loops serve a new id since this driver dialed them."""

    moved: tuple[str, ...] = ()
    kept: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()  # "<name> (<address>: <reason>)"

    @property
    def verdict(self) -> str:
        if self.unknown:
            return UNKNOWN
        if self.moved and not self.kept:
            return ALL
        return PARTIAL if self.moved else NONE


@dataclass(frozen=True)
class Reattached:
    """What one detour did, for the render's records."""

    handle: Any
    wall_s: float
    moved: tuple[str, ...]
    teardown: str  # "stop_confirmed" or "liveness_release"


def read_moves(config: Any) -> Moves:
    """Read every loop's id now and compare it with the id this driver dialed."""
    from . import mesh_attach
    from .cli import lifecycle, listener_generation

    hosts = tuple(getattr(config, "hosts", None) or ())
    try:
        readings = listener_generation.fleet(
            config, runner=lifecycle.run_on_host, timeout=mesh_attach.GENERATION_PROBE_S)
    except Exception as exc:
        return Moves(unknown=tuple(
            f"{host.name} ({host.address}: the probe raised {type(exc).__name__})"
            for host in hosts))
    moved: list[str] = []
    kept: list[str] = []
    unknown: list[str] = []
    for host in hosts:
        fields = readings.get(host.address) or {}
        value = fields.get(worker_proc_id.FIELD)
        dialed = mesh_attach.last_dialed(host.address)
        if not worker_proc_id.is_value(value):
            unknown.append(f"{host.name} ({host.address}: {worker_proc_id.problem(fields)[0]})")
        elif dialed is None:
            unknown.append(f"{host.name} ({host.address}: this driver holds no dialed id for it)")
        else:
            (moved if value != dialed else kept).append(host.address)
    return Moves(tuple(moved), tuple(kept), tuple(unknown))


def _labels(handle: Any, addresses: Any) -> str:
    names = {str(getattr(host, "address", "")): str(getattr(host, "name", "") or "")
             for host in getattr(getattr(handle, "config", None), "hosts", None) or ()}
    return ", ".join(f"{names[address]} ({address})" if names.get(address) else address
                     for address in addresses)


def _refuse(text: str, cause: BaseException | None) -> NoReturn:
    raise WorkerDeliveryError(refusal(
        RefusalClass.CAPACITY, text, guard=GUARD, waivable=False,
        troubleshooting=TROUBLESHOOTING)) from cause


def refuse_undelivered(handle: Any, failed: list[str], waited_s: float,
                       deadline_s: float) -> NoReturn:
    """End the price wait: a message to this fleet's actors was undeliverable."""
    _refuse(
        "this render needs every rank to price the load before any rank "
        f"allocates, and Monarch reported a message to this fleet's Worker actor "
        f"on {_labels(handle, failed)} undeliverable, so the price wait ended "
        f"{waited_s:.1f} s into its {deadline_s:.0f} s deadline. That happens when "
        "the Worker loop at that address restarted under this attached mesh "
        "(`dgxm restart`, a crash that systemd restarted, or a reboot): its actors "
        "are gone. Nothing was loaded on any rank. A sampler render that owns its "
        "session, and the UNET Loader's eager load, read every loop's id next and, "
        "when every loop restarted, attach again by the new ids and carry on. This "
        "refusal reaching you means that step did not run here (a pipelined "
        "render, an identity-gate leg or the Identity Gate node), or that the one "
        "dispatch on the new mesh met an undeliverable actor again. To recover, "
        f"{_ONE_SAMPLER_RENDER}.", None)


def _refuse_moves(handle: Any, moves: Moves, refused: BaseException) -> NoReturn:
    head = ("a message to this fleet's Worker actors was undeliverable, so the "
            "capacity price wait ended early, and")
    if moves.verdict == UNKNOWN:
        _refuse(
            f"{head} the id reading found none for {'; '.join(moves.unknown)}. "
            "Without it the driver cannot tell whether the Worker loops restarted, "
            "so it attached nothing and nothing was loaded. Queue again once every "
            f"loop publishes its id; if it repeats, {_RESTART_TOGETHER}.", refused)
    if moves.verdict == PARTIAL:
        _refuse(
            f"{head} the Worker loops at {_labels(handle, moves.moved)} serve new "
            f"proc ids while {_labels(handle, moves.kept)} still serves the id this "
            "mesh dialed. The loops that did not restart still hold this mesh's "
            "actors and their memory, so this render did not attach again and "
            f"nothing was loaded. To recover, {_RESTART_TOGETHER}: the next render "
            "attaches by the new ids.", refused)
    _refuse(
        f"{head} every Worker loop still serves the id this mesh dialed, so no "
        "loop restarted: the actor at that address is gone or unreachable for "
        "another reason. Nothing was loaded. Click Reset attached mesh in the DGX "
        "Monarch panel, then queue again; if the reset cannot stop the fleet, run "
        "`dgxm restart`.", refused)


def _refuse_retire(handle: Any, moves: Moves, cause: BaseException | None) -> NoReturn:
    from . import mesh_teardown

    evidence = getattr(handle, "_liveness_evidence", None)
    fold = ",".join(f"{address}={answer}"
                    for address, answer in getattr(evidence, "per_host", ()) or ())
    seen = (f"the liveness check answered {evidence.answer} ({fold})"
            if evidence is not None else "its stop has no confirmed outcome")
    _refuse(
        f"every Worker loop serves a new proc id ({_labels(handle, moves.moved)}), "
        f"but this driver could not retire the stale mesh: {seen}, so no "
        "replacement was attached beside processes that may still hold memory. "
        f"Nothing was loaded. {mesh_teardown.block_cause_sentence(handle)} "
        "`dgxm restart` also sweeps survivors (docs/TROUBLESHOOTING.md #51).", cause)


def _retire(handle: Any, moves: Moves, refused: BaseException) -> str:
    """Retire the stale fleet through the deliberate teardown; return how it ended."""
    from . import mesh_helpers, mesh_liveness
    from .mesh_safety import PriorTeardownUnknownError

    report = (f"worker restart: the Worker loops at {','.join(moves.moved)} serve "
              "new proc ids, and messages to this fleet's actors were undeliverable")
    # The publication a supervision failure makes: defunct, the death report a
    # liveness release requires, and the token that authorizes the stop below.
    token, deliberate, completed, deferred = (
        mesh_helpers.publish_supervision_failure_state(handle, refused, report))
    if not completed:
        if deliberate or deferred:
            _refuse_retire(handle, moves, refused)
        try:
            handle.shutdown(timeout_s=STOP_BOUND_S, _reconcile_token=token)
        except PriorTeardownUnknownError as exc:
            _refuse_retire(handle, moves, exc)
        except Exception as exc:
            # Expected: nothing at the old ids acknowledges the stop. The latch
            # it leaves names the after-death cause, which the proof below reads.
            log.info("restart_reattach: the stale mesh's stop did not confirm (%s); "
                     "checking that its processes are gone", type(exc).__name__)
    if getattr(handle, "teardown_complete", False):
        return "stop_confirmed"
    if mesh_liveness.try_release(handle):
        return "liveness_release"
    _refuse_retire(handle, moves, refused)


def _record(outcome: str, moves: Moves, started: float, *, seq: int = 0,
            teardown: str = "none", reason: str = "") -> float:
    """The trace line, the operator's log line and the telemetry event."""
    from .telemetry import emit

    wall = time.monotonic() - started
    log.info("%s", attach_trace.line(
        EVENT, seq=seq, outcome=outcome, verdict=moves.verdict, moved=list(moves.moved),
        kept=list(moves.kept), unknown=len(moves.unknown), teardown=teardown,
        reason=reason or "none", wall_s=wall))
    hosts = ",".join(moves.moved) or "none"
    if outcome == "attached":
        log.warning(
            "%s outcome=attached: every Worker loop serves a new proc id (%s); this "
            "render retired the stale mesh (%s) and attached again by the new ids in "
            "%.1f s, with no reset (docs/TROUBLESHOOTING.md #2)", EVENT, hosts, teardown, wall)
    else:
        log.warning(
            "%s outcome=%s reason=%s: this render did not attach again (loops with a "
            "new proc id: %s); its error names the remedy (docs/TROUBLESHOOTING.md #2)",
            EVENT, outcome, reason, hosts)
    emit(EVENT, outcome=outcome, verdict=moves.verdict, moved=hosts, teardown=teardown,
         wall_s=round(wall, 2))
    return wall


def heal(handle: Any, refused: BaseException, *,
         respawn: Callable[[Any], Any] | None = None) -> Reattached:
    """Retire a fleet whose every Worker loop restarted and attach a new one, or refuse.

    ``refused`` is the ``WorkerDeliveryError`` the price wait raised. Returns
    the new mesh for the caller's one retry, or raises a refusal that names what
    the check found. ``respawn`` attaches the new mesh (``mesh.ensure_live`` by
    default; the loader prices first). An attach failure propagates as raised.
    """
    from .mesh import ensure_live

    started = time.monotonic()
    moves = read_moves(getattr(handle, "config", None))
    if moves.verdict != ALL:
        _record("refused", moves, started, reason=moves.verdict)
        _refuse_moves(handle, moves, refused)
    try:
        teardown = _retire(handle, moves, refused)
    except WorkerDeliveryError:
        _record("refused", moves, started, reason="not_retired")
        raise
    try:
        fresh = (respawn or ensure_live)(handle)
    except BaseException as exc:
        _record("failed", moves, started, seq=attach_trace.last_sequence(),
                teardown=teardown, reason=type(exc).__name__)
        raise
    wall = _record("attached", moves, started, seq=attach_trace.last_sequence(),
                   teardown=teardown)
    return Reattached(fresh, wall, moves.moved, teardown)
