"""Attach again inside the render when every Worker loop restarted under it.

docs/TROUBLESHOOTING.md #2. A Worker loop that restarts under an attached mesh
takes the mesh's actors with it, and a message to one of them comes back to
the driver's fault hook as undeliverable, never to the call that sent it. The
render that meets the old actors used to wait the capacity guard's whole 30 s
deadline, refuse, and leave the reset to the operator. Now:

1. The capacity price wait ends within one poll of a delivery failure that
   names a fleet host and the mesh it prices on (``delivery_faults.await_rank``,
   ``watched_mesh``) and raises ``WorkerDeliveryError`` (``refuse_undelivered``).
   Monarch's supervision can give up on the old actors before any delivery
   failure is recorded (the restart runs traced its "stopped before report"
   about 2 s into the price), and the price then fails with a SupervisionError.
   The handle's own wait would evict the fleet and join its post-death stop
   before anything else ran. ``PriceWait`` waits plainly instead, and on the
   first SupervisionError ``refuse_if_restarted`` reads every loop's id once,
   with a short bound. When every loop serves a new id it raises the same
   error with that reading attached; otherwise the failure goes through the
   handle's own wait, and its eviction and reconcile run as before.
2. A sampler render that owns its render session, or the UNET Loader's eager
   load, calls ``heal``. It reads each loop's id with the listener-generation
   probe every attach already runs, unless the error carries that reading
   already, and compares it with the id this driver dialed
   (``mesh_attach.last_dialed``).
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
5. A KSampler Pipeline or Identity Gate render is pinned to its mesh: ``refuse_pinned``
   retires the stale fleet as step 3 does and refuses when every loop restarted and
   nothing else holds the fleet; otherwise it leaves the fleet as it always did.

Bounds: ``GENERATION_PROBE_S`` per host for the reading (``PRECHECK_PROBE_S``
for the one taken before an eviction), two ``STOP_BOUND_S``
waits for the stop, the liveness fleet budget, and the attach's own budgets.
The replacement attach reuses the one transport this process bound. Its
channel to a loop's address keeps reconnecting for a day
(``monarch_policy._FIXED_CHANNEL``); after a longer outage that channel stays
closed, the attach fails at its config-push budget, and the session asks for
the documented driver relaunch.

Every detour writes one ``attach_trace event=restart_reattach`` line, one
``restart_reattach outcome=...`` log line and one telemetry event.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NoReturn

from . import attach_trace, box_names, delivery_faults, driver_exit, worker_proc_id
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
# How long the reading taken before an eviction may wait on each host. On
# loopback a reading of two loops took 0.07 to 0.13 s (the pair is not
# measured); a host that does not answer is read as unknown here, and its
# eviction then runs as it always did.
PRECHECK_PROBE_S = 3.0
# The trace event for that reading.
CHECK_EVENT = "restart_check"

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
    unknown: tuple[str, ...] = ()  # "<plain name> (<reason>)", box_names.py

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


def read_moves(config: Any, *, timeout: float | None = None) -> Moves:
    """Read every loop's id now and compare it with the id this driver dialed.

    ``timeout`` bounds each host's reading; the attach's own bound by default.
    """
    from . import mesh_attach
    from .cli import lifecycle, listener_generation

    hosts = tuple(getattr(config, "hosts", None) or ())
    words = box_names.address_words(config)
    try:
        readings = listener_generation.fleet(
            config, runner=lifecycle.run_on_host,
            timeout=mesh_attach.GENERATION_PROBE_S if timeout is None else timeout)
    except Exception as exc:
        return Moves(unknown=tuple(
            f"{words.get(str(host.address), box_names.UNLISTED)} (the probe raised "
            f"{type(exc).__name__})"
            for host in hosts))
    moved: list[str] = []
    kept: list[str] = []
    unknown: list[str] = []
    for host in hosts:
        fields = readings.get(host.address) or {}
        value = fields.get(worker_proc_id.FIELD)
        dialed = mesh_attach.last_dialed(host.address)
        word = words.get(str(host.address), box_names.UNLISTED)
        if not worker_proc_id.is_value(value):
            unknown.append(f"{word} ({worker_proc_id.problem(fields)[0]})")
        elif dialed is None:
            unknown.append(f"{word} (this driver holds no dialed id for it)")
        else:
            (moved if value != dialed else kept).append(host.address)
    return Moves(tuple(moved), tuple(kept), tuple(unknown))


def _labels(handle: Any, addresses: Any) -> str:
    """The boxes at these Worker loop addresses in plain words (box_names.py)."""
    words = box_names.address_words(getattr(handle, "config", None))
    return box_names.join(words.get(str(address), box_names.UNLISTED) for address in addresses)


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
        "the Worker loop on that box restarted under this attached mesh "
        "(`dgxm restart`, a crash that systemd restarted, or a reboot): its actors "
        "are gone. Nothing was loaded on any rank. A sampler render that owns its "
        "session, and the UNET Loader's eager load, read every loop's id next and, "
        "when every loop restarted, attach again by the new ids and carry on; a "
        "pipelined render, an identity-gate leg or the Identity Gate node then "
        "retires the stale mesh and asks for a re-queue. This refusal reaching you "
        "means neither happened here (an earlier render's result was still in "
        "flight, or not every loop restarted, for example), or that the one dispatch "
        f"on the new mesh met an undeliverable actor again. To recover, {_ONE_SAMPLER_RENDER}.",
        None)


def _supervised(exc: BaseException) -> bool:
    """Whether ``exc``, or an exception chained behind it, is Monarch's SupervisionError."""
    from .error_utils import exception_chain

    try:
        from monarch._rust_bindings.monarch_hyperactor.supervision import SupervisionError
    except ImportError:
        return False
    return any(isinstance(link, SupervisionError) for link in exception_chain(exc))


def refuse_if_restarted(handle: Any, failure: BaseException, waited_s: float,
                        deadline_s: float) -> Moves:
    """End the price on a SupervisionError when every loop restarted; else return the reading.

    Reads every loop's id once, with ``PRECHECK_PROBE_S`` per host. When each
    serves a new id, raises the ``WorkerDeliveryError`` that a sampler render
    or the eager load catches to attach again, carrying the reading for
    ``heal``. Writes one ``attach_trace event=restart_check`` line either way,
    except at exit: once the driver is exiting it reads nothing and raises the
    exit refusal, as it does for any reading the exit overtook.
    """
    started = time.monotonic()
    _stand_down_at_exit(failure)  # no id reading over ssh once the driver is exiting
    moves = read_moves(getattr(handle, "config", None), timeout=PRECHECK_PROBE_S)
    _stand_down_at_exit(failure)  # a reading the exit overtook: the exit's refusal, not the price's
    log.info("%s", attach_trace.line(
        CHECK_EVENT, seq=0, verdict=moves.verdict, moved=list(moves.moved),
        kept=list(moves.kept), unknown=len(moves.unknown), waited_s=waited_s,
        wall_s=time.monotonic() - started))
    if moves.verdict != ALL:
        return moves
    raise WorkerDeliveryError(refusal(
        RefusalClass.CAPACITY,
        "this render needs every rank to price the load before any rank "
        "allocates, and Monarch reported this fleet's Worker actors failed "
        f"(SupervisionError) {waited_s:.1f} s into the price's {deadline_s:.0f} s "
        f"deadline, while every Worker loop serves a new proc id (on "
        f"{_labels(handle, moves.moved)}): the loops restarted under this attached "
        "mesh (`dgxm restart`, a crash that systemd restarted, or a reboot) and its "
        "actors are gone. Nothing was loaded on any rank. A sampler render that "
        "owns its session, and the UNET Loader's eager load, attach again by the "
        "new ids and carry on. This refusal reaching you means that step did not "
        "run here (a pipelined render, an identity-gate leg or the Identity Gate "
        "node), or that the one dispatch on the new mesh failed again. To "
        "recover, queue this workflow again: a pipelined or identity-gate render "
        "handed this fleet to the driver's failure cleanup before this refusal "
        "reached you, so the next render attaches a fresh one once that cleanup "
        "has stopped it, or names what blocks it, and a sampler render attaches "
        f"again by itself. If the next render refuses the same way, {_ONE_SAMPLER_RENDER}.",
        guard=GUARD, waivable=False, troubleshooting=TROUBLESHOOTING),
        moves=moves) from failure


def heals_unevicted(exc: BaseException) -> bool:
    """Whether ``exc`` is the restart check's error with every loop restarted.

    A submit guard's failure hook leaves a first attempt's to ``heal``, or in a
    pinned session to ``refuse_pinned``, as it leaves a delivery failure: they stop
    the old fleet with their own short bounds, where an eviction first would
    stop it with the post-death bound and wait. Every other error, a
    SupervisionError that is not a full restart among them, goes to the
    eviction as before.
    """
    moves = getattr(exc, "moves", None)
    return (isinstance(exc, WorkerDeliveryError) and isinstance(moves, Moves)
            and moves.verdict == ALL)


class PriceWait:
    """One wait step of a cluster fleet's capacity price, read before any eviction.

    ``MeshHandle._await_or_evict`` evicts the fleet on a supervision failure
    and joins its post-death stop before it raises, which can outlast the
    price's deadline. This wait reads the future plainly. On the first
    SupervisionError it asks ``refuse_if_restarted`` whether every loop
    restarted; that raises the heal's error when they did. Every other failure,
    and every later one, goes through the handle's own wait on the same
    future, which reads the outcome again at once and evicts exactly as it
    always did. A step that only times out evicts nothing there either.
    """

    def __init__(self, handle: Any, started: float, deadline_s: float) -> None:
        self.handle, self.started, self.deadline_s = handle, started, deadline_s
        self.checked = False

    def __call__(self, future: Any, step: float) -> Any:
        from .mesh_helpers import get_off_loop

        try:
            return get_off_loop(future, step)
        except TimeoutError:
            raise
        except Exception as exc:
            if not self.checked and _supervised(exc):
                self.checked = True
                refuse_if_restarted(self.handle, exc, time.monotonic() - self.started,
                                    self.deadline_s)
        return self.handle._await_or_evict(future, 0.0)


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
            f"{head} the Worker loops on {_labels(handle, moves.moved)} serve new "
            f"proc ids while those on {_labels(handle, moves.kept)} still serve the id "
            "this mesh dialed. The loops that did not restart still hold this mesh's "
            "actors and their memory, so this render did not attach again and "
            f"nothing was loaded. To recover, {_RESTART_TOGETHER}: the next render "
            "attaches by the new ids.", refused)
    _refuse(
        f"{head} every Worker loop still serves the id this mesh dialed, so no "
        "loop restarted: the actor on that box is gone or unreachable for "
        "another reason. Nothing was loaded. Click Reset attached mesh in the DGX "
        "Monarch panel, then queue again; if the reset cannot stop the fleet, run "
        "`dgxm restart`.", refused)


def _refuse_retire(handle: Any, moves: Moves, cause: BaseException | None) -> NoReturn:
    from . import mesh_teardown

    evidence = getattr(handle, "_liveness_evidence", None)
    words = box_names.address_words(getattr(handle, "config", None))
    fold = ", ".join(f"{words.get(str(address), box_names.UNLISTED)}: {answer}"
                     for address, answer in getattr(evidence, "per_host", ()) or ())
    seen = (f"the liveness check answered {evidence.answer} ({fold})"
            if evidence is not None else "its stop has no confirmed outcome")
    _refuse(
        f"every Worker loop serves a new proc id (on {_labels(handle, moves.moved)}), "
        f"but this driver could not retire the stale mesh: {seen}, so no "
        "replacement was attached beside processes that may still hold memory. "
        f"Nothing was loaded. {mesh_teardown.block_cause_sentence(handle)} "
        "`dgxm restart` also sweeps survivors (docs/TROUBLESHOOTING.md #51).", cause)


def _retire(handle: Any, moves: Moves, refused: BaseException) -> str:
    """Retire the stale fleet through the deliberate teardown; return how it ended."""
    from . import mesh_helpers, mesh_liveness
    from .mesh_safety import PriorTeardownUnknownError

    report = (f"worker restart: the Worker loops on {_labels(handle, moves.moved)} serve "
              "new proc ids, and messages to this fleet's actors were undeliverable")
    # The publication a supervision failure makes: defunct, the death report a
    # liveness release requires, and the token that authorizes the stop below.
    token, deliberate, completed, deferred = (
        mesh_helpers.publish_supervision_failure_state(handle, refused, report))
    if not completed:
        if deliberate or deferred:
            _refuse_retire(handle, moves, refused)
        _stand_down_at_exit(refused)
        try:
            handle.shutdown(timeout_s=STOP_BOUND_S, _reconcile_token=token)
        except PriorTeardownUnknownError as exc:
            _stand_down_at_exit(refused)
            _refuse_retire(handle, moves, exc)
        except Exception as exc:
            _stand_down_at_exit(refused)
            # Expected: nothing at the old ids acknowledges the stop. The latch
            # it leaves names the after-death cause, which the proof below reads.
            log.info("restart_reattach: the stale mesh's stop did not confirm (%s); "
                     "checking that its processes are gone", type(exc).__name__)
    _stand_down_at_exit(refused)  # no liveness probe over ssh, no attach
    if getattr(handle, "teardown_complete", False):
        return "stop_confirmed"
    if mesh_liveness.try_release(handle):
        return "liveness_release"
    _stand_down_at_exit(refused)  # a probe the exit overtook: the exit's refusal, not its answer's
    _refuse_retire(handle, moves, refused)


def _stand_down_at_exit(refused: BaseException) -> None:
    """Once the driver is exiting, end the detour: Monarch's exit step stops every mesh."""
    if driver_exit.exiting():
        _refuse("the driver is exiting, so this render stops here: Monarch's exit step "
                "stops the Worker actors, and nothing was loaded.", refused)


def _record(outcome: str, moves: Moves, started: float, *, seq: int = 0,
            teardown: str = "none", reason: str = "") -> float:
    """The trace line, the operator's log line and the telemetry event; none at exit."""
    from .telemetry import emit

    wall = time.monotonic() - started
    if driver_exit.exiting():
        return wall
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


@driver_exit.quiet_at_exit()
def heal(handle: Any, refused: BaseException, *,
         respawn: Callable[[Any], Any] | None = None) -> Reattached:
    """Retire a fleet whose every Worker loop restarted and attach a new one, or refuse.

    ``refused`` is the ``WorkerDeliveryError`` the price wait raised. Returns
    the new mesh for the caller's one retry, or raises a refusal that names what
    the check found. ``respawn`` attaches the new mesh (``mesh.ensure_live`` by
    default; the loader prices first). An attach failure propagates as raised.
    Once the driver is exiting, its thread logs nothing (driver_exit) and the
    detour stands down at the next step: no id reading, stop, probe or attach
    starts, and no dispatch follows on a mesh the exit overtook.
    """
    from .mesh import ensure_live

    started = time.monotonic()
    # refuse_if_restarted read every loop's id moments ago; read again otherwise.
    moves = getattr(refused, "moves", None)
    if not isinstance(moves, Moves):
        _stand_down_at_exit(refused)  # no id reading over ssh once the driver is exiting
        moves = read_moves(getattr(handle, "config", None))
    _stand_down_at_exit(refused)  # a reading the exit overtook: the exit's refusal, not its remedy
    if moves.verdict != ALL:
        _record("refused", moves, started, reason=moves.verdict)
        _refuse_moves(handle, moves, refused)
    try:
        teardown = _retire(handle, moves, refused)
    except WorkerDeliveryError:
        _record("refused", moves, started, reason="not_retired")
        raise
    _stand_down_at_exit(refused)
    try:
        fresh = (respawn or ensure_live)(handle)
    except BaseException as exc:
        _stand_down_at_exit(exc)  # an attach the exit cut short ends in the exit's refusal
        _record("failed", moves, started, seq=attach_trace.last_sequence(),
                teardown=teardown, reason=type(exc).__name__)
        raise
    _stand_down_at_exit(refused)  # an attach the exit overtook: no dispatch on the new mesh
    wall = _record("attached", moves, started, seq=attach_trace.last_sequence(),
                   teardown=teardown)
    return Reattached(fresh, wall, moves.moved, teardown)


def retry_after_heal(retry: Callable[[], Any], refused: BaseException) -> Any:
    """Run the caller's one retry on the mesh ``heal`` attached, unless the driver began
    to exit since ``heal`` returned: then no price and no dispatch, and the exit refusal.
    A sampler render's re-dispatch and the eager load's second load both come here."""
    _stand_down_at_exit(refused)  # an exit after heal returned: nothing reaches the new mesh
    return retry()


from .restart_pinned import refuse_pinned as refuse_pinned  # noqa: E402  # re-exported for render_submit
