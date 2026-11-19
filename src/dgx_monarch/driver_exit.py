"""Whether the driver's interpreter has begun to exit, so lost-fleet work stands down.

At a normal interpreter exit (ComfyUI stopped with Ctrl+C, or a script that
returns) Monarch's own exit step stops every mesh the driver owns. A call still
in flight then fails, and without this flag the eviction, the reconcile and the
fault hook would read that as a lost Worker: try a stop against the closing
client, fail, and log that replacement is blocked and ComfyUI must restart.

``install`` registers the flag's exit hook once, when the first mesh is built
(mesh_factory). Monarch registers its own hook when its client starts, which a
mesh build needs first, and exit hooks run last registered first, so this one
sets the flag before Monarch stops anything. It stops nothing itself, and it
logs the one INFO line for the whole exit from the main thread. From then on
the functions wrapped with ``skipped_at_exit`` do nothing, and a reconcile that
was already inside its stop when the exit began logs none of its own lines: the
mesh logger drops every record from the reconcile's thread (``RECONCILE_THREAD``)
once the flag is set, so its stop, cut short by Monarch's exit step, reports no
failure that the exit itself caused. The stop is not changed. A render's own
thread is quiet the same way while it runs a stop of its own inside
``quiet_at_exit`` (the restart heal and the pinned retire in restart_reattach):
the loggers that path writes through (``QUIET_LOGGERS``) drop its records once
the flag is set. So no daemon thread of the driver's writes to stderr while the
interpreter shuts down.

Nothing new starts on the Workers once the flag is set. A render, load or node
that starts then (ComfyUI's prompt worker takes every queued prompt until the
interpreter stops it) waits at its node's wrapper (``nodes.refusal_lead``) or at
the top of ``mesh.get_mesh`` and ``mesh.ensure_live`` (``hold_at_entry``) until
the process ends; only the first
to reach ``HOLD_AT_EXIT_S`` raises. Past the entry, each new send (a price, a setup,
a dispatch, an all-rank call, an RDMA read, a reset, the stop ``mesh.get_mesh``
sends for a config change) raises ``DriverExitingError`` (``refuse_new_call``),
and the best-effort ones (the cancel, an RDMA drop, the lease renewal, a setup's
rollback) skip themselves. ``MeshHandle.shutdown`` has no check of its own: a
stop that a teardown already under way sends still runs.
Once the exit has begun, a pack log line that carries that error, as an object
or as text, is not printed: the pack's loggers drop a record whose arguments or
exception carry it (itself or as an explicit cause), or whose message names it
(``TEXT_MARKS``: its class name, or the sentence every exit refusal opens with;
``log.SilentInLogs``). Where more than a line is at stake, the route checks the
exit itself: the native-return latch (``native_return_latch``), the identity
gate's stock quarantine and its stock reference leg (``nodes.gate_identity``,
``nodes.gate_cross_mode``), and the Worker-assisted decode, which then decodes
here. So does a line about a call Monarch's exit step cut, whose error names
no marker: the identity gate's abort lines (``nodes.gate_ceremony``,
``nodes.gate_abort``) and the RDMA poison report (``transfer``). What runs to
its end: a call in flight, a stop under way, the cleanup of a fleet bring-up
under way, and the steps of an attach after its dial.

The hook sets the flag only once the main thread has ended. Python marks it
ended before it runs exit hooks (checked on 3.11, 3.12 and 3.13), so a call
while ComfyUI still runs, which would leave every later lost Worker unevicted,
sets nothing.
"""
from __future__ import annotations

import atexit
import functools
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from .log import SilentInLogs, get_logger

_F = TypeVar("_F", bound=Callable[..., Any])
_T = TypeVar("_T")
_log = get_logger("dgx_monarch.mesh")
_EXITING = threading.Event()
_LOCK = threading.Lock()
_STATE = {"installed": False}
# The supervision reconcile's thread (mesh_helpers starts it under this name).
RECONCILE_THREAD = "dgxm-reconcile-stop"
NOTICE = ("The driver is exiting: Monarch's exit step stops the Worker actors, "
          "and the driver reports no Worker faults from here on")
# Threads inside quiet_at_exit, by ident.
_QUIET_THREADS: set[int] = set()
# Every logger the reconcile and a render's own stop write through.
QUIET_LOGGERS = ("dgx_monarch.mesh", "dgx_monarch.mesh_liveness", "dgx_monarch.restart_reattach")


def _quiet_reconcile(record: Any) -> bool:
    """The filter: once exit has begun, the reconcile's thread, and a thread
    inside ``quiet_at_exit``, log nothing."""
    if not _EXITING.is_set():
        return True
    return not (record.threadName == RECONCILE_THREAD or record.thread in _QUIET_THREADS)


for _name in QUIET_LOGGERS:
    get_logger(_name).addFilter(_quiet_reconcile)


@contextmanager
def quiet_at_exit() -> Iterator[None]:
    """Mark this thread quiet at exit while it runs a stop of its own; usable as a decorator.

    Nothing is dropped before exit begins, and only this thread's records then.
    """
    ident = threading.get_ident()
    with _LOCK:
        entered = ident not in _QUIET_THREADS
        _QUIET_THREADS.add(ident)
    try:
        yield
    finally:
        if entered:
            with _LOCK:
                _QUIET_THREADS.discard(ident)


def _main_thread_alive() -> bool:
    return threading.main_thread().is_alive()


def _begin() -> None:
    """The exit hook: set the flag and log the notice, once the main thread has ended."""
    if _main_thread_alive() or _EXITING.is_set():
        return
    _EXITING.set()
    try:
        _log.info(NOTICE)
    except Exception:
        pass


def install() -> None:
    """Register the exit hook once per process; never raises."""
    try:
        with _LOCK:
            if not _STATE["installed"]:
                atexit.register(_begin)
                _STATE["installed"] = True
    except Exception:
        pass


def exiting() -> bool:
    """Whether interpreter exit has begun (the hook ran)."""
    return _EXITING.is_set()


class DriverExitingError(RuntimeError, SilentInLogs):
    """A new call to the Worker actors that the driver did not send because it is exiting.

    ``SilentInLogs``: once the exit has begun, the pack's loggers drop a record
    that carries this error as an object (an argument or the record's exception,
    itself or as an explicit cause) or names it in its message (``TEXT_MARKS``),
    so a handler that catches it and warns (an unload or a status poll that
    "failed") prints nothing, whether it logs the error or its text.
    """


# The sentence every refusal the exit raises opens with: the send's and the
# not-started one here, the attach's (mesh_attach) and the render's
# (restart_reattach). The notice opens differently ("The driver is exiting:").
REFUSAL_OPENING = "the driver is exiting, so"
# What a log message carries when it names the exit's refusal as text; once the
# exit has begun, the pack's loggers drop such a line (log.SilentInLogs).
TEXT_MARKS = (DriverExitingError.__name__, REFUSAL_OPENING)
SENT_NOTHING = (f"{REFUSAL_OPENING} it sent nothing new to the Worker actors: "
                "Monarch's exit step stops them")


def refuse_new_call() -> None:
    """Raise ``DriverExitingError`` once exit has begun: no new call reaches the Workers.

    The one check every new send makes (a price, a setup, a dispatch, an
    all-rank call). A call already in flight and a stop already under way run
    to their end; a best-effort send (the cancel, the lease renewal) reads
    ``exiting`` and skips itself instead, since it raises nothing to its caller.
    """
    if _EXITING.is_set():
        raise DriverExitingError(SENT_NOTHING)


def send_unless_exiting(send: Callable[[], _T]) -> _T:
    """``send()``, unless exit has begun: the check at the send itself.

    ``mesh_setup.dispatch_endpoint`` sends through this under the handle's lock,
    so a caller that passed its own check and then waited for that lock (or for
    the render session) sends nothing once the exit began while it waited.
    """
    refuse_new_call()
    return send()


# How long a render, load or node that starts once the driver is exiting waits at
# the entry for the process to end before it refuses. A process ends about 0.5 s
# after the notice on loopback; one still alive at the bound has an exit hook that
# hung, and the refusal is then the one error it prints: every entry after it waits
# for the process to end with no bound (HOLD_AFTER_REFUSAL_S, None outside tests).
HOLD_AT_EXIT_S = 30.0
HOLD_AFTER_REFUSAL_S: float | None = None
_PROCESS_ENDS = threading.Event()  # never set: the process ends first
_REFUSED_AT_BOUND = threading.Event()  # this exit's one refusal at the bound has been raised
NOT_STARTED = (f"{REFUSAL_OPENING} this render, load or node did not start: "
               "Monarch's exit step stops the Worker actors, and nothing was sent to them")


def hold_at_entry(handle: Any = None) -> None:
    """Hold a render, load or node that starts once exit has begun; refuse at the bound.

    At a thread's outermost node method under ``nodes.refusal_lead.lead_with_refusal``
    (the samplers, the UNET loaders, Fleet and the Identity Gate), before any of
    the node's own work, and at the top of ``mesh.get_mesh`` and ``mesh.ensure_live``
    for the nodes with no wrapper, before any lock, registry change, attach, id
    reading or price. ComfyUI's prompt worker takes
    every queued prompt until the interpreter stops it, and prints two ERROR
    records for each that fails, so a refusal here would print a traceback per
    queued prompt while the process closes. Held, the first such prompt waits,
    logs nothing and takes no lock, and the worker takes no other: the process
    ends, and the daemon thread with it. After ``HOLD_AT_EXIT_S`` the first
    entry to get there raises ``DriverExitingError``; any entry after that one
    waits for the process to end with no bound, so a queue behind an exit hook
    that hangs prints that one refusal however long the hook hangs. A thread that
    owns ``handle``'s lock, which a stop takes, refuses at once instead of holding
    (no caller enters so; checked here).
    """
    if not _EXITING.is_set():
        return
    lock = getattr(handle, "lock", None)
    owned = getattr(lock, "_is_owned", None)
    if callable(owned) and owned():
        raise DriverExitingError(NOT_STARTED)
    if not _REFUSED_AT_BOUND.is_set():
        _PROCESS_ENDS.wait(HOLD_AT_EXIT_S)
        with _LOCK:
            first = not _REFUSED_AT_BOUND.is_set()
            _REFUSED_AT_BOUND.set()
        if first:
            raise DriverExitingError(f"{NOT_STARTED} (it waited {HOLD_AT_EXIT_S:.0f} s for the process to end)")
    _PROCESS_ENDS.wait(HOLD_AFTER_REFUSAL_S)  # no bound outside tests: no second refusal
    raise DriverExitingError(NOT_STARTED)


def skipped_at_exit(fn: _F) -> _F:
    """``fn``, except that once exit has begun a call does nothing and returns None."""
    @functools.wraps(fn)
    def run(*args: Any, **kwargs: Any) -> Any:
        if _EXITING.is_set():
            return None
        return fn(*args, **kwargs)

    return run  # type: ignore[return-value]


def reset() -> None:
    """Forget an exit (tests); the registered hook stays registered."""
    _EXITING.clear()
    _REFUSED_AT_BOUND.clear()
