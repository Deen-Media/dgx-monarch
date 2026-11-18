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

from .log import get_logger

_F = TypeVar("_F", bound=Callable[..., Any])
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
