"""In-process reaper for each worker actor.

A daemon thread ends the process on an expired client lease or sustained
parent loss. Lease expiry is armed only by a positive client renewal. Parent
loss is armed when the watcher binds, including before the first lease, but a
held lease outranks it. A non-positive renewal disables both triggers.

Grace values keep self-exit after teardown token authority and Monarch's own
reapers. A wedged dispatch loop stops answering the plain renewal endpoint and
therefore exits even while the client remains live. This stdlib-only leaf loads
before actor, ComfyUI, or CUDA imports. `os._exit` does not reap an actor-created
forkserver subtree; those children remain outside this mechanism.
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Shared with `client_lease` so the grace arithmetic cannot drift.
RENEW_INTERVAL_S = 30.0
# Ten missed renewals; 300 - 30 exceeds 240 s token authority and peer cleanup.
REAP_GRACE_S = 300.0
# Parent loss is conclusive, but this delay keeps Monarch's reapers first.
PARENT_LOSS_GRACE_S = 150.0
TICK_S = 5.0
# The last renewal may precede a teardown token by one interval. Preserve
# `lease - RENEW_INTERVAL_S > TOKEN_AUTHORITY_S` (240 s); configuration may
# lengthen this grace but must not shorten it below the proved ladder.
MIN_LEASE_S = REAP_GRACE_S
MAX_LEASE_S = 3600.0
REAP_EXIT_CODE = 66
REAP_MARKER = "dgxm-reap:"
LEASE_EXPIRY = "lease-expiry"
PARENT_LOSS = "parent-loss"


@dataclass(frozen=True)
class Verdict:
    """Why this process must exit now."""

    trigger: str
    detail: str


class ActorLifetime:
    """One process's reap state. Pure and injectable so tests never fork."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        getppid: Callable[[], int] = os.getppid,
        getpid: Callable[[], int] = os.getpid,
    ) -> None:
        self._clock = clock
        self._getppid = getppid
        self._getpid = getpid
        self._lock = threading.Lock()
        self._parent_pid: int | None = None
        self._parent_lost_at: float | None = None
        self._deadline: float | None = None
        self._last_renew: float | None = None
        self._armed = False
        self._disabled = False
        self._fired = False

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def disabled(self) -> bool:
        return self._disabled

    @property
    def deadline(self) -> float | None:
        return self._deadline

    @property
    def fired(self) -> bool:
        return self._fired

    def bind(self, ppid: int | None = None) -> int:
        """Record the spawning process now, not at the first tick.

        Delaying the baseline could mistake an early reparent for the parent.
        """
        with self._lock:
            if self._parent_pid is None:
                self._parent_pid = self._getppid() if ppid is None else int(ppid)
            return self._parent_pid

    def renew(self, lease_s: float, *, now: float | None = None) -> None:
        """Extend (or, at ``lease_s <= 0``, cancel) this process's lease.

        Invalid or non-positive values disable both triggers until a positive
        lease arrives. Positive values are clamped to the safe range.
        """
        moment = self._clock() if now is None else float(now)
        try:
            lease = float(lease_s)
        except (TypeError, ValueError):
            lease = 0.0
        with self._lock:
            if self._fired:
                return
            if not math.isfinite(lease) or lease <= 0.0:
                self._disabled = True
                self._armed = False
                self._deadline = None
                return
            lease = min(max(lease, MIN_LEASE_S), MAX_LEASE_S)
            self._disabled = False
            self._armed = True
            self._last_renew = moment
            self._deadline = moment + lease

    def observe(
        self, *, now: float | None = None, ppid: int | None = None
    ) -> Verdict | None:
        """One tick. Returns the verdict to act on, or None to keep living."""
        moment = self._clock() if now is None else float(now)
        parent = self._getppid() if ppid is None else int(ppid)
        with self._lock:
            if self._fired:
                return None
            if self._parent_pid is None:
                self._parent_pid = parent
            if self._disabled:
                return None
            # Reparenting is monotonic, so latch its first observation.
            if parent != self._parent_pid and self._parent_lost_at is None:
                self._parent_lost_at = moment
            lease_held = (
                self._armed
                and self._deadline is not None
                and moment < self._deadline
            )
            lost_at = self._parent_lost_at
            if (lost_at is not None
                    and moment - lost_at >= PARENT_LOSS_GRACE_S
                    and not lease_held):
                return Verdict(
                    PARENT_LOSS,
                    f"spawning process {self._parent_pid} gone for "
                    f"{moment - lost_at:.0f}s",
                )
            if self._armed and self._deadline is not None and moment >= self._deadline:
                silence = moment - (self._last_renew if self._last_renew is not None else moment)
                return Verdict(LEASE_EXPIRY, f"no client lease for {silence:.0f}s")
            return None

    def fire(
        self,
        verdict: Verdict,
        *,
        sink: Any | None = None,
        exit_hook: Callable[[int], None] | None = None,
    ) -> bool:
        """Announce and exit once without raising."""
        with self._lock:
            if self._fired:
                return False
            self._fired = True
        stream = sys.stderr if sink is None else sink
        leave = os._exit if exit_hook is None else exit_hook
        try:
            stream.write(
                f"{REAP_MARKER} actor pid={self._getpid()} exiting, "
                f"{verdict.trigger}, {verdict.detail}\n")
            stream.flush()
        except BaseException:
            # The inherited stream may belong to the dead parent.
            pass
        finally:
            # Bypass a possibly wedged runtime and CUDA/atexit teardown.
            leave(REAP_EXIT_CODE)
        return True


_LIFETIME = ActorLifetime()
_WATCHER: Any | None = None
_WATCHER_STARTING = False
_WATCH_LOCK = threading.Lock()


def lifetime() -> ActorLifetime:
    """The per-process reap state."""
    return _LIFETIME


def renew(lease_s: float) -> None:
    """Endpoint entry point: hold this process off for ``lease_s`` seconds."""
    _LIFETIME.renew(lease_s)


def _watch(
    state: ActorLifetime, tick_s: float, sleeper: Callable[[float], None]
) -> None:
    while True:
        sleeper(tick_s)
        try:
            verdict = state.observe()
        except BaseException:
            continue  # a procfs or clock error must never end the watch
        if verdict is not None:
            state.fire(verdict)
            return


def install(
    *,
    tick_s: float = TICK_S,
    sleeper: Callable[[float], None] = time.sleep,
    thread_factory: Any = threading.Thread,
    state: ActorLifetime | None = None,
) -> bool:
    """Start the watcher thread once per process. Returns True when it started.

    It must be a daemon outside the actor dispatch loop so the monitored wedge
    cannot block the watcher or keep a dying process alive.
    """
    global _WATCHER, _WATCHER_STARTING
    watched = _LIFETIME if state is None else state
    with _WATCH_LOCK:
        if _WATCHER is not None:
            alive = _thread_liveness(_WATCHER)
            if _WATCHER_STARTING:
                if alive is True:
                    _WATCHER_STARTING = False
                    return False
                raise RuntimeError(
                    "a prior actor-lifetime watcher start has an unresolved "
                    "outcome; restart the actor process before starting another")
            if alive is None:
                raise RuntimeError(
                    "the existing actor-lifetime watcher's liveness could not be "
                    "confirmed; restart the actor process before starting another")
            if alive:
                return False
        watched.bind()
        thread = thread_factory(
            target=_watch,
            args=(watched, tick_s, sleeper),
            name="dgxm-actor-lifetime",
            daemon=True,
        )
        _WATCHER = thread
        _WATCHER_STARTING = True
        try:
            thread.start()
            _WATCHER_STARTING = False
        except BaseException as exc:
            alive = _thread_liveness(thread)
            if alive is True:
                _WATCHER_STARTING = False
            elif isinstance(exc, Exception) and alive is False:
                _WATCHER = None
                _WATCHER_STARTING = False
            raise
        return True


def _thread_liveness(thread: Any) -> bool | None:
    try:
        return bool(thread.is_alive())
    except Exception:
        return None
