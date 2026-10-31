"""Two-ended unit scheduling for Worker-assisted VAE decode (stdlib leaf).

ComfyUI's own decode loop runs once on the driver and asks for its units in
its own order. The driver takes units from the front: a unit no Worker rank
has claimed is decoded here, by the stock function, at the moment the loop
reaches it. Worker ranks claim units from the back, one or two at a time, so
their results are ready by the time the loop arrives. A rank that fails,
declines or times out stops claiming, and every unit it held is decoded here
instead, which is the reference decode for that unit. Nothing in this module
can change an output: it only decides who decodes which unit, and every
result the loop accepts was produced for that exact unit.

The board knows no tensors, Monarch or ComfyUI, so CPU tests drive it with
plain callables (tests/test_vae_decode_schedule.py).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

UNCLAIMED, LOCAL, CLAIMED, DONE, FAILED = "unclaimed", "local", "claimed", "done", "failed"
# The loop polls this often while it waits on a rank, so an interrupt or a
# rank that dies silently is noticed within a fraction of a second.
_POLL_S = 0.25


@dataclass
class RankFailure:
    """Why one rank stopped taking units: the first cause wins."""

    rank: int
    cause: str


@dataclass
class BoardStats:
    """What the decode log line and telemetry report about one board."""

    units: int = 0
    local: int = 0
    remote: dict[int, int] = field(default_factory=dict)
    fallback: int = 0
    failures: list[RankFailure] = field(default_factory=list)


class UnitBoard:
    """Who decodes each unit of one decode call."""

    def __init__(self, count: int, *, lookahead: int = 1) -> None:
        if count < 0:
            raise ValueError("a board needs a non-negative unit count")
        self._cond = threading.Condition()
        self._state = [UNCLAIMED] * count
        self._results: dict[int, Any] = {}
        self._owner: dict[int, int] = {}
        self._front = 0  # the next unit the loop will reach
        self._back = count - 1  # the next unit a rank may claim
        self._lookahead = max(0, int(lookahead))
        self._closed = False
        self._retired: set[int] = set()
        self.stats = BoardStats(units=count)

    def claim(self, rank: int) -> int | None:
        """The next unit for ``rank``, from the back, or None when none is left.

        A rank never claims the unit the loop will reach next: that one stays
        here, so the driver's own GPU is never idle waiting for a transfer.
        """
        with self._cond:
            if self._closed or rank in self._retired:
                return None
            while self._back >= 0 and self._state[self._back] != UNCLAIMED:
                self._back -= 1
            if self._back < self._front + self._lookahead:
                return None
            unit = self._back
            self._back -= 1
            self._state[unit] = CLAIMED
            self._owner[unit] = rank
            return unit

    def deliver(self, unit: int, rank: int, result: Any) -> None:
        """Record one rank's result; a unit the loop already gave up on drops it."""
        with self._cond:
            if self._state[unit] == CLAIMED and self._owner.get(unit) == rank:
                self._state[unit] = DONE
                self._results[unit] = result
            self._cond.notify_all()

    def fail(self, rank: int, cause: str) -> None:
        """Retire ``rank``: every unit it still holds goes back to the driver."""
        with self._cond:
            self._retire_locked(rank, cause)
            self._cond.notify_all()

    def _retire_locked(self, rank: int, cause: str) -> None:
        if rank not in self._retired:
            self._retired.add(rank)
            self.stats.failures.append(RankFailure(rank, cause))
        for unit, owner in self._owner.items():
            if owner == rank and self._state[unit] == CLAIMED:
                self._state[unit] = FAILED

    def reject(self, unit: int, cause: str) -> None:
        """The loop refused a delivered result; count the unit as decoded here and retire its rank."""
        with self._cond:
            rank = self._owner.get(unit)
            if rank is not None and self.stats.remote.get(rank, 0) > 0:
                self.stats.remote[rank] -= 1
                self.stats.local += 1
                self.stats.fallback += 1
                self._retire_locked(rank, cause)
            self._cond.notify_all()

    def close(self) -> None:
        """No more claims; units already claimed may still arrive."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed

    def take(self, unit: int, *, max_wait_s: float,
             check_interrupt: Callable[[], None] | None = None) -> tuple[bool, Any]:
        """The loop reaches ``unit``: ``(True, result)`` from a rank, else ``(False, None)``.

        ``(False, None)`` means decode it here. A claimed unit whose rank goes
        quiet for ``max_wait_s`` is decoded here too, and the rank is retired.
        """
        deadline = time.monotonic() + max(0.0, float(max_wait_s))
        with self._cond:
            self._front = max(self._front, unit + 1)
            while True:
                state = self._state[unit]
                if state == UNCLAIMED:
                    self._state[unit] = LOCAL
                    self.stats.local += 1
                    return False, None
                if state == DONE:
                    rank = self._owner[unit]
                    self.stats.remote[rank] = self.stats.remote.get(rank, 0) + 1
                    return True, self._results.pop(unit)
                if state == FAILED:
                    self.stats.fallback += 1
                    self.stats.local += 1
                    return False, None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._retire_locked(
                        self._owner[unit], f"no answer for unit {unit} within {max_wait_s:.0f} s")
                    continue
                self._cond.wait(min(_POLL_S, remaining))
                if check_interrupt is not None:
                    self._cond.release()
                    try:
                        check_interrupt()
                    finally:
                        self._cond.acquire()


Send = Callable[[int, int], Any]
Wait = Callable[[Any, float], Any]


def pump(board: UnitBoard, rank: int, send: Send, wait: Wait, *,
         first_timeout_s: float, unit_timeout_s: float, window: int = 2,
         accept: Callable[[Any], str | None] = lambda _reply: None,
         unwrap: Callable[[Any], Any] = lambda reply: reply) -> None:
    """Feed one rank from the back of ``board`` until nothing is left for it.

    ``send(rank, unit)`` dispatches without waiting and returns a handle;
    ``wait(handle, timeout_s)`` returns the rank's reply or raises. ``accept``
    returns None for a reply the loop may use, or the reason it may not (a
    decline, a wrong shape); ``unwrap`` turns an accepted reply into what the
    loop receives. Up to ``window`` units are in flight, so the next unit's
    transfer overlaps this one's decode. The first wait gets
    ``first_timeout_s``: it may include loading the VAE on that rank.
    """
    inflight: deque[tuple[int, Any]] = deque()
    first = True
    while True:
        while len(inflight) < max(1, window):
            unit = board.claim(rank)
            if unit is None:
                break
            try:
                inflight.append((unit, send(rank, unit)))
            except BaseException as exc:
                board.fail(rank, f"could not send unit {unit}: {_summary(exc)}")
                return
        if not inflight:
            return
        unit, handle = inflight.popleft()
        try:
            reply = wait(handle, first_timeout_s if first else unit_timeout_s)
        except BaseException as exc:
            board.fail(rank, f"unit {unit}: {_summary(exc)}")
            return
        first = False
        problem = accept(reply)
        if problem is not None:
            board.fail(rank, problem)
            return
        board.deliver(unit, rank, unwrap(reply))


def _summary(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    first = text[0] if text else ""
    return f"{type(exc).__name__}: {first[:300]}" if first else type(exc).__name__
