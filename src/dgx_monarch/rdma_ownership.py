"""Durable RDMA descriptor ownership across actor result handoffs."""
from __future__ import annotations

import secrets
import threading
from collections import deque
from collections.abc import Callable
from typing import Any, Literal

from .transfer_utils import prefer_error, raise_with_distinct_cause

REGISTERING = "REGISTERING"
READY = "READY"
RETIRING = "RETIRING"
RELEASED = "RELEASED"
RDMA_ACK_ATTEMPTS = 2
RDMA_ACK_TIMEOUT_S = 120.0

AckStatus = Literal["released", "already_released", "unknown"]


class HandoffRegistry:
    """Own RDMA registrations and backing until a generation-bound ACK.

    One mapping remains the ownership record for its whole lifetime.  ACK does
    not move an entry between containers: it first publishes ``RETIRING`` on
    that mapping, clears only Python ownership, then leaves a bounded
    ``RELEASED`` tombstone so a lost reply can be retried safely.
    """

    def __init__(self, *, max_tombstones: int = 64) -> None:
        self._lock = threading.Lock()
        self._entries: dict[tuple[int, str], dict[str, Any]] = {}
        self._tombstones: deque[tuple[int, str]] = deque()
        self._max_tombstones = max(1, int(max_tombstones))

    @staticmethod
    def _identity(handoff: dict[str, Any]) -> tuple[int, str] | None:
        generation = handoff.get("setup_generation")
        token = handoff.get("token")
        if (isinstance(generation, bool) or not isinstance(generation, int)
                or generation < 1 or not isinstance(token, str)):
            return None
        return generation, token

    def _find_locked(
        self, handoff: dict[str, Any]
    ) -> tuple[tuple[int, str], dict[str, Any]] | None:
        identity = self._identity(handoff)
        if identity is not None and self._entries.get(identity) is handoff:
            return identity, handoff
        return next(
            ((key, entry) for key, entry in self._entries.items()
             if entry is handoff),
            None,
        )

    def publish(
        self,
        handoff: dict[str, Any],
        setup_generation: int,
        capacity: int,
    ) -> None:
        """Reserve an attempt before native construction.

        The caller creates and holds ``handoff`` before the call, so an interruption
        inside or right after it cannot leave the registry owning a mapping the
        caller has no reference to. After a successful call the registry owns it; a
        full registry leaves it unowned for message transport.
        """
        if (isinstance(setup_generation, bool)
                or not isinstance(setup_generation, int)
                or setup_generation < 1):
            raise ValueError("RDMA handoff setup generation must be positive")
        capacity = max(1, int(capacity))
        with self._lock:
            if self._find_locked(handoff) is not None:
                raise RuntimeError("RDMA handoff mapping is already registry-owned")
            handoff.clear()
            handoff.update(
                setup_generation=setup_generation,
                parts=[],
                keepalive=None,
                state="registering",
                _registry_state=REGISTERING,
            )
            live = sum(
                entry.get("_registry_state") != RELEASED
                for entry in self._entries.values()
            )
            if live >= capacity:
                return
            token = secrets.token_hex(16)
            identity = (setup_generation, token)
            while identity in self._entries:
                token = secrets.token_hex(16)
                identity = (setup_generation, token)
            handoff["token"] = token
            self._entries[identity] = handoff

    def owns(self, handoff: dict[str, Any]) -> bool:
        with self._lock:
            return self._find_locked(handoff) is not None

    def mark_ready(self, handoff: dict[str, Any]) -> None:
        with self._lock:
            found = self._find_locked(handoff)
            if found is None:
                raise RuntimeError("RDMA handoff lost registry ownership")
            state = found[1].get("_registry_state")
            if state == REGISTERING:
                found[1]["_registry_state"] = READY
            elif state != READY:
                raise RuntimeError(f"cannot ready RDMA handoff in state {state!r}")

    @staticmethod
    def _definitely_empty(handoff: dict[str, Any]) -> bool:
        # Process-poison adoption is not a driver ACK: a resource-bearing
        # generation entry remains an independent root until actor recycle.
        if handoff.get("parts"):
            return False
        # LatentReturn publishes its CPU backing immediately before entering
        # _register_parts. An interruption at that call boundary leaves an
        # owned REGISTERING entry with no part and therefore no possible native
        # handle. The backing alone is safe to forget and must not consume the
        # depth cap until actor recycle. Once registration changes phase, an
        # empty-parts/nonempty-keepalive combination is ambiguous and retained.
        return (
            handoff.get("keepalive") is None
            or (
                handoff.get("_registry_state") == REGISTERING
                and handoff.get("state") == "registering"
            )
        )

    def reconcile(self, handoff: dict[str, Any]) -> bool:
        """Discard only an owned attempt proven to have no resources.

        Resource-bearing ambiguous attempts remain durable and consume capacity
        until an ACK (for a delivered READY descriptor) or actor recycle.
        """
        with self._lock:
            found = self._find_locked(handoff)
            if found is None or not self._definitely_empty(found[1]):
                return False
            found[1].get("parts", []).clear()
            found[1]["keepalive"] = None
            self._entries.pop(found[0], None)
            return True

    def acknowledge(self, setup_generation: int, token: str) -> AckStatus:
        if (isinstance(setup_generation, bool)
                or not isinstance(setup_generation, int)
                or not isinstance(token, str)):
            return "unknown"
        identity = setup_generation, token
        with self._lock:
            handoff = self._entries.get(identity)
            if handoff is None:
                return "unknown"
            state = handoff.get("_registry_state")
            if state == RELEASED:
                self._record_tombstone_locked(identity)
                return "already_released"
            if state not in (READY, RETIRING):
                return "unknown"

            # Publish retry evidence before clearing either ownership handle.
            handoff["_registry_state"] = RETIRING
            self._record_tombstone_locked(identity)
            parts = handoff.get("parts")
            if isinstance(parts, list):
                parts.clear()
            else:
                handoff["parts"] = []
            handoff["keepalive"] = None
            handoff["state"] = "acknowledged"
            handoff["_registry_state"] = RELEASED
            return "released"

    def _record_tombstone_locked(self, identity: tuple[int, str]) -> None:
        if identity in self._tombstones:
            return
        while len(self._tombstones) >= self._max_tombstones:
            oldest = self._tombstones[0]
            entry = self._entries.get(oldest)
            if entry is not None and entry.get("_registry_state") == RELEASED:
                self._entries.pop(oldest, None)
            self._tombstones.popleft()
        self._tombstones.append(identity)

    def get(self, setup_generation: int, token: str) -> dict[str, Any] | None:
        with self._lock:
            return self._entries.get((setup_generation, token))

    @property
    def live_count(self) -> int:
        with self._lock:
            return sum(
                entry.get("_registry_state") != RELEASED
                for entry in self._entries.values()
            )

    @property
    def tombstone_count(self) -> int:
        with self._lock:
            return len(self._tombstones)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def fail_closed_handoff(
    handoff: dict,
    phase: str,
    exc: BaseException,
    fail_closed: Callable[[list[dict], Any, str, BaseException], tuple[list, Any]],
) -> tuple[list, Any]:
    failures, poison_error = fail_closed(
        handoff.get("parts", []), handoff.get("keepalive"), phase, exc)
    handoff["state"] = "settled"
    return failures, poison_error


def abort_handoff(
    handoff: dict,
    primary: BaseException,
    fail_closed: Callable[[list[dict], Any, str, BaseException], tuple[list, Any]],
    safe_note: Callable[[BaseException, str, Any], None],
) -> None:
    caught = primary
    strongest = primary

    def note(message: str, detail: Any) -> None:
        try:
            safe_note(primary, message, detail)
        except BaseException:
            pass

    if not handoff.get("parts") or handoff.get("state") == "settled":
        return
    try:
        failures, poison_error = fail_closed_handoff(
            handoff, "worker result publication", primary, fail_closed)
    except BaseException as recovery_error:
        strongest = prefer_error(
            strongest, recovery_error, "RDMA worker handoff recovery interrupted")
        note("RDMA worker handoff recovery interrupted", recovery_error)
        if strongest is not caught:
            raise_with_distinct_cause(strongest, caught)
        return
    if failures:
        detail: Any
        try:
            for failure in failures:
                strongest = prefer_error(
                    strongest,
                    failure.error,
                    "RDMA worker handoff buffer release also failed",
                )
            detail = tuple(failure.summary for failure in failures)
        except BaseException as summary_error:
            detail = summary_error
            strongest = prefer_error(
                strongest, summary_error, "RDMA cleanup summary failed")
        note("RDMA worker handoff cleanup failed", detail)
    if poison_error is not None:
        strongest = prefer_error(
            strongest, poison_error, "RDMA poison publication failed")
        note("RDMA poison publication failed", poison_error)
    if strongest is not caught:
        raise_with_distinct_cause(strongest, caught)


def acknowledge_with_retry(callback: Callable[..., Any], *identity: Any) -> None:
    """Retry only the idempotent ACK; native drops have already succeeded."""
    primary: BaseException | None = None
    for _attempt in range(RDMA_ACK_ATTEMPTS):
        try:
            callback(*identity)
        except BaseException as error:
            if primary is None:
                primary = error
                continue
            if (isinstance(primary, Exception)
                    and not isinstance(error, Exception)):
                try:
                    error.add_note(f"earlier RDMA handoff ACK failed: {primary!r}")
                except BaseException:
                    pass
                raise_with_distinct_cause(error, primary)
            try:
                primary.add_note(f"RDMA handoff ACK retry failed: {error!r}")
            except BaseException:
                pass
            raise_with_distinct_cause(primary, error)
        if primary is not None and not isinstance(primary, Exception):
            raise primary
        return
    if primary is not None:
        raise primary
