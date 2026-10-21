"""Opt-in, prompt-bound observations; never authority for runtime decisions.

A companion poll grants a short observer lease. Without one, hooks perform no
context lookup, disk I/O, hashing or serialization. Replay is bounded memory;
producer changes and evicted sequence gaps are explicit to durable consumers.
"""
from __future__ import annotations

import contextvars
import json
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

LEASE_SECONDS = 30.0
MAX_BYTES = 8 * 1024 * 1024
MAX_EVENTS = 8192
MAX_EVENT_BYTES = 32768
MAX_PAGE_BYTES = 2 * 1024 * 1024
KINDS = frozenset({"dispatch_submitted", "submission_failed", "worker_results", "result_validated",
                   "render_completed", "render_failed", "native_ack", "fleet_item_completed", "fleet_item_failed"})


@dataclass(frozen=True)
class Scope:
    prompt_id: str
    node_id: str
    list_index: int | None
    render_id: str
    fleet_index: int | None = None


class Journal:
    def __init__(self) -> None:
        self.producer_id = uuid.uuid4().hex
        self._lock = threading.Lock()
        self._events: deque[tuple[dict, int]] = deque()
        self._sequence = 0
        self._bytes = 0
        self._until = 0.0
        self._omitted = 0
        self._coverage_epoch = 0
        self._coverage_started = 0.0

    def active(self) -> bool:
        return time.monotonic() < self._until

    def publish(self, scope: Scope | None, kind: str, facts: dict[str, Any]) -> None:
        if scope is None or not self.active() or kind not in KINDS:
            return
        event = {"schema_version": 1, "producer_id": self.producer_id, "timestamp": time.time(),
                 "prompt_id": scope.prompt_id, "node_id": scope.node_id, "list_index": scope.list_index,
                 "render_id": scope.render_id, "fleet_index": scope.fleet_index, "kind": kind, "facts": facts}
        # Only explicit, JSON-compatible hook fields reach this boundary. Reject
        # oversized/nonfinite observations rather than retaining arbitrary objects.
        try:
            encoded = json.dumps(event, allow_nan=False, separators=(",", ":"))
            size = len(encoded.encode())
            if size > MAX_EVENT_BYTES:
                raise ValueError("oversized observation")
        except (TypeError, ValueError):
            with self._lock:
                self._omitted += 1
            return
        detached = json.loads(encoded)
        with self._lock:
            self._sequence += 1
            detached["sequence"] = self._sequence
            self._events.append((detached, size))
            self._bytes += size
            while len(self._events) > MAX_EVENTS or self._bytes > MAX_BYTES:
                self._bytes -= self._events.popleft()[1]

    def read(self, after: int = 0, producer: str | None = None) -> dict[str, Any]:
        if type(after) is not int or after < 0:
            raise ValueError("after must be a nonnegative sequence")
        with self._lock:
            now = time.monotonic()
            if now >= self._until:
                self._coverage_epoch += 1
                self._coverage_started = time.time()
            self._until = now + LEASE_SECONDS
            changed = producer is not None and producer != self.producer_id
            if changed:
                after = 0
            oldest = self._events[0][0]["sequence"] if self._events else self._sequence + 1
            rows: list[dict[str, Any]] = []
            total = 0
            for event, size in self._events:
                if event["sequence"] <= after:
                    continue
                if len(rows) >= 128 or total + size > MAX_PAGE_BYTES:
                    break
                rows.append(event)
                total += size
            next_sequence = rows[-1]["sequence"] if rows else self._sequence
            page: dict[str, Any] = {"schema_version": 1, "producer_id": self.producer_id, "producer_changed": changed,
                    "oldest_sequence": oldest, "next_sequence": next_sequence, "latest_sequence": self._sequence,
                    "more": next_sequence < self._sequence,
                    "gap": changed or (after < oldest - 1), "lease_seconds": LEASE_SECONDS,
                    "coverage_epoch": self._coverage_epoch, "coverage_started_at": self._coverage_started,
                    "omitted_events": self._omitted, "capture": "explicit_comfy_execution_context_only"}
        # Copy outside the lock so a 2 MiB page never stalls a render's publish.
        # Retained events are never mutated after publish, so the references stay valid.
        page["events"] = deepcopy(rows)
        return page


journal = Journal()
_current: contextvars.ContextVar[Scope | None] = contextvars.ContextVar("dgxm_execution_observer", default=None)


def capture(render_id: str, fleet_index: int | None = None) -> Scope | None:
    if not journal.active():
        return None
    try:
        from comfy_execution.utils import get_executing_context

        context = get_executing_context()
        if context is None:
            return None
        prompt_id, node_id = context.prompt_id, context.node_id
        if not isinstance(prompt_id, str) or not prompt_id or not isinstance(node_id, str) or not node_id:
            return None
        if (len(prompt_id) > 256 or len(node_id) > 512
                or not isinstance(render_id, str) or not render_id or len(render_id) > 512):
            return None
        index = context.list_index
        return Scope(prompt_id, node_id, index if type(index) is int else None, render_id, fleet_index)
    except Exception:
        return None


def emit(scope: Scope | None, kind: str, **facts: Any) -> None:
    try:
        journal.publish(scope, kind, facts)
    except Exception:
        # Observability must not alter sampling, ownership or refusal behavior.
        pass


@contextmanager
def activate(scope: Scope | None) -> Iterator[None]:
    token = _current.set(scope)
    try:
        yield
    finally:
        _current.reset(token)


def native_ack(generation: int, token: str, statuses: list[str]) -> None:
    scope = _current.get()
    if scope is None or not journal.active():
        return
    import hashlib

    # Owner tokens are capabilities, so journal only their one-way identity.
    emit(scope, "native_ack", setup_generation=generation,
         owner_token_sha256=hashlib.sha256(token.encode()).hexdigest(), statuses=statuses,
         authority="validated_full_world_owner_ack")


def bind_operation(operation: Callable[..., Any]) -> Callable[..., Any]:
    """Carry only observer identity to the existing scratch-thread operation."""
    scope = _current.get()
    if scope is None:
        return operation

    def observed(*args: Any, **kwargs: Any) -> Any:
        with activate(scope):
            return operation(*args, **kwargs)

    return observed


def context_available() -> bool:
    try:
        from comfy_execution.utils import get_executing_context

        return callable(get_executing_context)
    except Exception:
        return False
