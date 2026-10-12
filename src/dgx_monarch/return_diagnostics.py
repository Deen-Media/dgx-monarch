"""Observational return counters; never evidence for ACK, release, or reuse."""
from __future__ import annotations

import sys
import threading
from typing import Any

RETURN_LABELS = frozenset({
    ("rdma", "native"), ("message", "configured_message"),
    ("message", "registry_capacity"), ("message", "nested_tensor"),
    ("message", "below_threshold"), ("message", "pack_fallback"),
})


class ReturnCounts:
    """Fixed-cardinality descriptor production counts, not delivery receipts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[tuple[str, str], int] = {}

    def record(self, kind: str, reason: str) -> None:
        if (kind, reason) not in RETURN_LABELS:
            return
        with self._lock:
            key = kind, reason
            self._counts[key] = self._counts.get(key, 0) + 1

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [{"kind": kind, "reason": reason, "count": count}
                    for (kind, reason), count in sorted(self._counts.items())]


def _loaded_count(module_name: str, collection: str, lock_name: str) -> int | None:
    """Do not import a runtime or wait behind ownership work to get a metric."""
    module = sys.modules.get(module_name)
    values = getattr(module, collection, None)
    lock = getattr(module, lock_name, None)
    if type(values) is not list or lock is None:
        return None
    try:
        if not lock.acquire(blocking=False):
            return None
        try:
            return len(values)
        finally:
            lock.release()
    except Exception:
        return None


def driver_snapshot() -> dict[str, Any]:
    """Each field is independently sampled; missing does not mean zero."""
    result: dict[str, Any] = {"schema": 1, "available": False}
    for field, module, collection, lock in (
        ("pending_read_jobs", "dgx_monarch.rdma_job_registry", "_JOBS", "_LOCK"),
        ("native_poison_owners", "dgx_monarch.transfer", "_RDMA_POISONED_OWNERS", "_RDMA_POISON_LOCK"),
    ):
        count = _loaded_count(module, collection, lock)
        if count is not None:
            result[field] = count
            result["available"] = True
    return result


def actor_snapshot(worker: Any) -> dict[str, Any]:
    from .rdma_ownership import HandoffRegistry

    result = driver_snapshot()
    registry = getattr(worker, "_latent_handoffs", None)
    if type(registry) is HandoffRegistry:
        # These public counters neither reconcile nor retire registry entries.
        result.update(live_handoffs=registry.live_count, tombstones=registry.tombstone_count,
                      available=True)
    counts = getattr(worker, "_return_counts", None)
    if type(counts) is ReturnCounts:
        result.update(return_counts=counts.snapshot(), available=True)
    return result


def record_descriptor(worker: Any, descriptor: Any, reason: str) -> None:
    """A diagnostic failure cannot turn a published descriptor into cleanup."""
    try:
        counts = getattr(worker, "_return_counts", None)
        if type(counts) is ReturnCounts and isinstance(descriptor, dict):
            kind = descriptor.get("kind")
            if isinstance(kind, str):
                counts.record(kind, reason)
    except BaseException:
        pass


def message_reason(tensor: Any, threshold: int | None) -> str:
    """Classify only observable pack gates; opaque native failure stays generic."""
    try:
        from .transfer_utils import is_nested_tensor

        if is_nested_tensor(tensor):
            return "nested_tensor"
        if threshold is not None and tensor.numel() * tensor.element_size() < threshold:
            return "below_threshold"
    except Exception:
        pass
    return "pack_fallback"
