"""Process-lifetime RDMA poison records and one-time diagnostics."""
from __future__ import annotations

from typing import Any, NamedTuple


class RDMADropFailure(NamedTuple):
    part: dict
    error: BaseException
    summary: str


class RDMAPoisonOwner(NamedTuple):
    parts: tuple[dict, ...]
    keepalive: Any
    phase: str
    failures: tuple[str, ...]


def new_poison_owner(
    failures: list[RDMADropFailure], keepalive: Any, phase: str,
) -> RDMAPoisonOwner:
    return RDMAPoisonOwner(
        tuple(failure.part for failure in failures), keepalive, phase,
        tuple(failure.summary for failure in failures))


def constructor_intent(part: dict, keepalive: Any) -> RDMAPoisonOwner:
    return RDMAPoisonOwner(
        (part,), keepalive, "RDMA buffer construction",
        ("native registration outcome unknown",))


def unowned_failures(
    failures: list[RDMADropFailure], lock: Any,
    owners: list[RDMAPoisonOwner],
) -> list[RDMADropFailure]:
    with lock:
        return [
            failure for failure in failures
            if not any(failure.part is part for owner in owners
                       for part in owner.parts)
        ]


class PoisonReporter:
    """Atomically claim the first poison owner and report it at most once."""

    def __init__(self) -> None:
        self._reported_owner: RDMAPoisonOwner | None = None

    def report(self, lock: Any, owners: list[RDMAPoisonOwner],
               log_error: Any, safe_log: Any) -> None:
        try:
            with lock:
                if not owners:
                    return
                owner = owners[0]
                if self._reported_owner is owner:
                    return
                self._reported_owner = owner
        except BaseException:
            return
        if owner.phase == "RDMA buffer construction":
            safe_log(log_error,
                "RDMA registration ownership POISONED during %s: the native "
                "registration outcome is unknown, so its backing stays owned until "
                "the process exits and later latent returns from this process use "
                "messaging; Attached mesh reset required", owner.phase)
        else:
            safe_log(log_error,
                "RDMA registration ownership POISONED during %s: %d drop(s) "
                "were not confirmed, so the failed handles and any live backing "
                "stay owned until the process exits; this process registers no more "
                "RDMA buffers, so any latent it sends goes by messaging (cleanup=%s)",
                owner.phase, len(owner.parts), owner.failures)
