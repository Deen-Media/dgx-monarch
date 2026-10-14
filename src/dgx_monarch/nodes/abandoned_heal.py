"""The one-shot reset for an abandoned-lease wedge (docs/TROUBLESHOOTING.md #87).

An untagged worker crash or a failed result read abandons its sample lease,
and every later dispatch on that fleet refuses with ``LifecycleBusyError``.
When abandoned leases are the only obstruction, one observed recycle clears
them: the stopped fleet leaves the mesh cache, and the next ``ensure_live``
builds a fresh handle whose lease counts start empty. The normal render path
runs this on the fleet whose dispatch refused and then retries once. Fleet
runs it once per queue on its live handle, before the render session binds.
Neither reads the Init node's cached handle, which keeps its count after a
recycle. A failed stop can leave Fleet's handle retired, so after a recycle
that is not ok Fleet raises the abandoned-sample refusal itself instead of
binding. The nodes that set the fleet up before any render (Init with an
explicit preset, the UNET Loader, the BasicScheduler) run the same heal
through ``heal_before_eager_setup``.
"""
from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

from ..log import get_logger
from ..mesh import MeshHandle

log = get_logger(__name__)


class HealOutcome(Enum):
    """What one heal attempt did to the handle it was given."""

    SKIPPED = "skipped"  # not an abandoned-only wedge; the handle is untouched
    HEALED = "healed"  # the ProcMesh stop was observed; use the new fleet
    RECYCLE_FAILED = "recycle_failed"  # the recycle was not ok


def heal_abandoned_wedge_once(
        handle: Any, *, blocker: str, then: str) -> HealOutcome:
    """Recycle once when abandoned leases are the only dispatch obstruction.

    Recycle permits abandoned leases but refuses a live sample or a pending
    RDMA read, and this check refuses them first, so active work is never
    interrupted. ``HEALED`` means the ProcMesh stop was observed and the
    caller may dispatch on the new fleet. Any other busy state or an
    unreadable count answers ``SKIPPED``. A recycle that is not ok answers
    ``RECYCLE_FAILED``: its stop may have retired the handle, and the caller
    must raise its original refusal. ``blocker`` and ``then`` finish the log
    line for the caller's path.
    """
    from ..rdma_read_job import pending_job_count_for_handle

    if not isinstance(handle, MeshHandle):
        return HealOutcome.SKIPPED
    try:
        abandoned = sum(
            int(count) for count in handle.abandoned_sample_leases.values())
        active = sum(int(count) for count in handle.sample_leases.values())
        pending_reads = pending_job_count_for_handle(handle)
    except Exception:
        return HealOutcome.SKIPPED
    if not abandoned or active or pending_reads:
        return HealOutcome.SKIPPED
    log.info(
        "healing the abandoned-lease wedge: a crashed render left %d abandoned "
        "sample lease(s), so %s; recycling the attached mesh once, then %s "
        "(docs/TROUBLESHOOTING.md #87)", abandoned, blocker, then)
    outcome = handle.recycle_detailed()
    if not outcome.ok:
        log.warning(
            "abandoned-lease heal could not reset the mesh (%s: %s); the "
            "original refusal stands", outcome.status.value, outcome.detail)
        return HealOutcome.RECYCLE_FAILED
    # As on every process-stop path, clear the driver's residency memos: they
    # must not credit weights the respawned fleet will not hold.
    from . import recycle_drain

    recycle_drain.drop_residency_memos()
    return HealOutcome.HEALED


def heal_before_eager_setup(
        handle: Any, *, site: str, respawn: Callable[[Any], Any]) -> Any:
    """Heal an abandoned-only wedge before a node sets up the fleet eagerly.

    Init with an explicit preset, the UNET Loader and the BasicScheduler set
    up topology and worker arguments before any render, and that setup
    refuses while an abandoned lease is held (mesh_setup.require_no_sample_leases).
    A changed checkpoint, topology or lever after an untagged crash would
    otherwise need a manual Reset. Fleet's rule applies: a heal answers the
    fresh fleet from ``respawn``, a recycle that is not ok raises the
    abandoned-sample refusal, and anything else returns ``handle`` unchanged.
    """
    heal = heal_abandoned_wedge_once(
        handle, blocker=f"{site} would refuse its setup change",
        then=f"setting up {site} on the new fleet")
    if heal is HealOutcome.HEALED:
        return respawn(handle)
    if heal is HealOutcome.RECYCLE_FAILED:
        from .. import mesh_setup

        mesh_setup.require_no_abandoned_samples(handle, f"set up {site}")
    return handle
