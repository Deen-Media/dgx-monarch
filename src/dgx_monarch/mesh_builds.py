"""Refuse a fleet whose Workers run two builds before any rank waits on the other.

The ranks of a render make collectives whose count and size follow the code:
the partial-load guard's exchange carries the pair's backup agreement beside
its own flag (``actor/partial_load_guard.py``, ``actor/backup_agreement.py``),
and a rank on a build that sends another count would leave its peer waiting
out the group's timeout with no message.

Each Worker reports a short digest of the package source it runs in its setup
reply (``actor/worker_env.setup_info``), and the driver keeps the replies on
the mesh handle (``mesh_setup.publish_worker_capabilities``). Before a
multi-rank render is dispatched ``refuse_two_builds`` reads them and refuses,
class P, where two ranks report different digests. A rank that could not work
its digest out makes no claim and refuses nothing.
"""
from __future__ import annotations

from typing import Any

from .refusal import RefusalClass, refusal

GUARD = "fleet_two_builds"
SOURCE_FIELD = "source"
_DIGEST_CHARS = 16


class FleetTwoBuildsError(RuntimeError):
    """Typed refusal: the Workers of this fleet do not run the same package source."""


def source_digest() -> str | None:
    """The first characters of this process's package source digest, or None when it cannot be had."""
    try:
        from .runtime_provenance import cached_dgx_source_manifest_sha256

        return str(cached_dgx_source_manifest_sha256())[:_DIGEST_CHARS]
    except Exception:
        return None


def builds(reports: Any) -> dict[str, list[int]]:
    """The ranks reporting each source digest, from the Workers' setup replies; ranks with none are left out."""
    seen: dict[str, list[int]] = {}
    for report in reports if isinstance(reports, (list, tuple)) else ():
        if not isinstance(report, dict):
            continue
        digest = report.get(SOURCE_FIELD)
        if isinstance(digest, str) and digest:
            seen.setdefault(digest, []).append(int(report.get("rank") or 0))
    return seen


def refuse_two_builds(handle: Any) -> None:
    """Raise where the fleet's Workers reported more than one source digest at setup."""
    seen = builds(getattr(handle, "worker_capabilities", None))
    if len(seen) < 2:
        return
    told = "; ".join(f"rank(s) {', '.join(str(rank) for rank in sorted(ranks))} run {digest}"
                     for digest, ranks in sorted(seen.items()))
    raise FleetTwoBuildsError(refusal(
        RefusalClass.PHYSICS,
        "the Workers of this fleet run different builds of dgx-monarch (" + told + "). The ranks of a "
        "render exchange messages whose count follows the code, so two builds would leave one rank "
        "waiting on the other until the group's timeout. Nothing was dispatched. Update every box to the "
        "same release (dgxm update on each), restart its Worker, and reset the attached mesh.",
        guard=GUARD, waivable=False))


__all__ = ["FleetTwoBuildsError", "GUARD", "SOURCE_FIELD", "builds", "refuse_two_builds", "source_digest"]
