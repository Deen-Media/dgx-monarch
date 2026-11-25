"""Refuse a fleet whose Workers run two builds before any rank waits on the other.

The ranks of a render make collectives whose count and size follow the code:
the partial-load guard's exchange carries the pair's backup agreement beside
its own flag (``actor/partial_load_guard.py``, ``actor/backup_agreement.py``),
and a rank on a build that sends another count would leave its peer waiting
out the group's timeout with no message, or reading wrong values.

Each Worker reports a short digest of the package source it runs in its setup
reply (``actor/worker_env.setup_info``), and the driver keeps the replies on
the mesh handle (``mesh_setup.publish_worker_capabilities``). Before a load or
a render is dispatched to a fleet whose ranks share a collective (a world of
two or more), ``refuse_two_builds`` reads them and refuses, class P, where:

- two reports carry different digests; or
- one report carries the field and another does not have it at all. A Worker
  from before this check sends no such field, and that is the one mixed state
  a first deploy of this release can produce.

A report whose field is there and empty could not work its digest out: it
makes no claim and refuses nothing. A Fleet's actors each run a world of one
and share no collective, so they are not compared. A report is named by its
position in the replies, as ``worker_runtime.cohort_runtime`` names them: a
Fleet gives every actor rank 0.
"""
from __future__ import annotations

from typing import Any

from .refusal import RefusalClass, refusal

SOURCE_FIELD = "source"
OLDER_BUILD = "a build from before this check"
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
    """The reports that state each build, by position, among those whose ranks share a collective.

    A report with no source field at all is an older build (``OLDER_BUILD``);
    one whose field is empty makes no claim and is left out; one whose world is
    1 shares no collective and is left out.
    """
    seen: dict[str, list[int]] = {}
    for position, report in enumerate(reports if isinstance(reports, (list, tuple)) else ()):
        if not isinstance(report, dict):
            continue
        try:
            if int(report.get("world") or 0) < 2:
                continue
        except (TypeError, ValueError):
            continue
        if SOURCE_FIELD not in report:
            seen.setdefault(OLDER_BUILD, []).append(position)
            continue
        digest = report[SOURCE_FIELD]
        if isinstance(digest, str) and digest:
            seen.setdefault(digest, []).append(position)
    return seen


def refuse_two_builds(handle: Any) -> None:
    """Raise where the fleet's Workers stated more than one build at setup."""
    seen = builds(getattr(handle, "worker_capabilities", None))
    if len(seen) < 2:
        return
    told = "; ".join(f"Worker(s) {', '.join(str(position) for position in sorted(positions))} run {build}"
                     for build, positions in sorted(seen.items()))
    raise FleetTwoBuildsError(refusal(
        RefusalClass.PHYSICS,
        "the Workers of this fleet run different builds of dgx-monarch (" + told + ", counted from 0 in "
        "the order the fleet names its boxes). The ranks of a render exchange messages whose count "
        "follows the code, so two builds would leave one rank waiting on the other until the group's "
        "timeout. Nothing was dispatched. Use one release on every box: update each (dgxm update), "
        "restart its Worker, and reset the attached mesh.",
        waivable=False))


__all__ = ["OLDER_BUILD", "SOURCE_FIELD", "FleetTwoBuildsError", "builds", "refuse_two_builds", "source_digest"]
