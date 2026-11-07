"""Each rank's torch and CUDA build, and the rule Gate authority holds them to.

A capability context names the driver's build (``gate_runtime``), so a PASS
proven under one torch never vouches for another. That only holds when every
rank runs the driver's build, and a Worker host can change its torch on its
own. So each rank reports its build three ways: in its setup reply, in each
Gate provenance row, and by checking a grant's build against its own.

The driver refuses Gate authority on a pair whose ranks do not all report the
driver's build, a report that lacks the build included, and names each box's
torch and CUDA in plain words (``box_names``). The context string does not
change, so a pair that matches keeps its PASS rows.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from . import box_names, gate_runtime
from .adapters.base import UnsupportedModelError
from .gate_runtime import RUNTIME_FIELD
from .refusal import RefusalClass, refusal

# The fix every refusal here names; cohort_runtime spells it out at its raise site.
FIX_TEXT = ("Use one torch and CUDA build on every box: update the box that differs, and "
            "`dgxm doctor`'s `torch version match` row shows when they match "
            "(docs/TROUBLESHOOTING.md #8).")


class WorkerRuntimeMismatch(UnsupportedModelError):
    """A pair whose ranks do not all run the driver's torch and CUDA build."""


def build_words(runtime: object) -> str:
    """One build in plain words: 'torch 2.15.0 with CUDA 13.2', or 'no torch build'."""
    if not isinstance(runtime, dict):
        return "no torch build"
    return f"torch {runtime.get('torch', 'unknown')} with CUDA {runtime.get('cuda', 'unknown')}"


_NO_REPORT = "sent no report of its build"


def _box_phrase(runtime: object) -> str:
    if isinstance(runtime, dict):
        return f"runs {build_words(runtime)}"
    return "reports no torch build, as a release before this one does"


def mismatch(handle: Any, rows: Iterable[object], driver: dict | None = None) -> str | None:
    """The sentence that refuses Gate authority, or None when every rank matches.

    ``rows`` are per-rank replies that carry ``rank`` and, from this release,
    ``runtime``. A row without a usable rank reads as a box the config does not
    list, so it still counts, and a rank of the world with no row at all is a
    mismatch too: missing identity never authorizes.
    """
    driver = gate_runtime.runtime_build_identity() if driver is None else driver
    replies: list[dict] = [row for row in rows if isinstance(row, dict)]
    world = max([int(getattr(handle, "world", 0) or 0)]
                + [row["rank"] + 1 for row in replies if type(row.get("rank")) is int])
    boxes = [box for box, _gpu in box_names.fleet_boxes(handle, world)]
    found: dict[str, list[str]] = {}  # box -> what its ranks report, in rank order
    differs = False
    for row in replies:
        rank = row.get("rank")
        box = boxes[rank] if type(rank) is int and 0 <= rank < len(boxes) else box_names.UNLISTED
        runtime = row.get(RUNTIME_FIELD)
        differs = differs or runtime != driver
        phrase = _box_phrase(runtime)
        if phrase not in found.setdefault(box, []):
            found[box].append(phrase)
    seen = {row.get("rank") for row in replies}
    for rank in (rank for rank in range(world) if rank not in seen):
        differs = True
        if _NO_REPORT not in found.setdefault(boxes[rank], []):
            found[boxes[rank]].append(_NO_REPORT)
    if not differs:
        return None
    each = "; ".join(f"{box} {' and '.join(phrases)}" for box, phrases in found.items())
    return (f"Gate authority needs every box on the driver's torch build, and they differ: "
            f"the driver runs {build_words(driver)}; {each}. A Gate proof on one build says "
            f"nothing about another, so no PASS is used, no ceremony runs and no grant is "
            f"issued.")


def cohort_runtime(handle: Any) -> dict[str, str]:
    """The driver's build, once every setup report on ``handle`` agrees with it.

    A handle with no setup reports yet makes no claim: render submit authorizes
    before it sets up, and the dispatch check after setup, the Worker's own
    grant check and the ceremony's provenance rows each answer again.
    """
    driver = gate_runtime.runtime_build_identity()
    reports = getattr(handle, "worker_capabilities", None)
    if isinstance(reports, list) and reports:
        text = mismatch(handle, reports, driver)
        if text is not None:
            raise WorkerRuntimeMismatch(refusal(RefusalClass.PHYSICS, (
                f"{text} Use one torch and CUDA build on every box: update the box that "
                "differs, and `dgxm doctor`'s `torch version match` row shows when they "
                "match (docs/TROUBLESHOOTING.md #8).")))
    return driver


def assert_grant_runtime(context: dict, local: dict | None) -> None:
    """A Worker refuses a grant proven under a build other than its own."""
    if local is not None and context.get(RUNTIME_FIELD) != local:
        raise RuntimeError(
            f"this Worker runs {build_words(local)}, and the Gate grant names "
            f"{build_words(context.get(RUNTIME_FIELD))}. A Gate proof on one build says "
            f"nothing about another. {FIX_TEXT}")
