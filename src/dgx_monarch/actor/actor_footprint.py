"""What an actor exit would return beyond its residents, for the old-model card.

An in-process unload of a stock resident returns its GPU copy, about the file
size, but not the host-side arena the actor kept from earlier loads; an actor
exit returns both (measured 2026-11-04 on torch 2.12.0, docs/DESIGN.md section
5.9; with the purge delay at 0, cli/worker_process_env, a free on the thread
that allocated returns at once and the arena is mostly what other threads'
frees kept). The arena
is this process's anonymous memory now, less what it held at its first load,
before any weight was placed: this box's own floor for a fresh actor, which the
fresh actor a clear brings up pays again.

The three advisory quote fields live here too. They only ever decide whether a
refusal offers the "Old model in the way" card, so a row from an older Worker
that lacks them reads as "unknown" and offers none, where a missing required
key still refuses like a silent rank.

Reads ``/proc/self/status`` only: no torch, no CUDA, no lock beyond its own.
"""
from __future__ import annotations

import threading
from typing import Any

STATUS_PATH = "/proc/self/status"
_KIB = 1024
# What a fresh actor takes back on its first load (lazy imports, kernel
# caches); old_model.CLEAR_MARGIN_BYTES is the driver's copy of this figure.
RESET_MARGIN_BYTES = 2 << 30

ADVISORY_ROW_KEYS: frozenset[str] = frozenset({
    "resident_names", "resident_bytes", "arena_bytes",
})

# Process lifetime, written once: the first load's reading, or None when that
# reading failed. A later load never replaces it, because a later reading
# already holds weights and would hide the arena it should measure.
_FRESH: dict[str, int | None] = {}
_LOCK = threading.Lock()


def anonymous_bytes(path: str = STATUS_PATH) -> int | None:
    """This process's anonymous resident memory (``RssAnon``), or None."""
    try:
        with open(path) as handle:
            for line in handle:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * _KIB
    except (OSError, ValueError, IndexError):
        return None
    return None


def note_first_load(path: str = STATUS_PATH) -> None:
    """Take the fresh floor once, before this process places its first weight."""
    with _LOCK:
        if "floor" not in _FRESH:
            _FRESH["floor"] = anonymous_bytes(path)


def arena_bytes(path: str = STATUS_PATH) -> int:
    """Anonymous memory above the fresh floor, or 0 when either reading is missing."""
    with _LOCK:
        floor = _FRESH.get("floor")
    if floor is None:
        return 0
    now = anonymous_bytes(path)
    return 0 if now is None else max(now - floor, 0)


def ladder_clause(required: int, available: int | None) -> str:
    """The Worker ladder's sentence about memory an earlier load left here.

    The ladder prices after its own in-process drop, so the old model is gone
    by then and only its arena stays (a Fleet job and a one-GPU fleet meet
    this; neither asks the cross-rank agreement). When a reset would give back
    more than the shortfall, say so and name the reset; otherwise say nothing.
    """
    if available is None:
        return ""
    short = max(int(required) - int(available), 0)
    arena = arena_bytes()
    if short <= 0 or short + RESET_MARGIN_BYTES >= arena:
        return ""
    return (f" This Worker still holds about {arena / (1 << 30):.1f} GiB that a model "
            "loaded earlier left behind, and a reset gives it back: click Reset attached "
            "mesh in the DGX Monarch panel, then queue the render again.")


def block_clause(block: Any) -> str:
    """``ladder_clause`` from a quote block's own figures, or nothing."""
    try:
        return ladder_clause(int(block["required_bytes"]), block.get("mem_available_bytes"))
    except (KeyError, TypeError, ValueError, AttributeError):
        return ""


def forget() -> None:
    """Drop the floor (tests)."""
    with _LOCK:
        _FRESH.clear()


def _names(value: Any) -> dict[str, str | None]:
    if not isinstance(value, dict):
        return {"cond": None, "uncond": None}
    return {str(slot): (name if isinstance(name, str) and name else None)
            for slot, name in value.items()}


def _sizes(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(slot): size for slot, size in value.items()
            if isinstance(size, int) and not isinstance(size, bool) and size >= 0}


def advisory_fields(row: dict[str, Any]) -> dict[str, Any]:
    """The advisory half of a wire row, with "unknown" for whatever is absent."""
    arena = row.get("arena_bytes", 0)
    return {
        "resident_names": _names(row.get("resident_names")),
        "resident_bytes": _sizes(row.get("resident_bytes")),
        "arena_bytes": (arena if isinstance(arena, int) and not isinstance(arena, bool)
                        and arena >= 0 else 0),
    }
