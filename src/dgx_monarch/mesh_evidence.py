"""Shared, read-only predicates for driver mesh telemetry.

Status, Doctor, and readiness use the same predicates so a dirty fleet cannot
receive inconsistent health verdicts. This module uses only the standard
library.
"""
from __future__ import annotations

from collections.abc import Mapping

# Recognized states only. A newer, unknown state is not health evidence;
# ``poisoned`` is known unusable and must reach ``mesh_block_dirty``.
KNOWN_STATES = ("ok", "idle", "busy", "dirty", "poisoned")


def strict_count(value: object) -> int | None:
    """A non-negative plain int, or None for anything that is not one."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _text(value: object) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def mesh_block_dirty(block: Mapping) -> bool:
    """Whether the block reports a fleet the next attach cannot use."""
    verdict = _text(block.get("verdict"))
    replacement = block.get("replacement_blocked", False)
    busy = block.get("busy_phase")
    return (
        _text(block.get("state")) in {"dirty", "blocked", "poisoned"}
        or verdict in {"blocked", "poisoned"}
        or any(
            (strict_count(block.get(key)) or 0) > 0
            for key in ("abandoned_samples", "abandoned_leases")
        )
        or replacement is True
        or (isinstance(replacement, str) and bool(replacement.strip()))
        or block.get("poisoned") is True
        or (verdict == "unresolved"
            and not (busy.strip() if isinstance(busy, str) else ""))
    )


def mesh_block_unobserved(block: Mapping | None) -> bool:
    """Whether this view proves nothing about the attached mesh.

    Four ways to learn nothing: nobody answered, a driver answered with no
    mesh block, the driver said it could not read its own state, and a state
    this reader does not know. None of them is evidence of health.
    """
    if not block:
        return True
    return _text(block.get("state")) not in KNOWN_STATES
