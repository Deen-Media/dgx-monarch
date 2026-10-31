"""How long the driver waits for one render's sample call.

``nodes/render_submit`` hands this deadline to ``PendingRender``, whose
collect stops waiting once it passes, asks the actors to cancel and abandons
the result. The deadline grows with the work: the step count times a per-step
allowance that scales with the packed rows each rank samples, for the families
whose rows the driver counts (``nodes/render_rows.packed_rows_for_request``).
A render with no row count keeps the older rule, the larger of the floor and
``STEP_FLOOR_S`` per step. The stall guard in ``mesh_lease`` still catches a
silent fleet long before any of these bounds (docs/TROUBLESHOOTING.md #86).
docs/TROUBLESHOOTING.md #139 states the rule and the renders it was set from.

Standard library only, like ``h3_rows``, so the rule tests without torch.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Final

from .log import get_logger

log = get_logger(__name__)

SAMPLE_TIMEOUT_ENV: Final = "DGXM_SAMPLE_TIMEOUT"

# The least any render waits, in seconds. DGXM_SAMPLE_TIMEOUT replaces it,
# and the floor wins over every term below, the ceiling included, so an
# operator can always raise the deadline.
DEFAULT_FLOOR_S: Final = 3600.0

# The least any render waits per step, and the whole per-step allowance of a
# render whose rows the driver does not count.
STEP_FLOOR_S: Final = 120.0

# The per-step allowance at the anchor row count, scaled linearly in rows per
# rank. The anchor is the 15 s MiniMax H3 clip on one Spark (1344x768, 362
# frames, exact attention): it sampled at 239 s per step with 109,466 packed
# rows on its one rank (2026-10-30), and 600 s is 2.5 times that pace.
ANCHOR_ROWS_PER_RANK: Final = 109_466
ANCHOR_STEP_S: Final = 600.0

# The row-scaled deadline stops here, so a hung actor that keeps reporting
# progress is still caught.
CEILING_S: Final = 6 * 3600.0


@dataclass(frozen=True, slots=True)
class SampleDeadline:
    """One render's deadline and the terms that chose it."""

    seconds: float
    steps: int
    floor_s: float
    floor_from_env: bool
    step_s: float                     # the per-step allowance that applied
    rows: int | None = None           # the packed total; None when not counted
    ranks: int = 1                    # the sequence-parallel degree
    rows_per_rank: int | None = None
    source: str | None = None         # the family whose count gave the rows
    capped: bool = False              # the ceiling cut steps times step_s

    @property
    def counted(self) -> bool:
        return self.rows_per_rank is not None


def floor_s() -> tuple[float, bool]:
    """The floor in force, and whether ``DGXM_SAMPLE_TIMEOUT`` set it.

    Read on every render, as the stall budget is. A value that does not read
    as a number keeps the default, with a WARNING.
    """
    raw = os.environ.get(SAMPLE_TIMEOUT_ENV)
    if raw is None:
        return DEFAULT_FLOOR_S, False
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if math.isnan(value):
        log.warning("ignoring non-numeric %s=%r; using the default %g s floor",
                    SAMPLE_TIMEOUT_ENV, raw, DEFAULT_FLOOR_S)
        return DEFAULT_FLOOR_S, False
    return value, True


def step_allowance_s(rows_per_rank: int) -> float:
    """Seconds per step for this many packed rows on one rank, at least the step floor."""
    return max(STEP_FLOOR_S, ANCHOR_STEP_S * rows_per_rank / ANCHOR_ROWS_PER_RANK)


def deadline_s(steps: int, rows_per_rank: int | None, floor: float) -> float:
    """The deadline for ``steps`` steps at ``rows_per_rank``, without the env read.

    No rows: the larger of the floor and the step floor per step, uncapped, as
    before rows were counted. Rows: steps times the row-scaled allowance, cut
    at the ceiling, and never under the floor.
    """
    steps = max(int(steps), 0)
    if rows_per_rank is None:
        return max(floor, steps * STEP_FLOOR_S)
    return max(floor, min(steps * step_allowance_s(rows_per_rank), CEILING_S))


def sample_deadline(steps: int, rows: int | None = None, ranks: int = 1, *,
                    source: str | None = None) -> SampleDeadline:
    """The deadline for one render: ``steps`` steps over ``rows`` packed rows.

    ``rows`` divides over ``ranks``, the sequence-parallel degree, rounding up,
    because a pad row costs a step like any other. A missing or non-positive
    count takes the older rule.
    """
    floor, from_env = floor_s()
    steps = max(int(steps), 0)
    if rows is None or rows <= 0:
        return SampleDeadline(
            seconds=deadline_s(steps, None, floor), steps=steps, floor_s=floor,
            floor_from_env=from_env, step_s=STEP_FLOOR_S)
    ranks = max(int(ranks), 1)
    rows_per_rank = -(-int(rows) // ranks)
    step_s = step_allowance_s(rows_per_rank)
    return SampleDeadline(
        seconds=deadline_s(steps, rows_per_rank, floor), steps=steps,
        floor_s=floor, floor_from_env=from_env, step_s=step_s, rows=int(rows),
        ranks=ranks, rows_per_rank=rows_per_rank, source=source,
        capped=steps * step_s > CEILING_S)


def _clock(seconds: float) -> str:
    """``h:mm:ss``, to read beside ComfyUI's prompt time."""
    if seconds == math.inf:
        return "unbounded"
    whole = round(seconds) if seconds > 0 else 0
    return f"{whole // 3600}:{whole // 60 % 60:02d}:{whole % 60:02d}"


def describe(deadline: SampleDeadline) -> str:
    """The INFO line: the deadline, the term that set it, then every term."""
    floor_name = SAMPLE_TIMEOUT_ENV if deadline.floor_from_env else "default"
    steps_term = f"{deadline.steps} steps x {deadline.step_s:.1f} s"
    work = deadline.steps * deadline.step_s
    if deadline.counted:
        bound = min(work, CEILING_S)
        counted_by = f"{deadline.source}: " if deadline.source else ""
        terms = (f"{steps_term} = {work:.0f} s for {deadline.rows_per_rank} packed "
                 f"rows per rank ({counted_by}{deadline.rows} rows over "
                 f"{deadline.ranks} rank(s); {ANCHOR_STEP_S:g} s per step at "
                 f"{ANCHOR_ROWS_PER_RANK} rows per rank, at least {STEP_FLOOR_S:g} s), "
                 f"ceiling {CEILING_S:g} s")
        rule = "the ceiling" if deadline.capped else steps_term
    else:
        bound = work
        terms = (f"{steps_term} = {work:.0f} s, no ceiling; the driver counts no "
                 "packed rows for this render")
        rule = steps_term
    winner = "the floor" if deadline.floor_s >= bound else rule
    return (f"sample deadline: {deadline.seconds:.0f} s ({_clock(deadline.seconds)}) "
            f"from {winner}; terms: floor {deadline.floor_s:g} s ({floor_name}), {terms}")
