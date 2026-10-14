"""Per-render step marks for the render-progress tracker in telemetry.py.

Several senders feed the one tracker. In a dp>1 render, each data-parallel
group's result leader sends every step of the same render, and a Fleet wave
runs several renders at once, each with its own token. If a step were timed
from the last message of any sender, the rate would be the gap between two
senders, often a few milliseconds, and not the render's time per step.

So each render keeps a mark: the highest step it has reported and when. A
step is timed from its own render's mark, and only a step past that mark
counts. A second leader that repeats a step changes nothing.

Stdlib only and stateless. The marks live under `_step_marks` in the
tracker's state dict, and the tracker calls these under its lock.
"""
from __future__ import annotations


def seeded(previous: dict, key: object, now: float) -> dict:
    """The marks a start publishes, given the tracker's state before it.

    Renders still running keep theirs. The new render starts at step 0 now, so
    its first step is timed from its own start, not from another render's step.
    """
    kept = dict(previous.get("_step_marks", {})) if previous.get("active") else {}
    kept[key] = (0, now)
    return kept


def advance(state: dict, key: object, step: int, now: float) -> tuple[dict, float | None]:
    """Return the new marks and this step's seconds, given the tracker's state.

    The seconds are None, and the marks unchanged, when the step does not pass
    the render's mark. A render with no mark is timed from the last step that
    counted, as the tracker did before it kept marks.
    """
    marks = state.get("_step_marks", {})
    done, since = marks.get(key, (0, state.get("last_step_t", now)))
    if step <= done:
        return marks, None
    return {**marks, key: (step, now)}, round(now - since, 3)


def retired(state: dict, *keys: object) -> dict:
    """The tracker's marks without those of renders that finished."""
    marks = state.get("_step_marks", {})
    return {key: mark for key, mark in marks.items() if all(key is not gone for gone in keys)}
