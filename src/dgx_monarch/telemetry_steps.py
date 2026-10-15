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
tracker's state dict, and the tracker calls these under its lock. The same
keying holds each render's attention pin fields under `_pins`
(attention_pins.py), so an overlapping render cannot lend its pins to another.
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


def steps_kept(previous: dict, key: object, steps: int) -> dict:
    """The per-render step counts a start publishes: running renders keep theirs."""
    kept = dict(previous.get("_render_steps", {})) if previous.get("active") else {}
    kept[key] = int(steps)
    return kept


def sample_owner(state: dict, token: object | None, legacy_token: object) -> object | None:
    """The render an incoming sampler wall belongs to, or None when no single render owns it.

    A keyed wall belongs to its render while that render is live. An unkeyed
    wall belongs to the only render in flight; with several in flight, or
    none, it has no owner and is dropped rather than published under another
    render's settings.
    """
    tokens = set(state.get("_tokens", ()))
    legacy_count = int(state.get("_legacy_count", 0))
    if token is not None and token is not legacy_token:
        return token if token in tokens else None
    if len(tokens) + legacy_count != 1:
        return None
    return next(iter(tokens)) if tokens else legacy_token


def sampled(state: dict, key: object, sample_s: float) -> dict:
    """The tracker's state with one render's sampler wall and mean step, keyed by render.

    ``sample_s`` is the slowest rank's sampler wall from the render's results,
    so ``mean_step_s`` excludes the load and Gate time a cold ``wall_s`` holds.
    The mean divides by that render's own step count, never the latest
    start's, and the finish record publishes a sample only for the render
    whose facts it carries (``sample_for``), so two renders in flight cannot
    lend each other a time.
    """
    steps = state.get("_render_steps", {}).get(key) or 0
    wall = round(float(sample_s), 3)
    mean = round(float(sample_s) / steps, 3) if steps else None
    return {**state, "_samples": {**state.get("_samples", {}), key: (wall, mean)}}


def sample_for(state: dict, key: object) -> tuple[float | None, float | None]:
    """The sampler wall and mean step noted for one render, or unknown."""
    return state.get("_samples", {}).get(key, (None, None))


def pins_kept(previous: dict) -> dict:
    """The per-render pin records a start keeps: those of renders still running."""
    return dict(previous.get("_pins", {})) if previous.get("active") else {}


def pinned(state: dict, key: object, pins: dict) -> dict:
    """The tracker's state with one render's attention pin fields, keyed by render."""
    return {**state, "_pins": {**state.get("_pins", {}), key: dict(pins)}}


def pins_for(state: dict, key: object) -> dict:
    """The pin fields noted for one render; each None when it noted none."""
    from .attention_pins import RECORD_FIELDS

    noted = state.get("_pins", {}).get(key, {})
    return {name: noted.get(name) for name in RECORD_FIELDS}
