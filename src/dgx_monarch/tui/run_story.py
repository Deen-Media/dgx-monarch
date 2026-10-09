"""Combine telemetry samples into the phases and timing of one run.

Phases include bring-up, loading, identity checks, cross-residency comparison
and the requested render (docs/TROUBLESHOOTING.md #61). Use driver timestamps
or, when absent, sample timestamps so live and replayed results agree. This
module has no clock, HTTP, Rich or mesh dependencies.

Only ``first_render`` phases and render-tracker state create phases; ignore
unrelated notices. ``gate_done`` separates validation renders from the
requested render. Retain a completed run until a new phase arrives. An
unfinished run ends after STALE_S of inactivity with no gate or render active,
on a second gate notice, or on a render outside the current identity check.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

# Ignore additional phases after this limit to keep the display bounded.
MAX_LEGS = 24
# Idle-run expiry matches ``telemetry._CEREMONY_CLAIM_S`` (pinned by
# test_tui_run_story.py), so a later run cannot inherit an expired gate claim.
STALE_S = 300.0
_PHASES = frozenset({"deferred", "gate", "gate_proof", "gate_cross", "gate_done"})
_DEFERRED_TAIL = " on first render"
_DIGITS = re.compile(r"\d+")
# The gate cannot know how many proof renders it will need until it ends, so
# the live label quotes the notice's range.
_PROOF_RANGE = " of 2-4"


@dataclass(frozen=True)
class Leg:
    """One run phase with its label, start time and elapsed seconds."""

    kind: str
    label: str
    start: float
    seconds: float | None = None


@dataclass(frozen=True)
class Story:
    """Phases, duration and completion state for one run."""

    legs: tuple[Leg, ...]
    total_s: float | None
    complete: bool
    ceremony: bool
    truncated: bool
    gate_note: str | None


def _number(value: object) -> float | None:
    """Return a finite numeric value as float, or None for invalid input."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _proof_index(note: str, fallback: int) -> int:
    found = _DIGITS.search(note)
    return int(found.group()) if found else fallback


def leg_seconds(leg: Leg) -> str:
    """Format elapsed seconds, or a pending marker for an unfinished phase."""
    return "…" if leg.seconds is None else f"{leg.seconds:.1f}s"


def story_spans(story: Story | None) -> tuple[tuple[str, str], ...]:
    """The retained block as (text, style key) pairs; the TUI owns the colors.

    The layout lives here so it is testable without a terminal.
    """
    if story is None:
        return (("no run recorded yet", "dim"),)
    spans: list[tuple[str, str]] = [(" this run ", "head")]
    for leg in story.legs:
        spans.append((f"  {leg.label} ", "dim"))
        spans.append((leg_seconds(leg), "cost" if leg.kind == "render" else ""))
    if story.truncated:
        spans.append(("  …", "dim"))
    total = story.total_s if story.complete else None
    spans.append((f"   total {total:.0f}s", "good") if total is not None
                 else ("   running", "warn"))
    if story.gate_note:
        spans.append((f"\n {story.gate_note}", "dim"))
    return tuple(spans)


class RunStory:
    """Track the current run by consuming every telemetry sample in order."""

    def __init__(self, max_legs: int = MAX_LEGS) -> None:
        self._max_legs = max(1, int(max_legs))
        self._legs: list[Leg] = []
        self._complete = False
        self._gate_open = False
        self._saw_gate = False
        self._truncated = False
        self._gate_note: str | None = None
        self._last_end: float | None = None
        # Tick-stream state, not story state: `_begin` never clears these.
        self._rendering = False
        self._started: float | None = None

    def story(self) -> Story | None:
        """Return the current or most recent run, or None before any run."""
        if not self._legs:
            return None
        total = (None if self._last_end is None
                 else max(0.0, self._last_end - self._legs[0].start))
        return Story(legs=tuple(self._legs), total_s=total, complete=self._complete,
                     ceremony=self._saw_gate, truncated=self._truncated,
                     gate_note=self._gate_note)

    def live_label(self, render: object) -> str:
        """Return the display label for the active render."""
        if not isinstance(render, dict) or not render.get("active"):
            return "render"
        leg = self._open_leg()
        if leg is not None and leg.kind == "proof":
            return leg.label + _PROOF_RANGE
        if leg is not None and leg.kind in ("cross", "render"):
            return leg.label
        # No open phase identifies this render. The ceremony flag also covers
        # the requested image after validation, so require an open gate before
        # labeling it as a gate render.
        return ("gate render"
                if self._gate_open and render.get("ceremony") is not False else "render")

    def observe(self, tick: object) -> None:
        """Fold one tick in. Malformed shapes are dropped, never raised on."""
        if not isinstance(tick, dict):
            return
        events = tick.get("events")
        for event in events if isinstance(events, list) else ():
            self._notice(event, tick)
        render = tick.get("render")
        self._render_edge(render if isinstance(render, dict) else {}, tick)

    def _notice(self, event: object, tick: dict) -> None:
        if not isinstance(event, dict) or event.get("kind") != "notice":
            return
        phase = event.get("phase")
        if phase not in _PHASES:
            return
        at = self._stamp(event.get("t"), tick)
        note = str(event.get("note") or "")
        if phase == "deferred":
            self._begin_if_needed(at)
            label = (note[: -len(_DEFERRED_TAIL)] if note.endswith(_DEFERRED_TAIL)
                     else note) or "deferred cost"
            self._add("bringup" if label.upper().startswith("NCCL") else "load", label, at)
        elif phase == "gate":
            # A second gate starts a new run: each run opens validation once,
            # and identity checks cannot nest.
            if self._saw_gate:
                self._begin()
            else:
                self._begin_if_needed(at)
            self._saw_gate = True
            self._gate_open = True
            self._add("gate", "gate setup", at)
        elif phase == "gate_proof":
            self._begin_if_needed(at)
            leg = self._open_leg()
            if leg is not None and leg.kind == "cross":
                return  # the cross check's own render: one leg, not two
            proofs = sum(1 for entry in self._legs if entry.kind == "proof")
            self._add("proof", f"gate proof {_proof_index(note, proofs + 1)}", at)
        elif phase == "gate_cross":
            self._begin_if_needed(at)
            self._add("cross", "cross-residency check", at)
        else:  # gate_done: a boundary, never a leg of its own
            self._gate_open = False
            self._gate_note = note or None
            self._close(at)

    def _render_edge(self, render: dict, tick: dict) -> None:
        active = bool(render.get("active"))
        started = _number(render.get("started")) if active else None
        # Consecutive validation and requested renders can both appear active
        # across polls. A changed start timestamp detects the transition even
        # if no poll observed the inactive interval.
        handed_over = (self._rendering and started is not None
                       and self._started is not None and started != self._started)
        if active and (not self._rendering or handed_over):
            if handed_over:
                self._render_end(render, tick)  # `last` still holds the one that ended
            self._render_start(render, tick)
        elif self._rendering and not active:
            self._render_end(render, tick)
        self._rendering = active
        self._started = started

    def _render_start(self, render: dict, tick: dict) -> None:
        at = self._stamp(render.get("started"), tick)
        # Use the tracker's explicit validation membership. False during an
        # open gate marks a new run after an unfinished check.
        member = render.get("ceremony")
        if self._gate_open and member is not False:
            self._begin_if_needed(at)
            if self._open_leg() is None:  # the proof notice never arrived
                self._add("proof", "gate render", at)
            return
        # A render outside the previous gate's membership starts a new run.
        # Retries remain members within the tracker's STALE_S interval.
        if self._gate_open or (self._saw_gate and member is False):
            self._begin()
        else:
            self._begin_if_needed(at)
        self._add("render", "render", at)

    def _render_end(self, render: dict, tick: dict) -> None:
        last = render.get("last")
        last = last if isinstance(last, dict) else {}
        at = self._stamp(last.get("ended_at"), tick)
        leg = self._open_leg()
        if leg is None:
            return
        if leg.kind != "render":
            return  # Gate duration includes gaps until the next phase notice.
        self._close(at, seconds=_number(last.get("wall_s")))
        self._complete = True

    def _open_leg(self) -> Leg | None:
        return self._legs[-1] if self._legs and self._legs[-1].seconds is None else None

    def _add(self, kind: str, label: str, at: float) -> None:
        if len(self._legs) >= self._max_legs:
            self._truncated = True
            return
        self._close(at)
        self._legs.append(Leg(kind=kind, label=label, start=at))

    def _close(self, at: float, seconds: float | None = None) -> None:
        leg = self._open_leg()
        if leg is None:
            return
        self._legs[-1] = replace(
            leg, seconds=max(0.0, at - leg.start) if seconds is None else max(0.0, seconds))
        self._last_end = at

    def _begin(self) -> None:
        self._legs = []
        self._complete = False
        self._gate_open = False
        self._saw_gate = False
        self._truncated = False
        self._gate_note = None
        self._last_end = None

    def _begin_if_needed(self, at: float) -> None:
        if self._complete or not self._legs or self._stale(at):
            self._begin()

    def _stale(self, at: float) -> bool:
        """Return whether inactivity has exceeded the run-expiry interval."""
        if self._gate_open or self._rendering:
            return False  # Active validation or rendering prevents expiry.
        quiet = max(self._legs[-1].start, self._last_end or 0.0)
        return at - quiet > STALE_S

    @staticmethod
    def _stamp(value: object, tick: dict) -> float:
        """Driver time when the driver sent it, tick time otherwise."""
        number = _number(value)
        return number if number is not None else (_number(tick.get("t")) or 0.0)


class StoryTrack:
    """Maintain live run phases and a separate view for paused playback.

    Consume one sample per poll. Cache the paused view through the cursor until
    its position or the history length changes, keeping run phases and render
    labels aligned with the other panels.
    """

    def __init__(self) -> None:
        self.live = RunStory()
        self._key: tuple[int, int] | None = None
        self._folded = RunStory()
        self.last_error: str | None = None

    def observe(self, tick: object) -> bool:
        """Fold one live tick; on an exception, set `last_error` and return False."""
        try:
            self.live.observe(tick)
        except Exception as exc:
            # Keep the dashboard and recording running, and show that the story
            # stopped. Only the exception's type name reaches the screen.
            self.last_error = type(exc).__name__
            return False
        self.last_error = None
        return True

    def view(self, ticks: list, upto: int | None) -> RunStory:
        """Return run phases through ring index ``upto`` (-1 is newest), or live state for None."""
        if upto is None or not ticks:
            return self.live
        key = (upto, len(ticks))
        if key != self._key:
            folded = RunStory()
            for tick in ticks[: len(ticks) + upto + 1]:
                folded.observe(tick)
            self._key, self._folded = key, folded
        return self._folded
