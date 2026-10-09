"""Aggregate each run's progress in dgxm top (tui/run_story.py).

The aggregator groups polled ticks into load, gate and render stages and keeps
the completed run visible until the next starts. This makes the cost of a
first-use gate distinct from a warm render.

Tests feed recorded tick shapes without a terminal or driver. Notice fixtures
come from first_render so changes to its phase and note strings exercise the
same contract the aggregator reads.
"""
from __future__ import annotations

import pytest

from dgx_monarch import first_render, telemetry
from dgx_monarch.tui.run_story import (
    MAX_LEGS,
    STALE_S,
    RunStory,
    StoryTrack,
    story_spans,
)


@pytest.fixture(autouse=True)
def _reset_notices():
    first_render.reset()
    yield
    first_render.reset()


@pytest.fixture
def emitted():
    """Everything first_render has published since the last call."""
    mark = telemetry.event_sequence()

    def take() -> list[dict]:
        nonlocal mark
        rows = [event for event in telemetry.events_tail(256)
                if event["seq"] > mark and event["kind"] == "notice"]
        if rows:
            mark = rows[-1]["seq"]
        return rows

    return take


def _tick(t: float, events=(), render=None) -> dict:
    return {"t": t, "events": list(events), "render": render or {"active": False}}


class _Run:
    """Ticks in the order a poller would have recorded them, on a fake clock.

    Every tick is kept, so a test that needs a recording (replay, scrubbing)
    reads `ticks` instead of building the same stream a second way.
    """

    def __init__(self, story, t: float = 1000.0):
        self.story, self.t, self._seq = story, t, 0
        self.ticks: list[dict] = []

    def feed(self, tick: dict) -> None:
        self.ticks.append(tick)
        self.story.observe(tick)

    def idle(self, seconds: float = 1.0) -> _Run:
        self.t += seconds
        self.feed(_tick(self.t))
        return self

    def notice(self, phase: str, note: str, gap: float = 1.0) -> _Run:
        self._seq += 1
        return self.notices(
            [{"seq": self._seq, "kind": "notice", "phase": phase, "note": note}], gap)

    def notices(self, events, gap: float = 1.0) -> _Run:
        """Feed real notices, restamped onto this run's clock."""
        for event in events:
            self.t += gap
            self.feed(_tick(self.t, [dict(event, t=self.t)]))
        return self

    def begin(self, ceremony: bool = False, model: str = "krea2") -> dict:
        render = {"active": True, "step": 0, "steps": 8, "model": model,
                  "started": self.t, "ceremony": ceremony}
        self.feed(_tick(self.t, render=render))
        return render

    def end(self, seconds: float, ceremony: bool = False, model: str = "krea2") -> _Run:
        self.t += seconds
        self.feed(_tick(self.t, render={"active": False, "last": {
            "model": model, "steps": 8, "wall_s": seconds, "ended_at": self.t,
            "ceremony": ceremony}}))
        return self

    def render(self, seconds: float, ceremony: bool = False) -> _Run:
        self.begin(ceremony=ceremony)
        return self.end(seconds, ceremony=ceremony)


def _labels(story) -> list[str]:
    return [leg.label for leg in story.legs]


def test_a_first_use_press_is_one_story_with_every_leg(emitted):
    """The tester's press: bring-up, cold load, the gate, then his image."""
    story = RunStory()
    run = _Run(story)

    first_render.nccl_deferred()
    first_render.load_deferred("load_model")
    run.notices(emitted())
    first_render.gate_started("unknown")
    run.notices(emitted())
    for _ in range(2):
        first_render.note_proof_render()
        run.notices(emitted())
        run.render(20.0, ceremony=True)
    first_render.cross_residency_check()
    run.notices(emitted())
    first_render.note_proof_render()          # the cross leg renders too
    run.notices(emitted())
    run.render(25.0, ceremony=True)
    first_render.gate_finished({"verdict": "PASS", "wall_s": 100.0})
    run.notices(emitted())
    run.render(44.6, ceremony=True)

    told = story.story()
    assert _labels(told) == ["NCCL bring-up", "cold load_model", "gate setup",
                             "gate proof 1", "gate proof 2",
                             "cross-residency check", "render"]
    assert told.complete is True
    assert told.ceremony is True
    assert told.truncated is False
    assert told.gate_note == "gate PASS in 100.0s, render starting"
    # The operator's own image is the last leg, and the driver's own wall time
    # is what it cost, not the poll interval that noticed it end.
    assert told.legs[-1].kind == "render"
    assert told.legs[-1].seconds == pytest.approx(44.6)
    # Contiguous by construction: every leg ends where the next begins, so the
    # legs account for the whole press and the total is their sum.
    assert told.total_s == pytest.approx(sum(leg.seconds for leg in told.legs))
    assert told.total_s == pytest.approx(116.6)


def test_a_warm_press_is_just_the_render():
    """Second press of the same combination: no gate, no cold load, one leg."""
    story = RunStory()
    _Run(story).render(42.0)

    told = story.story()
    assert _labels(told) == ["render"]
    assert told.complete is True
    assert told.ceremony is False
    assert told.total_s == pytest.approx(42.0)


def test_the_cross_check_and_its_own_render_are_one_leg():
    """The cross check's own render also publishes a proof notice; the two make one leg."""
    story = RunStory()
    run = _Run(story)
    run.notice("gate", "identity gate running (2-4 proof renders)")
    run.notice("gate_cross", "cross-residency check")
    run.notice("gate_proof", "proof render 3")
    run.render(25.0, ceremony=True)

    assert _labels(story.story()) == ["gate setup", "cross-residency check"]


def test_a_press_interrupted_mid_ceremony_keeps_what_it_paid():
    story = RunStory()
    run = _Run(story)
    run.notice("gate", "identity gate running (2-4 proof renders)")
    run.notice("gate_proof", "proof render 1")
    run.render(20.0, ceremony=True)
    run.idle(30.0)

    told = story.story()
    assert _labels(told) == ["gate setup", "gate proof 1"]
    assert told.complete is False
    assert told.legs[-1].seconds is None      # still open: nothing ended it

    # The next press is not a member of that dead ceremony, and the tracker
    # says so. The story starts over rather than growing a stale leg.
    run.render(42.0, ceremony=False)
    told = story.story()
    assert _labels(told) == ["render"]
    assert told.complete is True
    assert told.ceremony is False


def test_a_second_press_replaces_the_story():
    story = RunStory()
    run = _Run(story)
    run.notice("gate", "identity gate running (2-4 proof renders)")
    run.notice("gate_proof", "proof render 1")
    run.render(20.0, ceremony=True)
    run.notice("gate_done", "gate PASS in 30.0s, render starting")
    run.render(60.0, ceremony=True)
    assert len(story.story().legs) == 3

    run.idle(120.0)                            # the story survives idle
    assert len(story.story().legs) == 3

    run.render(42.0)
    told = story.story()
    assert _labels(told) == ["render"]
    assert told.gate_note is None
    assert told.total_s == pytest.approx(42.0)


def test_the_gate_hands_straight_to_the_operators_render():
    """One tick spans the hand-off, so `active` never reads false between them.

    The gate finishes its last proof render, decides, and starts the render it
    cleared inside the same call. A one-second poll seldom lands in that gap,
    so the tracker's own start stamp marks the edge, not the poll.
    """
    story = RunStory()
    run = _Run(story)
    run.notice("gate", "identity gate running (2-4 proof renders)")
    run.notice("gate_proof", "proof render 1")
    run.begin(ceremony=True)
    run.t += 20.0
    run.feed(_tick(
        run.t,
        [{"kind": "notice", "phase": "gate_done", "t": run.t,
          "note": "gate PASS in 21.0s, render starting"}],
        render={"active": True, "step": 0, "steps": 8, "model": "krea2",
                "started": run.t, "ceremony": True,
                "last": {"model": "krea2", "wall_s": 20.0, "ended_at": run.t,
                         "ceremony": True}}))
    # The image the press was for is its own leg, not a proof: the tracker
    # flags it a ceremony member only because it spends the closed gate's claim.
    assert story.live_label({"active": True, "ceremony": True}) == "render"
    run.end(44.0, ceremony=True)

    told = story.story()
    assert _labels(told) == ["gate setup", "gate proof 1", "render"]
    assert told.complete is True
    assert told.legs[-1].seconds == pytest.approx(44.0)


def test_a_press_that_died_does_not_bill_the_next_one():
    """The cold load raised, so the render it was for never came.

    Nothing in the tick stream announces a failed prompt, so the story reads
    the only evidence there is: a press nothing has moved for STALE_S is over,
    and the next press does not inherit its open leg or its total.
    """
    story = RunStory()
    run = _Run(story)
    run.notice("deferred", "cold load_model on first render")
    run.idle(STALE_S + 60.0)
    run.render(42.0)

    told = story.story()
    assert _labels(told) == ["render"]
    assert told.total_s == pytest.approx(42.0)


def test_a_cold_load_that_takes_minutes_is_still_the_same_press():
    """The other side of the same rule: a slow leg is not a dead press."""
    story = RunStory()
    run = _Run(story)
    run.notice("deferred", "cold load_model on first render")
    run.idle(120.0)
    run.render(42.0)

    told = story.story()
    assert _labels(told) == ["cold load_model", "render"]
    assert told.total_s == pytest.approx(162.0)


def test_a_refused_gate_does_not_bill_the_next_press():
    """A FAIL can refuse the render outright, so the press ends at the gate.

    The next press arrives well inside STALE_S, so time says nothing. The
    tracker does: this render is no member of that ceremony.
    """
    story = RunStory()
    run = _Run(story)
    run.notice("gate", "identity gate running (2-4 proof renders)")
    run.notice("gate_proof", "proof render 1")
    run.render(20.0, ceremony=True)
    run.notice("gate_done", "gate FAIL in 21.0s, optimized residency off")
    run.idle(60.0)
    run.render(42.0)

    told = story.story()
    assert _labels(told) == ["render"]
    assert told.gate_note is None
    assert told.total_s == pytest.approx(42.0)


def test_the_stale_window_matches_the_trackers_ceremony_claim():
    """Two rules answer one question: is this render the one that press was for?

    The tracker expires a closed gate's claim after the same window the story
    calls a press dead. If the two drift apart, a render inside one window but
    outside the other reads as a member of a story that is over.
    """
    assert STALE_S == telemetry._CEREMONY_CLAIM_S


def test_a_scrubbed_cursor_reads_the_story_its_own_tick_knew():
    """Paused on an older tick, the story block and live label show that tick, like the rest of the screen."""
    track = StoryTrack()
    run = _Run(track)
    run.notice("gate", "identity gate running (2-4 proof renders)")
    run.notice("gate_proof", "proof render 1")
    proof = run.begin(ceremony=True)
    run.end(20.0, ceremony=True)
    run.notice("gate_done", "gate PASS in 22.0s, render starting")
    image = run.begin(ceremony=True)
    run.end(44.0, ceremony=True)
    recording = run.ticks

    assert _labels(track.live.story()) == ["gate setup", "gate proof 1", "render"]
    assert track.view(recording, None) is track.live

    at_proof = track.view(recording, -5)          # cursor on the proof render
    assert at_proof.live_label(proof) == "gate proof 1 of 2-4"
    assert "running" in "".join(text for text, _s in story_spans(at_proof.story()))
    assert _labels(at_proof.story()) == ["gate setup", "gate proof 1"]

    at_image = track.view(recording, -2)          # cursor on the operator's image
    assert at_image.live_label(image) == "render"
    assert _labels(at_image.story()) == ["gate setup", "gate proof 1", "render"]
    assert track.view(recording, -2) is at_image  # one fold per cursor position


def test_the_leg_list_is_capped():
    story = RunStory(max_legs=3)
    run = _Run(story)
    for index in range(1, 9):
        run.notice("gate_proof", f"proof render {index}")

    told = story.story()
    assert _labels(told) == ["gate proof 1", "gate proof 2", "gate proof 3"]
    assert told.truncated is True

    default = RunStory()
    spam = _Run(default)
    for index in range(1, MAX_LEGS + 6):
        spam.notice("gate_proof", f"proof render {index}")
    assert len(default.story().legs) == MAX_LEGS
    assert default.story().truncated is True


def test_story_track_surfaces_a_narrator_failure_without_raising():
    track = StoryTrack()

    class BrokenStory:
        def observe(self, _tick):
            raise RuntimeError("private narration detail")

    track.live = BrokenStory()

    assert track.observe({"t": 1.0}) is False
    assert track.last_error == "RuntimeError"
    assert "private narration detail" not in track.last_error


def test_malformed_ticks_are_dropped_never_raised_on():
    """Same promise tui/data.py makes: a bad shape costs the block, not the app."""
    story = RunStory()
    for junk in (None, "tick", 7, {}, {"events": "not-a-list"},
                 {"events": [None, 7, {"kind": "notice"}, {"kind": "swap"}]},
                 {"render": "not-a-dict"}, {"t": float("nan")},
                 {"t": None, "events": [{"kind": "notice", "phase": "gate_proof",
                                         "note": None, "t": float("inf")}]}):
        story.observe(junk)

    told = story.story()
    assert _labels(told) == ["gate proof 1"]   # only the one real notice landed
    assert told.legs[0].start == 0.0           # no clock of its own to fall back on


def test_the_render_box_names_the_live_leg():
    story = RunStory()
    run = _Run(story)
    assert story.live_label({"active": True}) == "render"
    assert story.live_label({"active": False}) == "render"
    assert story.live_label("not-a-dict") == "render"

    run.notice("gate", "identity gate running (2-4 proof renders)")
    run.notice("gate_proof", "proof render 1")
    live = run.begin(ceremony=True)
    assert story.live_label(live) == "gate proof 1 of 2-4"
    run.end(20.0, ceremony=True)

    run.notice("gate_cross", "cross-residency check")
    live = run.begin(ceremony=True)
    assert story.live_label(live) == "cross-residency check"
    run.end(25.0, ceremony=True)

    run.notice("gate_done", "gate PASS in 60.0s, render starting")
    live = run.begin(ceremony=True)
    assert story.live_label(live) == "render"


def test_the_block_reads_as_one_line():
    assert story_spans(None) == (("no run recorded yet", "dim"),)

    story = RunStory()
    run = _Run(story)
    run.notice("deferred", "NCCL bring-up on first render")
    run.idle(3.0)
    run.render(42.0)
    line = "".join(text for text, _style in story_spans(story.story()))
    assert "this run" in line
    assert "NCCL bring-up" in line
    assert "render 42.0s" in line
    assert "total 45s" in line

    run.begin()
    assert "running" in "".join(text for text, _style in story_spans(story.story()))
