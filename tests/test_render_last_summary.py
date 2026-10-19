"""A finished render keeps its summary, and the gate chip keeps its scope.

The tracker retains the last render's model, steps, wall time and end stamp,
and its ceremony flag tells a cold first render from a warm one.

Pure telemetry: no comfy import, no route, no browser.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch import first_render, telemetry
from dgx_monarch.telemetry import RenderProgress

REPO = Path(__file__).parents[1]
PANEL = REPO / "web" / "js" / "dgx_monarch_panel.js"
TIMELINE = REPO / "web" / "js" / "dgx_monarch_timeline.js"
TIMELINE_MODEL = REPO / "web" / "js" / "dgx_monarch_timeline_model.js"


def _run(progress: RenderProgress, model: str, steps: int) -> None:
    token = progress.start(steps, {"model": model}, token=object())
    for step in range(1, steps + 1):
        progress.step(step, steps)
    progress.finish(token)


def test_a_finished_render_retains_model_steps_wall_and_end_stamp():
    progress = RenderProgress()
    _run(progress, "krea2.safetensors", 4)

    snapshot = progress.snapshot()
    assert snapshot["active"] is False
    last = snapshot["last"]
    assert last["model"] == "krea2.safetensors"
    assert last["steps"] == 4
    assert last["wall_s"] == snapshot["last_wall_s"]
    assert last["wall_s"] >= 0
    assert last["ended_at"] > 0
    assert last["ceremony"] is False


_FACTS = ("topology", "auto_row", "megapixels", "quant", "attention", "sec_per_step",
          "family", "cfg", "keeps_uncond", "evidence_row", "skipped_cfg_row", "batch",
          "sample_s", "mean_step_s",
          # The kernels the model pins per block, beside "attention" (issue #30).
          "attention_pins", "attention_pins_ran", "attention_pins_dropped",
          # A re-attach inside the render after a Worker restart (issue #47).
          "reattach", "reattach_s")


def test_the_summary_keeps_where_the_render_ran_and_its_last_step_rate():
    # render_submit passes these at start, so readers of /dgxm/telemetry and
    # the panel can show where the auto table's choice is slow.
    progress = RenderProgress()
    meta = {"model": "krea2.safetensors", "topology": "uly2", "auto_row": 11,
            "megapixels": 2.359, "quant": "fp8", "attention": "SAGE_AUTO"}
    token = progress.start(3, meta, token=object())
    progress.step(1, 3)
    progress.step(2, 3)
    running = progress.snapshot()
    assert {key: running[key] for key in meta} == meta
    progress.step(3, 3)
    final_rate = progress.snapshot()["sec_per_step"]
    progress.finish(token)

    last = progress.snapshot()["last"]
    assert {key: last[key] for key in meta} == meta
    assert last["sec_per_step"] == final_rate
    assert set(last) == {"model", "steps", "wall_s", "ended_at", "ceremony", *_FACTS}


def test_the_summary_keeps_the_slowest_rank_sampler_wall_and_its_mean_step():
    # render_result notes the slowest rank's sample_s before the finish, so
    # the record carries a mean step comparable to BENCHMARKS, free of the
    # load and Gate time a cold wall_s holds.
    progress = RenderProgress()
    token = progress.start(8, {"model": "krea2.safetensors", "topology": "cfg2",
                               "auto_row": 15, "evidence_row": 15, "cfg": 1.0,
                               "keeps_uncond": True, "family": "krea2", "batch": 1},
                           token=object())
    for step in range(1, 9):
        progress.step(step, 8)
    progress.note_sample(31.39, token=token)
    progress.finish(token)
    last = progress.snapshot()["last"]
    assert (last["sample_s"], last["mean_step_s"]) == (31.39, 3.924)
    assert (last["auto_row"], last["evidence_row"], last["keeps_uncond"]) == (15, 15, True)
    _run(progress, "next.safetensors", 2)
    assert progress.snapshot()["last"]["sample_s"] is None  # never inherited


def test_an_unkeyed_sampler_wall_belongs_to_the_only_render_in_flight():
    progress = RenderProgress()
    token = progress.start(4, {"model": "m"}, token=object())
    progress.note_sample(10.0)
    progress.finish(token)
    assert progress.snapshot()["last"]["mean_step_s"] == 2.5

    # With two in flight an unkeyed wall has no owner, so it is dropped.
    first = progress.start(4, {"model": "a"}, token=object())
    second = progress.start(4, {"model": "b"}, token=object())
    progress.note_sample(10.0)
    progress.finish(first)
    progress.finish(second)
    assert progress.snapshot()["last"]["sample_s"] is None


def test_an_overlapping_render_never_publishes_another_renders_sampler_wall():
    # Pipelined submissions overlap. Render a reports its wall and finishes;
    # render b, whose facts the record carries, is cancelled before its result
    # is collected. The record must not show a's wall under b's settings, nor
    # a's wall over b's step count.
    progress = RenderProgress()
    first = progress.start(8, {"model": "a", "topology": "cfg2", "auto_row": 15},
                           token=object())
    second = progress.start(20, {"model": "b", "topology": "uly2", "auto_row": 11},
                            token=object())
    progress.note_sample(31.39, token=first)
    progress.finish(first)
    progress.finish(second)
    last = progress.snapshot()["last"]
    assert (last["model"], last["auto_row"], last["steps"]) == ("b", 11, 20)
    assert (last["sample_s"], last["mean_step_s"]) == (None, None)


def test_a_sampler_wall_keeps_its_own_renders_step_count():
    # Render b's mean divides by b's 20 steps even though a later start, whose
    # facts the record carries, ran 8.
    progress = RenderProgress()
    first = progress.start(20, {"model": "a"}, token=object())
    second = progress.start(8, {"model": "b"}, token=object())
    progress.note_sample(40.0, token=first)
    progress.note_sample(16.0, token=second)
    progress.finish(second)
    progress.finish(first)
    last = progress.snapshot()["last"]
    assert (last["model"], last["steps"]) == ("b", 8)
    assert (last["sample_s"], last["mean_step_s"]) == (16.0, 2.0)


def test_a_late_sampler_wall_for_a_finished_render_is_dropped():
    progress = RenderProgress()
    first = progress.start(4, {"model": "a"}, token=object())
    progress.finish(first)
    progress.note_sample(10.0, token=first)
    _run(progress, "b", 2)
    assert progress.snapshot()["last"]["sample_s"] is None


def test_the_rank_results_hand_the_tracker_their_slowest_sampler_wall(monkeypatch):
    from dgx_monarch.nodes import render_result

    tracker = RenderProgress()
    monkeypatch.setattr(telemetry, "render_progress", tracker)
    monkeypatch.setattr(render_result, "verify_cross_rank_signatures", lambda *_args: None)
    monkeypatch.setattr(render_result, "read_latent_result", lambda value: value)
    token = tracker.start(4, {"model": "m"}, token=object())
    import torch

    render_result._finish_render(
        [{"dp_rank": 0, "rank": 0, "latent": torch.zeros(1, 4, 8, 8), "sample_s": 12.0},
         {"rank": 1, "sample_s": 12.5}],
        SimpleNamespace(dp=1), {"samples": torch.zeros(1, 4, 8, 8)}, telemetry_token=token)
    tracker.finish(token)
    assert tracker.snapshot()["last"]["sample_s"] == 12.5
    assert tracker.snapshot()["last"]["mean_step_s"] == 3.125


def test_a_start_without_facts_records_each_as_unknown():
    # Fleet's receiver and older callers start with no facts; the record keeps
    # one shape and says unknown rather than inherit a previous render's.
    progress = RenderProgress()
    token = progress.start(2, {"model": "a", "topology": "uly2", "auto_row": 11},
                           token=object())
    progress.step(1, 2)
    progress.finish(token)
    _run(progress, "b", 0)
    last = progress.snapshot()["last"]
    assert last["model"] == "b"
    assert all(last[key] is None for key in _FACTS)


class _Clock:
    """A settable wall clock for the tracker, so step times are exact.

    Times count from a nonzero base: a start stamped 0.0 reads as no stamp.
    """

    def __init__(self, monkeypatch):
        self.now = 1000.0
        monkeypatch.setattr(telemetry, "time", SimpleNamespace(time=lambda: self.now))

    def at(self, seconds: float) -> None:
        self.now = 1000.0 + seconds


def test_a_second_dp_leader_repeating_a_step_does_not_shrink_the_rate(monkeypatch):
    # In a dp2 render each dp group's result leader sends every step on the
    # same port. The second copy lands milliseconds after the first; timing it
    # would record that gap as the render's seconds per step.
    clock = _Clock(monkeypatch)
    progress = RenderProgress()
    token = progress.start(4, {"model": "m", "topology": "dp2"}, token=object())
    live = []
    for step in range(1, 5):
        for leader in range(2):
            clock.at(2.0 * step + 0.01 * leader)
            progress.step(step, 4, token=token)
            live.append(progress.snapshot()["sec_per_step"])
    clock.at(8.05)
    progress.finish(token)

    assert live == [2.0] * 8
    assert progress.snapshot()["last"]["sec_per_step"] == 2.0


def test_interleaved_fleet_renders_each_time_their_own_steps(monkeypatch):
    # A Fleet wave runs several renders at once, one token each, and their
    # steps interleave on the one tracker. Job a steps every 2 s from t=0 and
    # job b every 3 s from t=0.5; no reading may be the gap between them.
    clock = _Clock(monkeypatch)
    progress = RenderProgress()
    clock.at(0.0)
    job_a = progress.start(4, token=object())
    clock.at(0.5)
    job_b = progress.start(4, token=object())
    steps = sorted([(2.0 * k, job_a, k) for k in range(1, 5)]
                   + [(0.5 + 3.0 * k, job_b, k) for k in range(1, 5)])
    live = []
    for at, job, step in steps:
        clock.at(at)
        progress.step(step, 4, token=job)
        live.append((job is job_a, progress.snapshot()["sec_per_step"]))
        if job is job_a and step == 4:
            progress.finish(job_a)
    clock.at(12.6)
    progress.finish(job_b)

    assert {rate for is_a, rate in live if is_a} == {2.0}
    assert {rate for is_a, rate in live if not is_a} == {3.0}
    # Job b stepped last, so the record carries its rate.
    assert progress.snapshot()["last"]["sec_per_step"] == 3.0


def test_two_fleet_jobs_stepping_together_record_the_step_time(monkeypatch):
    clock = _Clock(monkeypatch)
    progress = RenderProgress()
    jobs = [progress.start(4, token=object()) for _ in range(2)]
    for step in range(1, 5):
        for offset, job in enumerate(jobs):
            clock.at(2.0 * step + 0.02 * offset)
            progress.step(step, 4, token=job)
    clock.at(8.1)
    for job in jobs:
        progress.finish(job)

    assert progress.snapshot()["last"]["sec_per_step"] == 2.0


def test_overlapping_renders_record_the_last_starts_facts_as_they_do_its_model():
    progress = RenderProgress()
    first = progress.start(4, {"model": "a", "topology": "uly2", "auto_row": 11},
                           token=object())
    second = progress.start(4, {"model": "b", "topology": "cfg2", "auto_row": 20},
                            token=object())
    progress.finish(first)
    progress.finish(second)
    last = progress.snapshot()["last"]
    assert (last["model"], last["topology"], last["auto_row"]) == ("b", "cfg2", 20)


def test_the_next_render_replaces_the_retained_summary_and_survives_its_start():
    progress = RenderProgress()
    _run(progress, "first.safetensors", 2)
    first = progress.snapshot()["last"]

    token = progress.start(6, {"model": "second.safetensors"}, token=object())
    # The previous render's summary stays readable while the next one runs.
    assert progress.snapshot()["last"] == first
    progress.finish(token)

    last = progress.snapshot()["last"]
    assert last["model"] == "second.safetensors"
    assert last["steps"] == 6
    assert last["ended_at"] >= first["ended_at"]


def test_the_ceremony_flag_covers_the_proof_legs_and_the_render_they_cleared():
    progress = RenderProgress()
    progress.note_ceremony()
    _run(progress, "cold.safetensors", 2)          # a gate proof leg
    assert progress.snapshot()["last"]["ceremony"] is True

    progress.close_ceremony()
    _run(progress, "cold.safetensors", 20)         # the operator's own render
    assert progress.snapshot()["last"]["ceremony"] is True

    _run(progress, "cold.safetensors", 20)         # the warm render after it
    assert progress.snapshot()["last"]["ceremony"] is False


def test_an_active_ceremony_render_says_so_while_it_runs():
    progress = RenderProgress()
    progress.note_ceremony()
    progress.close_ceremony()
    progress.start(20, {"model": "cold.safetensors"}, token=object())

    assert progress.snapshot()["ceremony"] is True


def test_a_ceremony_on_another_thread_is_not_this_threads_first_render():
    progress = RenderProgress()
    progress.note_ceremony()
    seen = []

    def unrelated():
        token = progress.start(4, {"model": "warm.safetensors"}, token=object())
        seen.append(progress.snapshot()["ceremony"])
        progress.finish(token)

    thread = threading.Thread(target=unrelated)
    thread.start()
    thread.join(timeout=5)

    assert seen == [False]
    assert progress.snapshot()["last"]["ceremony"] is False


def test_a_plain_render_overlapping_a_ceremony_never_inherits_its_flag():
    # Only a ceremony member carries the flag (RenderProgress.start): a Fleet
    # render that overlaps a gate must not inherit it.
    progress = RenderProgress()
    progress.note_ceremony()
    ceremony_token = progress.start(2, {"model": "cold.safetensors"}, token=object())
    assert progress.snapshot()["ceremony"] is True

    plain: dict = {}

    def fleet_render():
        plain["token"] = progress.start(4, {"model": "warm.safetensors"}, token=object())
        plain["seen_while_gate_runs"] = progress.snapshot()["ceremony"]

    thread = threading.Thread(target=fleet_render)
    thread.start()
    thread.join(timeout=5)

    # The display flag stays up while the ceremony member is active...
    assert plain["seen_while_gate_runs"] is True
    progress.close_ceremony()
    progress.finish(ceremony_token)
    # ...and drops when it finishes, though the plain render still runs.
    assert progress.snapshot()["ceremony"] is False
    progress.finish(plain["token"])
    # The drained bundle contained the ceremony, so its one record says so.
    assert progress.snapshot()["last"]["ceremony"] is True

    finished = []

    def later_fleet_render():
        token = progress.start(4, {"model": "warm.safetensors"}, token=object())
        finished.append(progress.snapshot()["ceremony"])
        progress.finish(token)

    thread = threading.Thread(target=later_fleet_render)
    thread.start()
    thread.join(timeout=5)
    # Nothing leaks past the bundle into an unrelated later render.
    assert finished == [False]
    assert progress.snapshot()["last"]["ceremony"] is False


def test_a_pipelined_render_after_the_claim_was_spent_is_not_a_ceremony():
    # On one thread: the operator's render spends the one-shot claim, and a
    # pipelined submission that overlaps it must not inherit the flag.
    progress = RenderProgress()
    progress.note_ceremony()
    progress.close_ceremony()

    operator = progress.start(20, {"model": "cold.safetensors"}, token=object())
    assert progress.snapshot()["ceremony"] is True
    pipelined = progress.start(20, {"model": "cold.safetensors"}, token=object())

    progress.finish(operator)
    # The claim's render is gone; the pipelined one runs unflagged.
    assert progress.snapshot()["ceremony"] is False
    progress.finish(pipelined)
    assert progress.snapshot()["last"]["ceremony"] is True

    _run(progress, "cold.safetensors", 20)
    assert progress.snapshot()["last"]["ceremony"] is False


def test_an_inner_gate_finishing_first_leaves_the_outer_window_open():
    progress = RenderProgress()
    progress.note_ceremony()
    progress.note_ceremony()

    progress.close_ceremony()
    _run(progress, "cold.safetensors", 2)          # still inside the outer gate
    assert progress.snapshot()["last"]["ceremony"] is True

    progress.close_ceremony()
    _run(progress, "cold.safetensors", 20)         # the render both gates cleared
    assert progress.snapshot()["last"]["ceremony"] is True

    _run(progress, "cold.safetensors", 20)
    assert progress.snapshot()["last"]["ceremony"] is False


def test_a_gate_that_refused_the_render_lets_its_claim_expire(monkeypatch):
    # The FSDP proof path raises instead of rendering, so nothing spends the
    # claim. Hours later it must not label a warm render a first render.
    monkeypatch.setattr(telemetry, "_CEREMONY_CLAIM_S", 0.0)
    progress = RenderProgress()
    progress.note_ceremony()
    progress.close_ceremony()

    _run(progress, "warm.safetensors", 20)
    assert progress.snapshot()["last"]["ceremony"] is False


def test_reset_forgets_a_window_the_gate_left_open(monkeypatch):
    tracker = RenderProgress()
    monkeypatch.setattr(telemetry, "render_progress", tracker)
    first_render.gate_started("unknown")
    first_render.reset()

    _run(tracker, "warm.safetensors", 20)
    assert tracker.snapshot()["last"]["ceremony"] is False


def test_the_snapshot_publishes_no_private_bookkeeping():
    progress = RenderProgress()
    progress.note_ceremony()
    token = progress.start(2, {"model": "m"}, token=object())
    assert not [key for key in progress.snapshot() if key.startswith("_")]
    progress.finish(token)
    assert not [key for key in progress.snapshot() if key.startswith("_")]


def test_the_panel_hides_an_all_pass_chip_only_after_a_plain_render():
    source = PANEL.read_text()
    gates = source[source.index("    // gates"):source.index("    // recent events")]
    # A FAIL or a waiver is signal: the quiet branch is reachable only when
    # every verdict is a PASS, no waiver exists, and no ceremony ran.
    assert 'verdict === "PASS"' in gates
    assert "!waivers" in gates
    assert "!ceremonyRender" in gates
    assert "gates for this render: " in gates


def test_the_timeline_is_a_passive_reader_of_comfys_own_event_stream():
    source = TIMELINE.read_text()
    for event in ("execution_start", "executing", "execution_success",
                  "execution_error", "execution_interrupted"):
        assert event in source
    # No route, no poll, no write: the recorder only listens.
    assert "fetchApi" not in source
    assert "setInterval" not in source
    for text in (source, TIMELINE_MODEL.read_text()):
        assert "\u2014" not in text  # em dash
        assert "\u2013" not in text  # en dash
    assert 'from "./dgx_monarch_timeline_model.js"' in source
    panel = PANEL.read_text()
    assert "startTimelineRecording();" in panel
    render_body = panel[panel.index("render: (container)"):panel.index("destroy: ()")]
    assert "startTimelineRecording" not in render_body


def test_the_collapsed_block_survives_the_poll_that_redraws_the_panel():
    panel = PANEL.read_text()
    # As in the consent zone: one <details> element for the tab's lifetime,
    # moved by replaceChildren instead of rebuilt. A fresh el("details", ...)
    # per poll would close the block every 2.5 s.
    assert panel.count('el("details"') == 1
    zone = panel[panel.index("function ensureDetailsZone()"):panel.index("async function refresh")]
    assert "if (detailsZone) return;" in zone
    assert 'el("details"' in zone
    assert "detailsTimeline.replaceChildren(" in zone
    assert "detailsLedger.textContent =" in zone


def _drive_recorder(script: str):
    """Run the real recorder module under node and return what it printed."""
    node = shutil.which("node")
    if node is None:  # pragma: no cover - environment dependent
        pytest.skip("node not installed; JS recorder check skipped")
    driver = (
        "import { createTimelineRecorder } from "
        f"{json.dumps(TIMELINE_MODEL.as_uri())};\n"
        "let clock = 0;\n"
        "const recorder = createTimelineRecorder({ clock: () => (clock += 500) });\n"
        + script
        + "process.stdout.write(JSON.stringify(recorder.last()));\n"
    )
    result = subprocess.run(
        [node, "--input-type=module", "-e", driver],
        capture_output=True, text=True, timeout=60, check=False,
    )
    if result.returncode != 0:  # pragma: no cover - surfaced only on a real break
        raise AssertionError(f"node driver failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def test_the_recorder_reads_the_detail_shape_comfy_actually_sends():
    # ComfyUI dispatches `executing` with the node id as the detail, not the
    # message object, and null when the last node is done. A recorder that read
    # detail.node would record nothing and raise no error.
    recorded = _drive_recorder(
        'recorder.open("abc");\n'
        'recorder.executing("4");\n'
        'recorder.executing("7");\n'
        "recorder.executing(null);\n"
        "recorder.end();\n"
    )
    assert recorded["truncated"] is False
    assert [row["label"] for row in recorded["rows"]] == ["node 4", "node 7"]
    assert [row["seconds"] for row in recorded["rows"]] == [0.5, 0.5]


def test_the_recorder_survives_a_missed_execution_start_and_a_message_shaped_detail():
    recorded = _drive_recorder(
        'recorder.executing("11");\n'                       # no execution_start
        'recorder.executing({ node: "12" });\n'             # a shape change
        "recorder.end();\n"
    )
    assert [row["label"] for row in recorded["rows"]] == ["node 11", "node 12"]


def test_a_prompt_that_ran_no_nodes_leaves_the_previous_timeline_alone():
    recorded = _drive_recorder(
        'recorder.open("first");\n'
        'recorder.executing("3");\n'
        "recorder.end();\n"
        'recorder.open("cached");\n'
        "recorder.executing(null);\n"
        "recorder.end();\n"
    )
    assert [row["label"] for row in recorded["rows"]] == ["node 3"]
