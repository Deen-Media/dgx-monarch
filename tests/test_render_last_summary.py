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
