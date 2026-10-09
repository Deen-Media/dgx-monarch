"""The Fleet wave's per-job rows, from the driver's ledger to the sidebar.

A Fleet render fans one job per GPU, and the sidebar timeline records per node,
so without these rows a wave shows one row for the whole node and nothing about
which box ran which prompt. This file covers the ledger that records each job,
the telemetry render block that carries it, and the web files that draw it.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

import pytest

from dgx_monarch import telemetry_fleet
from dgx_monarch.telemetry import RenderProgress

REPO = Path(__file__).resolve().parents[1]
PANEL = REPO / "web" / "js" / "dgx_monarch_panel.js"
TIMELINE = REPO / "web" / "js" / "dgx_monarch_timeline.js"
TIMELINE_MODEL = REPO / "web" / "js" / "dgx_monarch_timeline_model.js"

JOB_FIELDS = {"job", "prompt", "host", "rank", "box", "wall_s",
              "started_s", "ended_s"}


class _Host:
    def __init__(self, name):
        self.name = name


class _Config:
    """The one field the ledger reads off the driver's cluster config."""

    def __init__(self, names):
        self.hosts = [_Host(name) for name in names]


def _wave(count=2, gpus_per_host=1, texts=None, names=None):
    ledger = telemetry_fleet.FleetJobs()
    jobs = [
        {"text": None if texts is None else texts[index], "seed": index}
        for index in range(count)
    ]
    ledger.open_wave(jobs, gpus_per_host,
                     None if names is None else _Config(names))
    return ledger


def test_a_recorded_job_publishes_exactly_the_fields_the_panel_reads():
    ledger = _wave(texts=["a cat", "a dog"])
    ledger.record_job(0, 0, {"host": "worker-a", "rank": 0, "sample_s": 4.126})
    ledger.record_job(1, 1, {"host": "worker-b", "rank": 1, "sample_s": 8.25})

    block = ledger.snapshot()
    assert block["count"] == 2
    assert block["truncated"] is False
    assert [set(row) for row in block["jobs"]] == [JOB_FIELDS, JOB_FIELDS]
    assert [row["box"] for row in block["jobs"]] == ["box 1", "box 2"]
    assert [row["wall_s"] for row in block["jobs"]] == [4.13, 8.25]
    assert block["jobs"][0]["prompt"] == telemetry_fleet.prompt_digest("a cat")
    # A digest, not the operator's text: this block rides an HTTP route.
    assert "a cat" not in json.dumps(block)


def test_a_dispatched_job_opens_its_row_before_any_answer_arrives():
    # The row the sidebar reads after a Stop: the driver knows the box it sent
    # the job to and when, and nothing else about it yet.
    ledger = _wave(texts=["a cat", "a dog"])
    ledger.start_job(0, 0)
    ledger.start_job(1, 1)

    first, second = ledger.snapshot()["jobs"]
    assert set(first) == JOB_FIELDS
    assert (first["box"], first["host"], first["wall_s"]) == ("box 1", None, None)
    assert first["prompt"] == telemetry_fleet.prompt_digest("a cat")
    assert first["started_s"] >= 0 and first["ended_s"] is None
    assert (second["box"], second["rank"]) == ("box 2", 1)


def test_an_answer_closes_the_row_its_dispatch_opened():
    ledger = _wave(count=2)
    ledger.start_job(0, 0)
    ledger.start_job(1, 1)
    ledger.record_job(0, 0, {"host": "worker-a", "sample_s": 4.0})

    first, second = ledger.snapshot()["jobs"]
    # One row for the job, not a second one beside the dispatch row.
    assert [row["job"] for row in ledger.snapshot()["jobs"]] == [0, 1]
    assert first["host"] == "worker-a"
    assert first["started_s"] <= first["ended_s"]
    # The job that never came back keeps its row and stays open.
    assert (second["host"], second["ended_s"]) == (None, None)


def test_an_answer_with_no_dispatch_row_still_publishes_one():
    # The ledger is best effort at both ends: a dispatch row it never got to
    # open costs the start time, never the row.
    ledger = _wave(count=1)
    ledger.record_job(0, 0, {"host": "worker-a", "sample_s": 1.0})

    row, = ledger.snapshot()["jobs"]
    assert row["started_s"] is None
    assert row["ended_s"] >= 0


def test_the_rows_come_back_in_dispatch_order_whatever_order_answers_did():
    # The node publishes a wave's answers in prompt order
    # (nodes/fleet_cleanup.py), but the ledger sorts by job itself, so the
    # rows read in dispatch order whatever order the answers reach it in.
    ledger = _wave(count=3)
    for index in (2, 0, 1):
        ledger.record_job(index, index, {"host": f"worker-{index}"})

    assert [row["job"] for row in ledger.snapshot()["jobs"]] == [0, 1, 2]


def test_a_second_answer_does_not_reopen_a_closed_row():
    ledger = _wave(count=1)
    ledger.start_job(0, 0)
    ledger.record_job(0, 0, {"host": "worker-a", "sample_s": 1.0})
    ledger.record_job(0, 0, {"host": "worker-b", "sample_s": 9.0})

    row, = ledger.snapshot()["jobs"]
    assert (row["host"], row["wall_s"]) == ("worker-a", 1.0)


def test_a_dispatch_recorded_with_no_wave_open_is_dropped():
    ledger = telemetry_fleet.FleetJobs()
    ledger.start_job(0, 0)

    assert ledger.snapshot() is None


def test_two_gpus_on_one_box_put_both_of_its_ranks_under_one_label():
    ledger = _wave(count=4, gpus_per_host=2)
    for index in range(4):
        ledger.record_job(index, index, {"host": f"worker-{index // 2}"})

    assert [row["box"] for row in ledger.snapshot()["jobs"]] == [
        "box 1", "box 1", "box 2", "box 2",
    ]


def test_a_box_label_falls_back_to_a_short_name_and_never_an_address():
    # With no usable rank or gpus_per_host there is no arithmetic to do, so the
    # host's own first label stands in. Nothing on either path prints a full address.
    assert telemetry_fleet.box_label(0, 0, "worker-a.example.test") == "worker-a"
    assert telemetry_fleet.box_label(None, 2, "worker-a.example.test") == "worker-a"
    assert telemetry_fleet.box_label(True, 2, "worker-a") == "worker-a"
    assert telemetry_fleet.box_label(1, 1, None) == "box 2"
    assert telemetry_fleet.box_label(None, 0, None) is None
    assert len(telemetry_fleet.box_label(None, 0, "w" * 200)) == 32


def test_a_healthy_two_box_wave_marks_nothing():
    # A fleet job runs world-1 and the driver sets every one of those workers
    # up as rank 0 of its own mesh, so every reply says rank 0. A marker that
    # compares that with the driver's pair-wide rank fires on every job sent to
    # a box other than the first, in a healthy wave (docs/VALIDATION.md, Fleet
    # job row marker record, 2026-09-06).
    ledger = _wave(count=4, names=["worker-a", "worker-b"])
    for index in range(4):
        ledger.record_job(index, index % 2, {
            "host": ["worker-a", "worker-b"][index % 2], "rank": 0})

    jobs = ledger.snapshot()["jobs"]
    assert [row["box"] for row in jobs] == ["box 1", "box 2", "box 1", "box 2"]
    assert all("host_reported" not in row for row in jobs)


def test_a_worker_answering_from_another_box_is_marked():
    ledger = _wave(count=2, names=["worker-a", "worker-b"])
    ledger.record_job(0, 0, {"host": "worker-a", "rank": 0})
    # Sent to the second box, answered by the first box's host.
    ledger.record_job(1, 1, {"host": "worker-a.example.test", "rank": 0})

    first, second = ledger.snapshot()["jobs"]
    assert "host_reported" not in first
    assert (second["box"], second["host_reported"]) == (
        "box 2", "worker-a.example.test")


def test_a_reply_naming_a_rank_of_its_own_no_longer_marks_the_row():
    # The rank a fleet worker reports is 0 by construction, so the row keeps
    # the rank the driver submitted to and carries no rank marker.
    ledger = _wave(count=1, names=["worker-a", "worker-b"])
    ledger.record_job(0, 0, {"host": "worker-a", "rank": 5})

    row, = ledger.snapshot()["jobs"]
    assert row["rank"] == 0
    assert set(row) == JOB_FIELDS


def test_a_config_naming_boxes_the_workers_never_name_marks_nothing():
    # A config names its boxes by whatever reaches them and a worker names
    # itself, so two spellings of one box are ordinary. A ledger that cannot
    # tell those apart stays quiet rather than marking every row.
    ledger = _wave(count=2, names=["head-fabric", "worker-fabric"])
    ledger.record_job(0, 0, {"host": "worker-a"})
    ledger.record_job(1, 1, {"host": "worker-b"})

    assert all("host_reported" not in row for row in ledger.snapshot()["jobs"])


def test_a_local_fleet_configures_no_hosts_and_marks_nothing():
    ledger = _wave(count=2)
    ledger.record_job(0, 0, {"host": "worker-a"})
    ledger.record_job(1, 1, {"host": "worker-a"})

    assert all("host_reported" not in row for row in ledger.snapshot()["jobs"])


def test_a_wave_bounds_the_rows_it_publishes_and_says_the_listing_is_partial():
    cap = telemetry_fleet.MAX_PUBLISHED_JOBS
    ledger = _wave(count=cap + 4)
    for index in range(cap + 4):
        ledger.record_job(index, index, {"host": "worker-a"})

    block = ledger.snapshot()
    assert block["count"] == cap + 4
    assert len(block["jobs"]) == cap
    assert block["truncated"] is True


def test_the_same_bound_holds_when_the_dispatches_open_the_rows():
    cap = telemetry_fleet.MAX_PUBLISHED_JOBS
    ledger = _wave(count=cap + 4)
    for index in range(cap + 4):
        ledger.start_job(index, index)

    block = ledger.snapshot()
    assert len(block["jobs"]) == cap
    assert block["truncated"] is True


@pytest.mark.parametrize("job_index", [-1, 2, "0", True, None])
def test_a_row_outside_the_open_wave_is_dropped(job_index):
    ledger = _wave(count=2)
    ledger.record_job(job_index, 0, {"host": "worker-a"})

    assert ledger.snapshot()["jobs"] == []


def test_a_second_row_for_one_job_does_not_double_it():
    ledger = _wave(count=2)
    ledger.record_job(0, 0, {"host": "worker-a"})
    ledger.record_job(0, 0, {"host": "worker-a"})

    assert [row["job"] for row in ledger.snapshot()["jobs"]] == [0]


def test_the_next_wave_replaces_the_last_one_whole():
    ledger = _wave(count=2)
    ledger.record_job(0, 0, {"host": "worker-a"})
    ledger.open_wave([{"text": "a cat"}], 1)

    block = ledger.snapshot()
    assert block["count"] == 1
    assert block["jobs"] == []


class _Hostile(Mapping):
    """A reply that raises when it is read.

    A Mapping on purpose: the ledger reads a reply only when it is one, so a
    reply of some other shape never reaches the raise these tests pin.
    """

    def __getitem__(self, _key):
        raise RuntimeError("worker reply refuses to be read")

    def __iter__(self):
        return iter(("host",))

    def __len__(self):
        return 1


def test_a_hostile_answer_costs_a_row_and_never_the_render():
    # The telemetry_fleet module docstring gives the rule this pins.
    ledger = _wave(count=2)
    ledger.record_job(0, 0, _Hostile())
    ledger.record_job(1, 1, "not a mapping at all")

    jobs = ledger.snapshot()["jobs"]
    # The hostile reply costs its own row and nothing else: the job beside it,
    # whose reply is no mapping, still publishes a row with no host and no wall.
    assert [row["job"] for row in jobs] == [1]
    assert (jobs[0]["host"], jobs[0]["wall_s"]) == (None, None)
    assert jobs[0]["box"] == "box 2"


def test_a_hostile_answer_leaves_the_row_its_dispatch_opened_open():
    # The ledger reads the whole reply before it writes any of it, so a reply
    # that raises halfway through costs the close and never half a row. The
    # dispatch row stands, open, on the box the driver sent the job to, which
    # is the row a Stop leaves too.
    ledger = _wave(count=1)
    ledger.start_job(0, 0)
    ledger.record_job(0, 0, _Hostile())

    row, = ledger.snapshot()["jobs"]
    assert (row["host"], row["wall_s"], row["ended_s"]) == (None, None, None)
    assert row["started_s"] is not None
    assert row["box"] == "box 1"


def test_a_wave_opened_on_something_that_is_not_a_job_list_holds_no_rows():
    ledger = _wave(count=2)
    ledger.record_job(0, 0, {"host": "worker-a"})
    ledger.open_wave("not a job list", 1)

    block = ledger.snapshot()
    assert (block["count"], block["truncated"], block["jobs"]) == (0, False, [])


def test_a_reply_that_answers_with_giant_numbers_still_serializes():
    # This block rides an HTTP route and json.dumps refuses an integer of a
    # few thousand digits. The wave is retained, so one unbounded worker field
    # would keep the route down until the next Fleet render. The ledger cuts
    # the host to telemetry_events' 192 characters, drops a wall no float can
    # hold, and ignores the reply's rank.
    ledger = _wave(count=1)
    ledger.record_job(0, 0, {
        "host": "h" * 200_000, "rank": 10 ** 5000, "sample_s": 10 ** 400})

    block = ledger.snapshot()
    row, = block["jobs"]
    assert len(row["host"]) == 192
    assert row["wall_s"] is None
    assert "host_reported" not in row
    assert json.dumps(block)


def test_a_wave_is_stamped_with_the_moment_it_opened():
    # The block outlives the render that ran it, so a reader looking at it
    # after an unrelated prompt can tell which wave these rows belong to.
    before = time.time()
    block = _wave(count=1).snapshot()

    assert before <= block["t"] <= time.time()


def test_a_job_recorded_with_no_wave_open_is_dropped():
    ledger = telemetry_fleet.FleetJobs()
    ledger.record_job(0, 0, {"host": "worker-a"})

    assert ledger.snapshot() is None


def test_a_driver_that_ran_no_fleet_wave_publishes_no_fleet_key(monkeypatch):
    monkeypatch.setattr(telemetry_fleet, "fleet_jobs", telemetry_fleet.FleetJobs())
    render = RenderProgress().snapshot()

    assert telemetry_fleet.with_fleet(render) is render
    assert "fleet" not in render


def test_the_block_rides_the_render_block_in_place(monkeypatch):
    ledger = _wave(count=2)
    ledger.record_job(0, 0, {"host": "worker-a", "sample_s": 1.5})
    monkeypatch.setattr(telemetry_fleet, "fleet_jobs", ledger)

    progress = RenderProgress()
    token = progress.start(4, {"model": "krea2.safetensors"}, token=object())
    progress.finish(token)
    render = progress.snapshot()

    # The same object the route publishes, with one key added beside the
    # finished render's own fields rather than replacing them.
    assert telemetry_fleet.with_fleet(render) is render
    assert render["last"]["model"] == "krea2.safetensors"
    assert render["fleet"]["count"] == 2
    assert render["fleet"]["jobs"][0]["box"] == "box 1"
    # The whole block survives a JSON round trip: it rides an HTTP route.
    assert json.loads(json.dumps(render["fleet"])) == render["fleet"]


def test_the_route_attaches_the_block_to_the_render_block_it_publishes(monkeypatch):
    from dgx_monarch import gate_ledger, mesh_health, telemetry
    from dgx_monarch.nodes import routes

    ledger = _wave(count=1)
    ledger.record_job(0, 0, {"host": "worker-a", "rank": 0, "sample_s": 2.0})
    monkeypatch.setattr(telemetry_fleet, "fleet_jobs", ledger)
    monkeypatch.setattr(routes, "_workers", lambda: [])
    monkeypatch.setattr(routes, "_ledger_summary", lambda: {})
    monkeypatch.setattr(mesh_health, "mesh_health_snapshot", lambda: {})
    monkeypatch.setattr(telemetry, "host_stats", lambda: {})
    monkeypatch.setattr(telemetry, "events_tail", lambda _limit: [])
    monkeypatch.setattr(gate_ledger, "comfy_commit", lambda: "test-commit")

    payload = routes._telemetry_uncached()

    assert payload["render"]["fleet"]["jobs"][0]["host"] == "worker-a"


def test_the_timeline_file_names_the_fields_the_driver_publishes():
    # A text pin: dgx_monarch_timeline.js stays the timeline's entry point,
    # and the block's field names can be read from it and its model.
    source = TIMELINE.read_text() + TIMELINE_MODEL.read_text()
    for name in ("fleetJobRows", "jobs", "box", "host", "wall_s", "count",
                 "truncated", "block.t", "host_reported", "started_s",
                 "ended_s"):
        assert name in source
    assert "fleetJobRows" in TIMELINE.read_text()
    # A passive reader: the rows arrive on the panel's own poll.
    assert "fetchApi" not in TIMELINE.read_text()
    assert "setInterval" not in TIMELINE.read_text()
    for text in (TIMELINE.read_text(), TIMELINE_MODEL.read_text()):
        assert "\u2014" not in text  # em dash
        assert "\u2013" not in text  # en dash


def test_the_panel_feeds_the_fleet_block_in_and_keeps_the_memory_bars():
    panel = PANEL.read_text()
    assert "renderDetails(t.ledger, render.fleet)" in panel
    assert "fleet wave${when}" in panel
    assert "fleetJobRows" in panel
    # The span the driver measured rides the row the panel already drew.
    assert "timelineRow(row.label, row.seconds, row.span)" in panel
    # A row whose wall is unknown prints a dash, not a measured-looking zero.
    # The panel needs a DOM, so this column is pinned as text.
    assert 'Number.isFinite(seconds) ? `${seconds.toFixed(2)}s` : "-"' in panel
    # The per-box cards and their bars are what a fleet wave is watched on;
    # the job rows are added beside them and displace nothing.
    assert "selectMemoryHostCards(t)" in panel
    assert "frag.appendChild(hostCard(name, stats, meta));" in panel


def _fleet_rows(block):
    """Run the real model module under node and return fleetJobRows(block)."""
    node = shutil.which("node")
    if node is None:  # pragma: no cover - environment dependent
        pytest.skip("node not installed; JS row check skipped")
    driver = (
        "import { fleetJobRows } from "
        f"{json.dumps(TIMELINE_MODEL.as_uri())};\n"
        f"const block = {json.dumps(block)};\n"
        "process.stdout.write(JSON.stringify(fleetJobRows(block)));\n"
    )
    result = subprocess.run(
        [node, "--input-type=module", "-e", driver],
        capture_output=True, text=True, timeout=60, check=False,
    )
    if result.returncode != 0:  # pragma: no cover - surfaced only on a real break
        raise AssertionError(f"node driver failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def test_the_rows_the_sidebar_draws_carry_the_box_that_ran_each_job():
    view = _fleet_rows({
        "t": 1756900000.5,
        "count": 3,
        "truncated": False,
        "jobs": [
            {"job": 0, "prompt": "abc123", "host": "worker-a", "rank": 0,
             "box": "box 1", "wall_s": 12.5},
            {"job": 1, "prompt": None, "host": "worker-b", "rank": 1,
             "box": "box 2", "wall_s": 13.0},
        ],
    })

    assert view["count"] == 3
    assert view["at"] == 1756900000.5
    assert [row["box"] for row in view["rows"]] == ["box 1", "box 2"]
    assert [row["host"] for row in view["rows"]] == ["worker-a", "worker-b"]
    assert [row["seconds"] for row in view["rows"]] == [12.5, 13.0]
    assert view["rows"][0]["label"].startswith("job 1")
    assert "box 1" in view["rows"][0]["label"]
    # The host too, not the box label alone; the fleetJobRows comment in
    # web/js/dgx_monarch_timeline_model.js says why.
    assert "worker-a" in view["rows"][0]["label"]
    assert "worker-b" in view["rows"][1]["label"]
    assert "abc123" in view["rows"][0]["label"]
    # A job with no prompt line says so by leaving the digest off the row.
    assert "null" not in view["rows"][1]["label"]


def test_a_marked_row_says_the_box_it_was_sent_to_is_not_the_one_that_answered():
    view = _fleet_rows({
        "jobs": [
            {"job": 0, "host": "worker-a", "rank": 0, "box": "box 1",
             "wall_s": 1.0},
            {"job": 1, "host": "worker-a", "rank": 1, "box": "box 2",
             "wall_s": 1.0, "host_reported": "worker-a"},
        ],
    })

    assert [row["elsewhere"] for row in view["rows"]] == [False, True]
    assert "another box" not in view["rows"][0]["label"]
    assert view["rows"][1]["label"].endswith("answered from another box")
    # The box it was sent to still leads the row: the marker adds to the
    # driver's own name for the job, it does not replace it.
    assert view["rows"][1]["label"].startswith("job 2 \u00b7 box 2")


def test_a_row_whose_only_name_is_its_host_does_not_print_it_twice():
    view = _fleet_rows({"jobs": [{"job": 0, "host": "worker-a"}]})

    assert view["rows"][0]["box"] == "worker-a"
    assert view["rows"][0]["label"] == "job 1 \u00b7 worker-a"


def test_a_row_carries_the_span_the_driver_measured_over_the_job():
    view = _fleet_rows({
        "count": 2,
        "jobs": [
            {"job": 0, "host": "worker-a", "box": "box 1", "wall_s": 12.5,
             "started_s": 0.02, "ended_s": 12.75},
            {"job": 1, "host": "worker-b", "box": "box 2", "wall_s": 13.0,
             "started_s": 0.05, "ended_s": 13.4},
        ],
    })

    assert [row["span"] for row in view["rows"]] == [
        "+0.02s to +12.75s", "+0.05s to +13.40s",
    ]
    assert [row["started"] for row in view["rows"]] == [0.02, 0.05]
    assert [row["ended"] for row in view["rows"]] == [12.75, 13.4]
    assert all("no result" not in row["label"] for row in view["rows"])


def test_a_dispatched_job_with_nothing_back_says_so_and_keeps_its_start():
    # What a Stop mid-fleet leaves: the box the job was sent to, when it went,
    # and no end. The row says no result rather than guess a wall for it.
    view = _fleet_rows({
        "count": 2,
        "jobs": [
            {"job": 0, "host": "worker-a", "box": "box 1", "wall_s": 4.0,
             "started_s": 0.01, "ended_s": 4.2},
            {"job": 1, "host": None, "box": "box 2", "wall_s": None,
             "started_s": 0.03, "ended_s": None},
        ],
    })

    open_row = view["rows"][1]
    assert open_row["span"] == "+0.03s"
    assert open_row["ended"] is None
    assert open_row["label"] == "job 2 · box 2 · no result"
    # The wall the worker never reported stays unknown. Reading it as 0 would
    # print a measured-looking 0.00s beside a row that measured nothing.
    assert open_row["seconds"] is None
    assert view["rows"][0]["seconds"] == 4.0


def test_the_rows_keep_the_order_the_driver_published_them_in():
    # The driver publishes its rows in dispatch order and the panel prints
    # them in the order it is given: one home for that decision, not two.
    view = _fleet_rows({"jobs": [{"job": 2}, {"job": 0}, {"job": 1}]})

    assert [row["label"].split(" ")[1] for row in view["rows"]] == ["3", "1", "2"]


def test_a_row_the_driver_timed_nothing_on_carries_no_span_and_no_marker():
    view = _fleet_rows({"jobs": [
        {"job": 0, "host": "worker-a", "box": "box 1", "wall_s": 2.0},
        {"job": 1, "box": "box 2", "started_s": "soon", "ended_s": None},
    ]})

    assert [row["span"] for row in view["rows"]] == ["", ""]
    assert all("no result" not in row["label"] for row in view["rows"])


@pytest.mark.parametrize("block", [None, "", 0, {"jobs": "not a list"}])
def test_a_block_the_driver_never_sent_costs_no_rows(block):
    view = _fleet_rows(block)

    assert view is None or view["rows"] == []


def test_a_row_missing_every_field_still_draws_something():
    view = _fleet_rows({"jobs": [{}, None, {"job": 4}]})

    assert [row["seconds"] for row in view["rows"]] == [None, None]
    assert view["at"] is None
    assert view["rows"][0]["box"] == "box unknown"
    assert view["rows"][0]["host"] == ""
    assert view["rows"][1]["label"].startswith("job 5")
    # A missing host leaves the row short, never the word undefined in it.
    assert all("undefined" not in row["label"] for row in view["rows"])
