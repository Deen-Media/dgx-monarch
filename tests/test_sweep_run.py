"""The sweep runner scores a cell, and ends a session the rig has wedged.

These tests use a fake driver: they check what the runner does with an answer,
not how it is fetched. run.SessionWatch says why a wedged session must end.
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmark.sweep import matrix, run  # noqa: E402
from benchmark.sweep.driver import SessionWedged, queue_and_wait, waiver_run_ids  # noqa: E402

from dgx_monarch import mesh_runtime  # noqa: E402
from dgx_monarch.accuracy_waiver import STAMPED_RESULT_KEY  # noqa: E402
from dgx_monarch.nodes.samplers import _sampler_result  # noqa: E402

# The openings of the two operator-visible capacity walls. mesh_safety.py
# raises the first untagged. nodes/loader_preflight.py raises the second as
# class C, or as class K when the combination's quarantine reads FAIL (the
# form here).
ACTIVATION_WALL = ("activation footprint preflight refuses ltx.safetensors: estimated "
                   "90.0 GiB (weights 40.0 GiB + ~50.0 GiB activations for 8000 tokens "
                   "[8000 video+ref, 0 pose] over 2 rank(s)) exceeds 60.0 GiB available.")
DRIVER_WALL = ("[dgxm:K] driver-side footprint preflight at the loader node: 61.7 GiB "
               "to place on the driver host, 53.8 GiB usable. Refuses a bf16 checkpoint "
               "at the resident window.")


class _FakeSampler:
    """The memory sampler, minus the per-second ssh the unit suite cannot make."""

    def __init__(self, hosts, path, label="") -> None:
        self.hosts, self.label = hosts, label
        self.summary: dict = {}

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass


def test_the_cell_deadline_caps_submit_and_every_history_request(monkeypatch):
    """The cell override must reach HTTP, not only the loop after a response."""
    from benchmark.sweep import driver

    moments = iter((100.0, 100.0, 102.5, 103.0))
    calls: list[tuple[str, float]] = []
    monkeypatch.setattr(driver.time, "perf_counter", lambda: next(moments))
    monkeypatch.setattr(driver, "post", lambda url, body, *, timeout:
                        calls.append((url, timeout)) or {"prompt_id": "p1"})
    monkeypatch.setattr(driver, "get", lambda url, *, timeout:
                        calls.append((url, timeout)) or {"p1": {"status": {"completed": True}}})

    wall, entry, state, prompt_id = queue_and_wait("http://fake", {}, "client", 10.0)

    assert (wall, state, prompt_id) == (3.0, "", "p1")
    assert entry == {"status": {"completed": True}}
    assert calls == [("http://fake/prompt", 10.0), ("http://fake/history/p1", 7.5)]


def test_a_lost_submit_response_ends_the_session_before_another_cell_queues(monkeypatch):
    """A timed-out POST can have queued a prompt but gives no id to drain."""
    from benchmark.sweep import driver

    monkeypatch.setattr(driver.time, "perf_counter", lambda: 100.0)
    calls: list[float] = []

    def lost_response(url, body, *, timeout):
        del url, body
        calls.append(timeout)
        raise TimeoutError("timed out")

    monkeypatch.setattr(driver, "post", lost_response)
    with pytest.raises(SessionWedged, match="submission outcome is unknown"):
        queue_and_wait("http://fake", {}, "client", 15.0)
    assert calls == [15.0]


@pytest.mark.parametrize("response", [
    {"prompt_id": ""}, {"prompt_id": "  \t"}, {"prompt_id": None}, {"prompt_id": 7},
    [], "not-json-object",
])
def test_an_unusable_submit_response_ends_the_session_before_polling(monkeypatch, response):
    """A response must identify a prompt before the runner can recover it."""
    from benchmark.sweep import driver

    monkeypatch.setattr(driver.time, "perf_counter", lambda: 100.0)
    calls: list[float] = []
    monkeypatch.setattr(driver, "post", lambda _url, _body, *, timeout:
                        calls.append(timeout) or response)
    monkeypatch.setattr(driver, "get", lambda *_args, **_kwargs:
                        pytest.fail("an untrackable submission must not poll"))

    with pytest.raises(SessionWedged, match="submission outcome is unknown"):
        queue_and_wait("http://fake", {}, "client", 15.0)
    assert calls == [15.0]


def test_a_history_transport_error_is_labelled_and_drained_by_the_known_prompt(
        monkeypatch, tmp_path):
    """A known prompt is cleaned up; only a lost submit is unsafe to continue."""
    monkeypatch.setattr(run, "Sampler", _FakeSampler)
    monkeypatch.setattr(run, "queue_and_wait", lambda *args, **kwargs:
                        (5.0, None, "poll transport TimeoutError: timed out", "p1"))
    drained: list[str] = []
    monkeypatch.setattr(run, "drain", lambda _driver, prompt_id: drained.append(prompt_id) or "drained")
    monkeypatch.setattr(run, "frames_for", lambda output_dir, prefix: [])

    leg = run.run_leg(_cell(), {"driver": "http://fake", "comfy_dir": tmp_path}, "client",
                      "cold", {}, "prefix", 10.0, [""], tmp_path)

    assert leg["outcome"] == "cell-error"
    assert leg["message"] == "poll transport TimeoutError: timed out"
    assert leg["drain"] == "drained" and drained == ["p1"]


def _cell(**over) -> dict:
    cell = {"id": "abc123", "label": "refuse:untyped", "capacity_risk": True,
            "waiver": False, "waiver_guards": []}
    cell.update(over)
    return cell


def test_both_capacity_walls_score_as_a_capacity_pass():
    # Matching the activation wall alone, the runner scored the four minimax-h3
    # capacity_risk cells that met the driver-side wall FINDING (2026-09-02),
    # for an answer the matrix cannot price.
    cell = _cell()
    for wall in (ACTIVATION_WALL, DRIVER_WALL):
        verdict, detail = run.verdict_for(cell, "refuse:K", wall)
        assert verdict == "PASS-capacity" and "capacity boundary" in detail, wall
    assert run.verdict_for(cell, "refuse:untyped")[0] == "PASS"
    assert run.verdict_for(cell, "refuse:K", "some other refusal")[0] == "FINDING"
    assert run.verdict_for(_cell(capacity_risk=False), "refuse:K", DRIVER_WALL)[0] == "FINDING"


def test_a_capacity_risk_keeps_non_capacity_outcomes_as_findings():
    cell = _cell(label="refuse:P", capacity_risk_basis="flux2-fp8mixed-33gib-observed")
    # The observed flux2 fp8mixed boundary lets only the capacity class answer
    # ahead of P. P still passes as the labelled physics answer, and the marker
    # alone turns no other outcome into a capacity pass.
    assert run.verdict_for(cell, "refuse:C")[0] == "PASS-capacity"
    assert run.verdict_for(cell, "refuse:P")[0] == "PASS"
    assert run.verdict_for(cell, "refuse:untyped")[0] == "FINDING"
    assert run.verdict_for(cell, "crash")[0] == "FINDING"


def test_a_refusal_card_survives_the_message_cap(monkeypatch, tmp_path):
    # A worker refusal reaches /history under its remote framing and a
    # traceback, with the card at the end, so a cut that keeps the head stores
    # everything except the answer (2026-09-02).
    framed = "A remote actor call has failed.\n" + ("stack frame\n" * 3000) + DRIVER_WALL
    entry = {"status": {"status_str": "error", "messages": [
        ["execution_error", {"exception_message": framed,
                             "exception_type": "dgx_monarch.actor.failure.CapacityError"}]]}}
    monkeypatch.setattr(run, "Sampler", _FakeSampler)
    monkeypatch.setattr(run, "queue_and_wait",
                        lambda *args, **kwargs: (1.0, entry, "", "prompt-1"))
    monkeypatch.setattr(run, "frames_for", lambda output_dir, prefix: [])
    config = {"driver": "http://fake", "comfy_dir": tmp_path}
    leg = run.run_leg(_cell(), config, "client", "cold", {}, "prefix", 10.0, [""], tmp_path)
    assert len(framed) > run.MESSAGE_KEEP and len(leg["message"]) == run.MESSAGE_KEEP
    assert leg["message"].endswith(DRIVER_WALL)
    # The stored message is what the verdict reads, so the card has to be in it.
    assert run.verdict_for(_cell(), leg["outcome"], leg["message"])[0] == "PASS-capacity"


def test_a_render_leg_keeps_only_its_prompt_waiver_dispatch_ids(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "Sampler", _FakeSampler)
    entry = {"status": {"completed": True}, "outputs": {
        "sampler": {"dgxm_waiver_runs": ["leg-run", "leg-run", None]},
        "other": {"dgxm_waiver_runs": "foreign"},
    }}
    monkeypatch.setattr(run, "queue_and_wait", lambda *args, **kwargs: (1.0, entry, "", "p1"))
    monkeypatch.setattr(run, "frames_for", lambda *_args: [tmp_path / "image.png"])
    leg = run.run_leg(_cell(), {"driver": "http://fake", "comfy_dir": tmp_path}, "client",
                      "cold", {}, "prefix", 10.0, [""], tmp_path)
    assert leg["outcome"] == "render"
    assert leg["waiver_run_ids"] == ["leg-run"]


def test_a_recycle_the_driver_refuses_twice_ends_the_session():
    # A ProcMesh stop that times out latches the driver: every replacement
    # answers 503 and every cell after it crashes in two seconds (2026-09-02).
    watch = run.SessionWatch("http://fake", 70.0)
    watch.recycled(run.RECYCLE_BUSY)
    watch.recycled("ok")  # the run has to be consecutive
    watch.recycled(run.RECYCLE_BUSY)
    with pytest.raises(SessionWedged, match="in a row"):
        watch.recycled(run.RECYCLE_BUSY)


def test_a_message_naming_a_fleet_no_attach_can_replace_ends_the_session():
    watch = run.SessionWatch("http://fake", 70.0)
    watch.message("")
    watch.message("[dgxm:P] cfg-parallel has nothing to split")
    # A transport not reusable while another creation runs is a wait, not a
    # wedge, so the poison clause is "not reusable after", not "not reusable".
    watch.message("MeshAttachError: Monarch transport is not reusable while another "
                  "mesh creation is still in progress")
    for text in ("MeshAttachError: the previous worker fleet could not be stopped "
                 "safely; replacement is blocked (stop timed out). Resolve teardown "
                 "or restart ComfyUI.",
                 "MeshAttachError: this ComfyUI session's Monarch transport is not "
                 "reusable after a partial bring-up failure; restart ComfyUI",
                 "MeshAttachError: Monarch transport is not reusable after a creation "
                 "failure (cleanup pending); restart ComfyUI"):
        with pytest.raises(SessionWedged, match="restart the driver"):
            watch.message(text)


def test_the_attach_pace_holds_after_a_replacement_and_reads_the_mesh(monkeypatch):
    # An attach right behind a fleet replacement meets a fleet still coming up.
    slept: list[float] = []
    monkeypatch.setattr(run.time, "sleep", slept.append)
    watch = run.SessionWatch("http://fake", 70.0)
    monkeypatch.setattr(watch, "mesh_block", lambda: {"state": "ok", "verdict": "none"})
    watch.wait_to_attach()
    assert not slept  # nothing was replaced, so nothing is owed
    watch.recycled("ok")
    watch.wait_to_attach()
    assert len(slept) == 1 and 60 < slept[0] <= 70
    # One replacement owes one pace.
    watch.wait_to_attach()
    assert len(slept) == 1


def test_a_dirty_mesh_block_is_reported_and_never_wedges_on_its_own(monkeypatch):
    # The block is evidence in a wedge reason, never a wedge of its own
    # (SessionWatch.wait_to_attach says why).
    monkeypatch.setattr(run.time, "sleep", lambda seconds: None)
    watch = run.SessionWatch("http://fake", 0.0)
    watch.recycled("ok")
    monkeypatch.setattr(watch, "mesh_block", lambda: {"state": "dirty"})
    watch.wait_to_attach()
    assert watch.dirty_block == {"state": "dirty"}
    # It appears in the reason of the wedge that does fire.
    with pytest.raises(SessionWedged, match=re.escape('mesh block read {"state": "dirty"}')):
        watch.message("MeshAttachError: replacement is blocked")
    # A clean read, and a driver that did not answer, both leave it empty.
    for block in ({"state": "ok", "verdict": "none"}, {}):
        watch.recycled("ok")
        monkeypatch.setattr(watch, "mesh_block", lambda block=block: block)
        watch.wait_to_attach()
        assert watch.dirty_block == {}
    # The predicate is the runtime's own, not a copy.
    assert run.fleet_is_dirty({"replacement_blocked": "stop timed out"})
    assert not run.fleet_is_dirty({"state": "ok", "verdict": "none"})


def test_the_attach_pace_follows_the_budget_monarch_was_launched_with(monkeypatch):
    # A pace under monarch's own attach budget ends inside the window it should
    # wait out, and the sweep driver launches at 180 s.
    monkeypatch.setenv(matrix.ATTACH_TIMEOUT_ENV, "180s")
    assert matrix.attach_pace_default() == 180.0
    # The runtime's own wait is the floor, whatever the budget says.
    monkeypatch.setenv(matrix.ATTACH_TIMEOUT_ENV, "60s")
    assert matrix.attach_pace_default() == float(matrix.ATTACH_INIT_WAIT_S)
    monkeypatch.setenv(matrix.ATTACH_TIMEOUT_ENV, "a while")
    assert matrix.attach_pace_default() == float(matrix.ATTACH_INIT_WAIT_S)
    monkeypatch.delenv(matrix.ATTACH_TIMEOUT_ENV)
    assert matrix.attach_pace_default() == float(matrix.ATTACH_INIT_WAIT_S)


def test_the_launched_pace_clears_the_budget_the_runtime_itself_read(monkeypatch):
    """The launched path, where the variable is exported before python starts.

    The test above sets it after import, so the runtime's wait stays at the
    shipped 70 s and the pace equals the budget. A launched sweep exports it
    first, the runtime reads 190, and the pace must clear the budget, not equal
    it, or the hold ends inside the runtime's own attach wait.
    """
    monkeypatch.setenv(matrix.ATTACH_TIMEOUT_ENV, "180s")
    monkeypatch.setattr(matrix, "ATTACH_INIT_WAIT_S", mesh_runtime._attach_init_wait())

    assert matrix.ATTACH_INIT_WAIT_S == 190
    assert matrix.attach_pace_default() == 190.0 > 180.0


def test_a_config_that_names_no_pace_takes_the_one_the_environment_names(
        monkeypatch, tmp_path):
    monkeypatch.setenv(matrix.ATTACH_TIMEOUT_ENV, "180s")
    config = tmp_path / "sweep.toml"
    config.write_text(f'out_dir = "{tmp_path}"\ncomfy_dir = "{tmp_path}"\n')
    assert matrix.load_config(config)["attach_pace_s"] == 180.0
    config.write_text(f'out_dir = "{tmp_path}"\ncomfy_dir = "{tmp_path}"\n'
                      "attach_pace_s = 30.0\n")
    assert matrix.load_config(config)["attach_pace_s"] == 30.0


def test_a_fake_driver_wedges_the_reset_on_its_own_answers(monkeypatch):
    # The clear graph meets the wedge before the recycle (reset_fleet says why).
    blocked = {"status": {"status_str": "error", "messages": [
        ["execution_error", {"exception_message": "MeshAttachError: the previous worker "
                             "fleet could not be stopped safely; replacement is blocked."}]]}}
    monkeypatch.setattr(run, "clear_graph", lambda *args, **kwargs: {})
    monkeypatch.setattr(run, "queue_and_wait",
                        lambda *args, **kwargs: (1.0, blocked, "", "prompt-1"))
    monkeypatch.setattr(run, "recycle", lambda driver: "ok")
    config = {"driver": "http://fake"}
    watch = run.SessionWatch(config["driver"], 0.0)
    with pytest.raises(SessionWedged, match="replacement is blocked"):
        run.reset_fleet(config, "client", "off", "cluster", watch)
    # Without a watch the reset reads no message and still recycles.
    assert run.reset_fleet(config, "client", "off", "cluster")["recycle"] == "ok"


def test_a_reset_that_keeps_answering_503_ends_the_session(monkeypatch):
    monkeypatch.setattr(run, "clear_graph", lambda *args, **kwargs: {})
    monkeypatch.setattr(run, "queue_and_wait", lambda *args, **kwargs: (1.0, {}, "", "prompt-1"))
    monkeypatch.setattr(run, "recycle", lambda driver: run.RECYCLE_BUSY)
    config = {"driver": "http://fake"}
    watch = run.SessionWatch(config["driver"], 0.0)
    monkeypatch.setattr(watch, "mesh_block", dict)
    run.reset_fleet(config, "client", "off", "cluster", watch)
    with pytest.raises(SessionWedged, match="restart the driver"):
        run.reset_fleet(config, "client", "off", "cluster", watch)


def _record(verdict: str, findings: list[str]) -> dict:
    return {"verdict": verdict, "findings": list(findings)}


def test_no_pass_survives_a_finding_against_the_same_cell():
    # A cold leg that matched its label, took its waiver and then crashed must
    # not keep PASS with the crash in the findings list (2026-09-02).
    assert run.settle_verdict(_record("PASS", [])) == "PASS"
    assert run.settle_verdict(
        _record("PASS", ["waiver granted, the render still crash"])) == "FINDING"
    assert run.settle_verdict(_record("PASS-capacity", ["warm leg: crash"])) == "FINDING"
    # A compare the harness could not make is unproven, not a fault, and the
    # step-nrms bar the null A/B calibrates keeps its own CHECK.
    assert run.settle_verdict(
        _record("PASS", ["reference compare: missing frames (0 vs 0)"])) == "CHECK"
    # The cache line calls itself probable, and a cold leg that paid the
    # identity gate makes any warm leg look like a cache hit.
    assert run.settle_verdict(_record("PASS", [
        "warm leg ran in 12.0 s against a 300.0 s cold leg; the execution cache "
        "probably served it"])) == "CHECK"
    assert run.settle_verdict(_record("PASS", [
        "warm leg ran in 12.0 s against a 300.0 s cold leg; the execution cache "
        "probably served it", "waiver granted, the render still crash"])) == "FINDING"
    assert run.settle_verdict(_record("CHECK", ["step-nrms 0.14 over the 0.1 floor"])) == "CHECK"
    assert run.settle_verdict(_record("FINDING", [])) == "FINDING"


def test_a_cold_leg_that_passed_never_survives_a_waived_leg_that_crashed(
        monkeypatch, tmp_path):
    # The cold leg refuses the class its waiver admits, the card is granted,
    # and the waived render crashes. The cold leg matched its label, but the
    # findings list is the defect list, so the cell reads FINDING.
    out = tmp_path / "out"
    (out / "graphs").mkdir(parents=True)
    (out / "graphs" / "t.json").write_text(json.dumps({}))
    cell = {"id": "abc123", "template": "t", "session": "cluster", "preset": "ring2",
            "mode": "cluster", "attention": "TORCH_FLASH", "label": "refuse:K",
            "waiver": True, "waiver_guards": ["ring_pad"], "levers": {}, "loras": [],
            "unets": {}, "auto_gate": "first_use", "gpus_per_host": 0, "batch": 1,
            "batch_toggled": False, "batch_node": "", "capacity_risk": False,
            "probe": False, "audio": False, "timeout_s": 10,
            "reference": {"kind": "none", "bar": "reference"}}
    refused = {"status": {"status_str": "error", "messages": [
        ["execution_error", {"exception_message": "[dgxm:K] ring pad rows"}]]}}
    crashed = {"status": {"status_str": "error", "messages": [
        ["execution_error", {"exception_message": "CUDA error: illegal memory access"}]]}}
    answers = iter([(2.0, refused, "", "p1"), (3.0, crashed, "", "p2")])
    monkeypatch.setattr(run, "Sampler", _FakeSampler)
    monkeypatch.setattr(run, "queue_and_wait", lambda *args, **kwargs: next(answers))
    monkeypatch.setattr(run, "frames_for", lambda output_dir, prefix: [])
    monkeypatch.setattr(run, "wait_for_memory", lambda *args, **kwargs: {"cleared": True})
    monkeypatch.setattr(run, "journal_lines", lambda host, since: [])
    monkeypatch.setattr(run, "free_cache", lambda driver: "freed")
    monkeypatch.setattr(run, "grant_pending_consent", lambda driver, cell: {"granted": True})
    config = {"driver": "http://fake", "out_dir": out, "comfy_dir": tmp_path,
              "sibling": "", "mem_floor_gib": 0.0, "reference": {"nrms_floor": 0.10}}
    args = type("Args", (), {"timeout_s": 0.0})()
    record = run.run_cell(cell, config, {}, "client", args)
    assert record["observed"] == "refuse:K" and record["legs"]["waived"]["outcome"] == "crash"
    assert record["verdict"] == "FINDING"
    assert any("the render still crash" in line for line in record["findings"])


def test_the_cache_flag_reads_the_resident_leg_before_the_cold_one():
    # These timings are from one of four campaign runs
    # cells that scored CHECK on a identical output (recorded 2026-09-04).
    # CACHE_BASELINE_LEGS says why the resident leg rules.
    legs = {"cold": {"outcome": "render", "wall_s": 284.2},
            "warm": {"outcome": "render", "wall_s": 40.0},
            "resident": {"outcome": "render", "wall_s": 42.0}}
    assert run.cache_flag({"legs": legs}) == ""
    record = {"verdict": "PASS", "findings": []}
    assert run.settle_verdict(record) == "PASS"
    # With no resident leg, the same warm leg against the cold leg still reads
    # as a cache hit.
    no_twin = {"cold": legs["cold"], "warm": legs["warm"]}
    assert run.cache_flag({"legs": no_twin}).startswith("warm leg ran in 40.0 s against a 284.2 s cold leg")
    # A warm leg the cache did serve is flagged against the resident wall too,
    # and the line keeps the prefix that scores it unproven.
    served = dict(legs, warm={"outcome": "render", "wall_s": 4.0})
    flagged = run.cache_flag({"legs": served})
    assert "against a 42.0 s resident leg" in flagged
    assert flagged.startswith(run.UNPROVEN_FINDINGS[1])
    assert run.settle_verdict({"verdict": "PASS", "findings": [flagged]}) == "CHECK"
    # Without a warm render there is nothing to read.
    assert run.cache_flag({"legs": {"cold": legs["cold"]}}) == ""
    assert run.cache_flag({"legs": dict(legs, warm={"outcome": "crash", "wall_s": 1.0})}) == ""


def test_the_cache_flag_reads_the_history_before_the_wall():
    # W3c, 2026-09-07: anima's warm leg ran in 14 s against a 70 s cold leg
    # with compile_dit and against 72 s with slab_weights
    # on (output identical to its reference). The cold leg paid a
    # compile or a slab build the warm leg never pays, and the wall ratio read
    # both as cache hits. ComfyUI's history names the nodes its cache served, so
    # a warm leg that ran its seeded nodes is a render whatever its wall, and one
    # whose seeded node the cache served is the seed shift not taking.
    ran = {"cold": {"outcome": "render", "wall_s": 72.0},
           "warm": {"outcome": "render", "wall_s": 14.0, "cached_seeded": []}}
    assert run.cache_flag({"legs": ran}) == ""
    served = {"cold": ran["cold"],
              "warm": {"outcome": "render", "wall_s": 40.0, "cached_seeded": ["3"]}}
    flagged = run.cache_flag({"legs": served})
    assert flagged.startswith(run.UNPROVEN_FINDINGS[2]) and "seeded node(s) 3" in flagged
    assert run.settle_verdict({"verdict": "PASS", "findings": [flagged]}) == "CHECK"
    # A record written before the signal was kept still reads the wall ratio.
    legacy = {"cold": ran["cold"], "warm": {"outcome": "render", "wall_s": 14.0}}
    assert run.cache_flag({"legs": legacy}).startswith("warm leg ran in 14.0 s")


def test_seeded_nodes_and_the_cache_message_meet_on_node_ids():
    from benchmark.sweep import driver

    graph = {"3": {"class_type": "KSampler", "inputs": {"seed": 5, "steps": 4}},
             "7": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise_seed": 1}},
             "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "x"}},
             "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "seed"}}}
    assert run.seeded_nodes(graph) == ["3", "7"]
    entry = {"status": {"status_str": "success", "messages": [
        ["execution_start", {"prompt_id": "p"}],
        ["execution_cached", {"nodes": [2, "3"], "prompt_id": "p"}],
        ["execution_success", {"prompt_id": "p"}]]}}
    assert driver.cached_nodes(entry) == ["2", "3"]
    assert driver.cached_nodes({}) == [] and driver.cached_nodes({"status": {"messages": [["x"]]}}) == []
    assert sorted(set(driver.cached_nodes(entry)) & set(run.seeded_nodes(graph))) == ["3"]


def test_a_record_a_wedged_driver_wrote_re_runs_on_the_next_resume():
    # Under a wedged driver the record is the rig's answer, not the cell's,
    # whatever class it was scored as. The 2026-09-02 wedge left krea2 cfg2
    # records of this kind.
    pin = {"comfy_commit": "c", "source_digest": "s", "torch": "t", "nccl": "n", "world": 2}
    reference = {"kind": "none", "bar": "reference", "cell": None, "probe_steps": 1}
    clean = {"observed": "refuse:P", "verdict": "PASS", "graph_digest": "g", "pin": pin,
             "cell": {"label": "refuse:P", "reference": reference},
             "legs": {"cold": {"outcome": "refuse:P", "message": "[dgxm:P] no split"}}}
    assert run.settled(clean) and run.resumable(clean, pin, "g", "refuse:P", reference)
    for message in ("MeshAttachError: the previous worker fleet could not be stopped "
                    "safely; replacement is blocked (stop timed out).",
                    "MeshAttachError: this session's transport is not reusable after a "
                    "partial bring-up failure"):
        wedged = json.loads(json.dumps(clean))
        wedged["legs"]["cold"]["message"] = message
        assert run.wedged_answer(wedged) and not run.settled(wedged)
        assert not run.resumable(wedged, pin, "g", "refuse:P", reference)
    # A bare wedge clause with the exception type in its own field still marks
    # the record wedged, and so does a queue that would not clear.
    typed = json.loads(json.dumps(clean))
    typed["legs"]["cold"]["exception_type"] = "dgx_monarch.mesh.MeshAttachError"
    typed["legs"]["cold"]["message"] = "replacement is blocked"
    assert not run.settled(typed)
    undrained = json.loads(json.dumps(clean))
    undrained["legs"]["warm"] = {"outcome": "render", "drain": "still running"}
    assert not run.settled(undrained)
    # A replacement the driver refused leaves the previous fleet in place, so
    # the cell answered on its leftover load rather than on its own guard.
    latched = json.loads(json.dumps(clean))
    latched["reset"] = {"recycle": run.RECYCLE_BUSY, "reason": "template"}
    assert not run.settled(latched)
    # A crash is never settled.
    crashed = json.loads(json.dumps(clean))
    crashed.update(observed="crash", verdict="FINDING")
    assert not run.settled(crashed)


def test_a_wedged_session_writes_its_reason_and_exits_non_zero(monkeypatch, tmp_path):
    # The outer shell restarts the driver and the worker loops on this.
    out = tmp_path / "out"
    (out / "graphs").mkdir(parents=True)
    cell = {"id": "abc123", "template": "t", "session": "cluster", "sampled_out": False,
            "label": "render", "klass": "image", "levers": {}, "preset": "auto",
            "mode": "cluster", "attention": "TORCH_FLASH", "waiver": False,
            "waiver_guards": [], "reference": {"kind": "none", "bar": "reference"}}
    (out / "cells.jsonl").write_text(json.dumps(cell) + "\n")
    config = tmp_path / "sweep.toml"
    config.write_text(f'out_dir = "{out}"\ncomfy_dir = "{tmp_path}"\n')
    monkeypatch.setattr(run, "session_pin", lambda config: {})
    monkeypatch.setattr(run, "recycle", lambda driver: "ok")
    # The config names no driver, so the default is the operator's own port,
    # and no test in this suite may reach a live driver.
    monkeypatch.setattr(run.SessionWatch, "mesh_block", lambda self: {})
    monkeypatch.setattr(run, "probe_transport", lambda *args, **kwargs: "cluster")
    monkeypatch.setattr(run, "run_cell", lambda *args, **kwargs: (_ for _ in ()).throw(
        SessionWedged("the driver answered 'not reusable'")))
    monkeypatch.setattr(run.time, "sleep", lambda seconds: None)
    assert run.main(["--config", str(config), "--session", "cluster"]) == 1
    summary = json.loads((out / "run_summary_cluster.json").read_text())
    assert "not reusable" in summary["wedged"] and summary["ran"] == 0
    assert summary["runnable"] == 1


def _fidelity_cell(**over) -> dict:
    cell = {"id": "abc123", "audio": False, "probe": False, "probe_reason": "",
            "reference": {"kind": "cell", "cell": "ref456",
                          "bar": matrix.FULL_RENDER_BAR, "probe_steps": 0}}
    cell.update(over)
    return cell


def test_a_full_render_compare_is_a_check_the_one_step_floor_never_scores(
        monkeypatch, tmp_path):
    # ideogram4 bf16 uly2 read 0.198 as a full render and fp8 0.331 (recorded
    # 2026-09-03), against a floor calibrated for a one-step probe. The runner
    # measures and reports the number, and invents no floor for it.
    reason = "the step count lives in FutureScheduler, which the probe rewrite does not know"
    cell = _fidelity_cell(probe_reason=reason)
    seen: list[tuple] = []

    def fake_compare(output_dir, candidate, reference, audio=False, nrms=0.0):
        seen.append((candidate, reference))
        return {"nrms": nrms, "max_abs": 74, "frames": 1, "identical": False,
                "reference_file": "sweep_ref456_warm_00001_.png"}

    for nrms in (0.331, 0.0):
        monkeypatch.setattr(run, "compare_outputs",
                            lambda *args, nrms=nrms, **kwargs: fake_compare(*args, nrms=nrms))
        record = {"verdict": "PASS", "findings": [], "notes": []}
        run._fidelity(record, cell, tmp_path, 0.10)
        assert record["fidelity"]["bar"] == matrix.FULL_RENDER_BAR
        assert record["verdict"] == "CHECK" and run.settle_verdict(record) == "CHECK"
        finding = record["findings"][0]
        assert finding.startswith(f"{matrix.FULL_RENDER_BAR} {nrms} ")
        assert "no probe leg is available" in finding and reason in finding
        assert "floor" not in finding.split("no floor applies")[1]
        # The report matches this line rather than reading its wording.
        assert record["full_render"] == finding and record["notes"] == []
    # Both sides are the warm leg: there is no probe leg to compare.
    assert seen == [("sweep_abc123_warm", "sweep_ref456_warm")] * 2
    # A cell that already failed keeps its FINDING; the bar never softens one.
    record = {"verdict": "FINDING", "findings": ["warm leg: crash"], "notes": []}
    run._fidelity(record, cell, tmp_path, 0.10)
    assert record["verdict"] == "FINDING"


def test_a_full_render_that_matched_to_the_pixel_is_not_a_check(monkeypatch, tmp_path):
    # Two renders equal to the pixel are the exact measurement and need no
    # floor, so the PASS stands.
    monkeypatch.setattr(run, "compare_outputs", lambda *args, **kwargs: {
        "nrms": 0.0, "max_abs": 0, "frames": 1, "identical": True,
        "reference_file": "sweep_ref456_warm_00001_.png"})
    record = {"verdict": "PASS", "findings": [], "notes": []}
    run._fidelity(record, _fidelity_cell(), tmp_path, 0.10)
    assert record["verdict"] == "PASS" and run.settle_verdict(record) == "PASS"
    assert record["findings"] == [] and "full_render" not in record
    assert record["notes"][0].startswith(f"{matrix.FULL_RENDER_BAR} 0.0 ")


def test_a_waived_kernel_number_is_recorded_and_never_checked(monkeypatch, tmp_path):
    # A sol kernel is an approximation the class K waiver admits on purpose, so
    # the render is the PASS and the deviation is the result. Scored at the
    # one-step floor, every sol cell of the wave would read CHECK and file its
    # result as a doubt.
    monkeypatch.setattr(run, "compare_outputs", lambda *args, **kwargs: {
        "nrms": 0.42, "max_abs": 91, "frames": 1, "identical": False,
        "reference_file": "sweep_ref456_probe_00001_.png"})
    cell = _fidelity_cell(probe=True, attention="SOL_ATTN_TAU0.7",
                          reference={"kind": "cell", "cell": "ref456",
                                     "bar": matrix.WAIVED_BAR, "probe_steps": 1})
    record = {"verdict": "PASS", "findings": [], "notes": []}
    run._fidelity(record, cell, tmp_path, 0.10)
    assert record["fidelity"]["bar"] == matrix.WAIVED_BAR
    assert record["verdict"] == "PASS" and run.settle_verdict(record) == "PASS"
    assert record["findings"] == []
    line = record["waived_render"]
    assert record["notes"] == [line] and line.startswith(f"{matrix.WAIVED_BAR} 0.42 ")
    assert "SOL_ATTN_TAU0.7" in line and "no floor applies" in line
    # A fault filed against the same cell still stands: the bar reports a
    # number, it never clears a finding.
    record = {"verdict": "FINDING", "findings": ["warm leg: crash"], "notes": []}
    run._fidelity(record, cell, tmp_path, 0.10)
    assert run.settle_verdict(record) == "FINDING"


def _waived_leg_cell(**over) -> dict:
    """A cell the matrix could only label refuse:K.

    The guard fires on a scale the driver reads out of the file, so the kernel
    name says nothing about it and the matrix hands the cell the step-nrms bar.
    """
    cell = _fidelity_cell(probe=True, label="refuse:K", attention="TORCH_FLASH",
                          waiver=True, waiver_guards=["shard_quant_scale:chroma"],
                          reference={"kind": "cell", "cell": "ref456",
                                     "bar": "step-nrms", "probe_steps": 1})
    cell.update(over)
    return cell


def _waived_leg_record(waived: str | None = "render", *, granted: bool = True,
                       rows=None) -> dict:
    legs = {"cold": {"outcome": "refuse:K"}, "warm": {"outcome": "render"},
            "probe": {"outcome": "render"}}
    if waived is not None:
        legs["waived"] = {"outcome": waived, "waiver_run_ids": ["this-waived-run"]}
    return {"verdict": "PASS", "findings": [], "notes": [], "observed": "refuse:K",
            "legs": legs, "consent": {"granted": granted},
            "ledger_rows": [_use_row(run_id="this-waived-run")] if rows is None else rows}


def _waived_leg_compare(monkeypatch) -> None:
    monkeypatch.setattr(run, "compare_outputs", lambda *args, **kwargs: {
        "nrms": 0.120494, "max_abs": 196, "frames": 1, "identical": False,
        "reference_file": "sweep_ref456_probe_00001_.png"})


def test_a_waived_render_on_a_label_k_cell_is_scored_on_the_waived_bar(monkeypatch, tmp_path):
    # A chroma nvfp4 cell refused K as its label said, the runner granted the
    # card, the waived leg rendered, and the probe read 0.1205 (2026-09-06). On
    # the step-nrms bar that render read CHECK over the 0.1 floor, which files
    # the deviation the waiver bought as a doubt. The contract held, so the
    # render is the PASS and the number is the result.
    _waived_leg_compare(monkeypatch)
    record = _waived_leg_record()
    run._fidelity(record, _waived_leg_cell(), tmp_path, 0.10)
    assert record["fidelity"]["bar"] == matrix.WAIVED_BAR
    assert record["verdict"] == "PASS" and run.settle_verdict(record) == "PASS"
    assert record["findings"] == []
    line = record["waived_render"]
    assert record["notes"] == [line] and line.startswith(f"{matrix.WAIVED_BAR} 0.120494 ")
    # The granted guard is the subject. TORCH_FLASH is exact, so calling it an
    # approximation would name the wrong reason for the number.
    assert "shard_quant_scale:chroma" in line and "no floor applies" in line
    assert "approximation" not in line and "TORCH_FLASH" not in line
    # A fault filed against the same cell still stands.
    record = _waived_leg_record()
    record.update({"verdict": "FINDING", "findings": ["warm leg: crash"]})
    run._fidelity(record, _waived_leg_cell(), tmp_path, 0.10)
    assert run.settle_verdict(record) == "FINDING"


def test_a_waived_leg_that_did_not_render_keeps_the_step_nrms_floor(monkeypatch, tmp_path):
    # The bar moves for a render under a granted card and for nothing else.
    _waived_leg_compare(monkeypatch)
    over_the_floor = ["step-nrms 0.120494 over the 0.1 floor against "
                      "sweep_ref456_probe_00001_.png"]
    # Neither a waived leg that refused again nor a missing waived leg moves it.
    for record in (_waived_leg_record("refuse:K"), _waived_leg_record(None)):
        run._fidelity(record, _waived_leg_cell(), tmp_path, 0.10)
        assert record["fidelity"]["bar"] == "step-nrms" and "waived_render" not in record
        assert record["findings"] == over_the_floor
        assert run.settle_verdict(record) == "CHECK"
    # A rendered leg on a cell carrying no waiver guard is an ordinary render.
    record = _waived_leg_record()
    run._fidelity(record, _waived_leg_cell(label="render", waiver=False, waiver_guards=[]),
                  tmp_path, 0.10)
    assert record["fidelity"]["bar"] == "step-nrms" and record["findings"] == over_the_floor
    # A pixel contract holds whatever the waiver admits: identical is the
    # residency question, and no card answers it.
    record = _waived_leg_record()
    run._fidelity(record, _waived_leg_cell(reference={"kind": "cell", "cell": "ref456",
                                                      "bar": "bit-identical"}),
                  tmp_path, 0.10)
    assert record["fidelity"]["bar"] == "bit-identical" and record["verdict"] == "FINDING"


def test_a_runtime_discovered_ring_pad_uses_the_waived_bar(monkeypatch, tmp_path):
    """A static render label may still encounter a driver-only token guard."""
    _waived_leg_compare(monkeypatch)
    cell = _waived_leg_cell(label="render", waiver=False, waiver_guards=["ring_pad"])
    record = _waived_leg_record(rows=[_use_row(guard="ring_pad", run_id="this-waived-run")])
    run._fidelity(record, cell, tmp_path, 0.10)
    assert record["fidelity"]["bar"] == matrix.WAIVED_BAR
    assert record["verdict"] == "PASS" and record["findings"] == []
    assert "ring_pad" in record["waived_render"]


@pytest.mark.parametrize("over", [
    {"granted": False},
    {"granted": "yes"},
    {"rows": []},
    {"rows": ["malformed"]},
    {"rows": [{"verdict": "PASS", "waiver_class": "K", "action": "use",
                "target_guard": "ring_pad"}]},
    {"rows": [{"verdict": "WAIVER", "waiver_class": "U", "action": "use",
                "target_guard": "ring_pad"}]},
    {"rows": [{"verdict": "WAIVER", "waiver_class": "K", "action": "use",
                "target_guard": "other_guard"}]},
    {"rows": [{"verdict": "WAIVER", "waiver_class": "K", "action": "grant",
                "target_guard": "ring_pad"}]},
    {"waived": "refuse:K"},
])
def test_runtime_discovered_waiver_requires_proven_success(monkeypatch, tmp_path, over):
    _waived_leg_compare(monkeypatch)
    cell = _waived_leg_cell(label="render", waiver=False, waiver_guards=["ring_pad"])
    record = _waived_leg_record(**over)
    run._fidelity(record, cell, tmp_path, 0.10)
    assert record["fidelity"]["bar"] == "step-nrms"
    assert "waived_render" not in record
    assert run.settle_verdict(record) == "CHECK"


def test_unrelated_expected_refusal_cannot_be_hidden_by_a_waiver(monkeypatch, tmp_path):
    _waived_leg_compare(monkeypatch)
    cell = _waived_leg_cell(label="refuse:P", waiver=False, waiver_guards=["ring_pad"])
    record = _waived_leg_record(rows=[_use_row(guard="ring_pad")])
    run._fidelity(record, cell, tmp_path, 0.10)
    assert record["fidelity"]["bar"] == "step-nrms"
    assert "waived_render" not in record


def test_dynamic_waiver_never_softens_a_bit_identical_contract(monkeypatch, tmp_path):
    _waived_leg_compare(monkeypatch)
    cell = _waived_leg_cell(label="render", waiver=False, waiver_guards=["ring_pad"],
                            reference={"kind": "cell", "cell": "ref456", "bar": "bit-identical"})
    record = _waived_leg_record(rows=[_use_row(guard="ring_pad", run_id="this-waived-run")])
    run._fidelity(record, cell, tmp_path, 0.10)
    assert record["fidelity"]["bar"] == "bit-identical"
    assert record["verdict"] == "FINDING"


def test_an_audio_compare_the_probe_could_not_make_is_reported(monkeypatch, tmp_path):
    # A one-step probe on an audio template can render near silence, which no
    # envelope correlates. Left unread, that nested error leaves the cell at
    # PASS with its audio evidence gone.
    monkeypatch.setattr(run, "compare_outputs", lambda *args, **kwargs: {
        "nrms": 0.0, "max_abs": 0, "frames": 1, "identical": True,
        "reference_file": "sweep_ref456_probe_00001_.png",
        "audio": {"error": "audio too short or too quiet to correlate"}})
    cell = _fidelity_cell(audio=True, probe=True,
                          reference={"kind": "cell", "cell": "ref456",
                                     "bar": "step-nrms", "probe_steps": 1})
    record = {"verdict": "PASS", "findings": [], "notes": []}
    run._fidelity(record, cell, tmp_path, 0.10)
    # Unproven, not a fault: the frames matched and the audio was unreadable.
    assert record["findings"] == ["reference compare: audio envelope: audio too short "
                                  "or too quiet to correlate"]
    assert run.settle_verdict(record) == "CHECK"


def test_the_step_nrms_floor_still_scores_a_probe_comparison(monkeypatch, tmp_path):
    cell = _fidelity_cell(probe=True, reference={"kind": "cell", "cell": "ref456",
                                                 "bar": "step-nrms", "probe_steps": 1})
    seen: list[tuple] = []

    def fake_compare(output_dir, candidate, reference, audio=False, nrms=0.0):
        seen.append((candidate, reference))
        return {"nrms": nrms, "max_abs": 3, "frames": 1,
                "reference_file": "sweep_ref456_probe_00001_.png"}

    for nrms, verdict in ((0.14, "CHECK"), (0.0, "PASS")):
        monkeypatch.setattr(run, "compare_outputs",
                            lambda *args, nrms=nrms, **kwargs: fake_compare(*args, nrms=nrms))
        record = {"verdict": "PASS", "findings": [], "notes": []}
        run._fidelity(record, cell, tmp_path, 0.10)
        assert record["verdict"] == verdict
    assert seen == [("sweep_abc123_probe", "sweep_ref456_probe")] * 2
    assert record["findings"] == []


def test_the_probe_leg_rewrites_a_scheduler_widget_and_every_stage(tmp_path):
    # The leg the runner queues, not the graph on disk: a probe leg that left
    # the schedule at its full length would render the warm leg over again.
    cell = {"id": "abc123", "unets": {}, "loras": [], "levers": {}, "mode": "cluster",
            "attention": "TORCH_FLASH", "gpus_per_host": 0, "batch_toggled": False,
            "batch_node": "", "batch": 1}
    graph = {"1": {"class_type": "Ideogram4Scheduler",
                   "inputs": {"steps": 20, "width": 1024}},
             "2": {"class_type": "ManualSigmas", "inputs": {"sigmas": "1.0, 0.5, 0.0"}},
             "3": {"class_type": "DGXMonarchKSamplerAdvanced",
                   "inputs": {"steps": 20, "start_at_step": 0, "end_at_step": 10,
                              "noise_seed": 7}},
             "4": {"class_type": "DGXMonarchKSamplerAdvanced",
                   "inputs": {"steps": 20, "start_at_step": 10, "end_at_step": 20,
                              "noise_seed": 7}}}
    warm = run.patch_graph(graph, cell, "auto", "first_use", 1, "sweep_abc123_warm")
    assert warm["1"]["inputs"]["steps"] == 20 and warm["2"]["inputs"]["sigmas"] == "1.0, 0.5, 0.0"
    probe = run.patch_graph(graph, cell, "auto", "first_use", 0, "sweep_abc123_probe", 1)
    assert probe["1"]["inputs"] == {"steps": 1, "width": 1024}
    assert probe["2"]["inputs"]["sigmas"] == "1.0, 0.0"
    assert [(probe[node]["inputs"]["steps"], probe[node]["inputs"]["start_at_step"],
             probe[node]["inputs"]["end_at_step"]) for node in ("3", "4")] == [(2, 0, 1), (2, 1, 2)]
    # The graph on disk is never touched, and the seed shift still lands.
    assert graph["1"]["inputs"]["steps"] == 20 and probe["3"]["inputs"]["noise_seed"] == 7
    assert warm["3"]["inputs"]["noise_seed"] == 8


# One box's /proc/meminfo. It holds SwapCached, which Cached takes without the
# start anchor, and the bare Active and Inactive totals beside the LRU fields.
MEMINFO = """MemTotal:       127533828 kB
MemFree:        36820016 kB
MemAvailable:   80573984 kB
Buffers:          370176 kB
Cached:         44875144 kB
SwapCached:            0 kB
Active:         21607668 kB
Inactive:       44401472 kB
Active(anon):    8109652 kB
Inactive(anon): 12966296 kB
Active(file):   13498016 kB
Inactive(file): 31435176 kB
Unevictable:       45324 kB
Mlocked:           45324 kB
SwapTotal:             0 kB
SwapFree:              0 kB
Dirty:               916 kB
Writeback:             0 kB
AnonPages:      20802768 kB
Shmem:            302448 kB
SReclaimable:     472804 kB
"""


def test_the_meminfo_parser_reads_every_field_a_capacity_leg_is_stated_in():
    values = run.parse_meminfo(MEMINFO)
    assert list(values) == list(run.MEMINFO_FIELDS) and len(values) == 18
    # MemAvailable and AnonPages, the whole record until 2026-09-03, read as
    # before, so an older reader still finds them.
    assert values["MemAvailable"] == 76.841 and values["AnonPages"] == 19.839
    assert round(values["Active(file)"] + values["Inactive(file)"], 3) == 42.852
    assert values["SwapTotal"] == 0.0 and values["Shmem"] == 0.288


def test_the_meminfo_pattern_takes_no_neighbouring_field():
    # The remote grep keeps what this pattern takes (MEMINFO_PATTERN says why
    # both ends are anchored).
    taken = [line.partition(":")[0] for line in MEMINFO.splitlines()
             if re.match(run.MEMINFO_PATTERN, line)]
    assert sorted(taken) == sorted(run.MEMINFO_FIELDS)
    assert "SwapCached" not in taken and "Active" not in taken and "Inactive" not in taken


def test_a_sample_row_carries_every_field_on_every_box(monkeypatch):
    monkeypatch.setattr(run, "_on", lambda host, command, timeout: (MEMINFO, ""))
    sampler = run.Sampler(["", "second-box"], Path("unused"), label="m1-leg3")
    handle = io.StringIO()
    row = sampler.sample_once(handle)
    assert row["leg"] == "m1-leg3" and isinstance(row["t"], float)
    for box in ("head", "second-box"):
        assert list(row[box]) == list(run.MEMINFO_FIELDS)
        assert sampler.summary[box]["Shmem"] == {"min": 0.29, "peak": 0.29, "last": 0.29}
    assert json.loads(handle.getvalue())["head"]["MemFree"] == 35.114


def test_a_box_that_answers_nothing_is_recorded_and_the_other_box_still_lands(monkeypatch):
    monkeypatch.setattr(run, "_on", lambda host, command, timeout:
                        ("", "rc 255: connection closed") if host else (MEMINFO, ""))
    sampler = run.Sampler(["", "second-box"], Path("unused"))
    row = sampler.sample_once(io.StringIO())
    assert "leg" not in row and row["second-box"] == {}
    assert row["head"]["MemAvailable"] == 76.841
    assert sampler.read_errors == {"second-box": "rc 255: connection closed"}


def test_the_memory_record_covers_both_boxes_on_a_world_one_leg():
    config, cell = {"sibling": "second-box"}, {"session": "local"}
    # session_hosts and sampling_hosts say why the two lists differ.
    assert run.session_hosts(config, cell) == [""]
    assert run.session_hosts(config, {"session": "cluster"}) == ["", "second-box"]
    assert run.sampling_hosts(config) == ["", "second-box"]
    assert run.sampling_hosts({"sibling": ""}) == [""]


def _split_pair(tmp_path, cfg_warm, single_warm, single_cfg=1):
    """One cfg cell and the single reference record its wall is read against."""
    (tmp_path / "cells").mkdir(parents=True, exist_ok=True)
    (tmp_path / "cells" / "single01.json").write_text(json.dumps({
        "cell": {"id": "single01", "resolved_cfg": single_cfg},
        "legs": {"warm": {"outcome": "render", "wall_s": single_warm}},
    }))
    cell = {"resolved_cfg": 2, "reference": {"kind": "cell", "cell": "single01"}}
    record = {"legs": {"warm": {"outcome": "render", "wall_s": cfg_warm}}}
    return record, cell


def _pinned_reference(tmp_path, pin: dict, ref_id: str = "single01", digest: str = "g1"):
    (tmp_path / "cells").mkdir(parents=True, exist_ok=True)
    (tmp_path / "cells" / f"{ref_id}.json").write_text(json.dumps({
        "cell": {"id": ref_id, "resolved_cfg": 1}, "pin": pin, "verdict": "PASS",
        "graph_digest": digest,
        "legs": {"warm": {"outcome": "render", "wall_s": 32.1}},
    }))


def test_a_reference_rendered_on_another_build_scores_check_not_finding(tmp_path):
    # A reference rendered by another build cannot establish a fidelity
    # finding for this build. Report the comparison as unproven.
    old = {"comfy_commit": "c", "source_digest": "old", "torch": "t", "nccl": "n", "world": 2}
    new = dict(old, source_digest="new")
    _pinned_reference(tmp_path, old)
    cell = {"id": "cfg01", "reference": {"kind": "cell", "cell": "single01", "bar": "bit-identical"},
            "probe": False, "audio": False, "resolved_cfg": 2}
    record = {"pin": new, "graph_digest": "g1", "verdict": "PASS", "findings": [], "notes": [],
              "legs": {"warm": {"outcome": "render", "wall_s": 34.0}}}
    run._fidelity(record, cell, tmp_path / "output", 0.1, tmp_path)
    assert record["verdict"] == "PASS"
    assert record["findings"] == [
        "reference compare: reference cell single01 last rendered on another build "
        "(source_digest moved); its image on disk is not this build's, so name it to "
        "re-render before this cell"]
    assert run.settle_verdict(record) == "CHECK"
    # The wall compare reads the same pin and prices nothing across builds.
    assert run.cfg_split_note(record, cell, tmp_path) == ""
    # A missing record says so too; a reference on this build, or a resident
    # leg, reads clean.
    assert run.stale_reference(record, {"kind": "cell", "cell": "nobody"}, tmp_path).startswith(
        "reference cell nobody has no record in this session")
    assert run.stale_reference({"pin": old, "graph_digest": "g1"}, cell["reference"], tmp_path) == ""
    assert run.stale_reference(record, {"kind": "resident-leg"}, tmp_path) == ""
    # The same pin with a moved graph digest is another build's image too
    # (stale_reference says why).
    assert "graph_digest moved" in run.stale_reference(
        {"pin": old, "graph_digest": "g2"}, cell["reference"], tmp_path)


def test_a_named_cell_pulls_its_stale_reference_in_before_it(tmp_path):
    # Seven reference cells sat outside W3c's named list (2026-09-08) on records
    # from earlier builds, and their dependents scored against them.
    # A stale or missing reference joins the run once, ahead of its first
    # dependent; a reference whose record resumes for this pin stays settled.
    pin = {"comfy_commit": "c", "source_digest": "d", "torch": "t", "nccl": "n", "world": 2}
    (tmp_path / "graphs").mkdir()
    (tmp_path / "graphs" / "chroma.json").write_text("{}")
    digest = run.graph_digest(tmp_path, "chroma")
    base = {"session": "cluster", "template": "chroma", "sampled_out": False, "label": "render",
            "reference": {"kind": "resident-leg"}}
    single = dict(base, id="single01")
    other = dict(base, id="single02")
    dep_a = dict(base, id="cfg01", reference={"kind": "cell", "cell": "single01", "bar": "x"})
    dep_b = dict(base, id="cfg02", reference={"kind": "cell", "cell": "single01", "bar": "x"})
    dep_c = dict(base, id="lever01", reference={"kind": "cell", "cell": "single02", "bar": "x"})
    cells = [single, other, dep_a, dep_b, dep_c]
    # single01 has no record; single02 has a settled record on this pin.
    (tmp_path / "cells").mkdir()
    (tmp_path / "cells" / "single02.json").write_text(json.dumps({
        "cell": {"id": "single02", "label": "render", "reference": {"kind": "resident-leg"}},
        "pin": pin, "graph_digest": digest, "verdict": "PASS", "observed": "render",
        "legs": {"warm": {"outcome": "render"}}}))
    ordered, pulled, blocked = run.with_stale_references([dep_a, dep_b, dep_c], cells, tmp_path, pin)
    assert [cell["id"] for cell in ordered] == ["single01", "cfg01", "cfg02", "lever01"]
    assert pulled == ["single01"]
    assert blocked == {}
    # A record from another build is stale the same way, and a reference the
    # operator named already is never doubled.
    (tmp_path / "cells" / "single02.json").write_text(json.dumps({
        "cell": {"id": "single02", "label": "render", "reference": {"kind": "resident-leg"}},
        "pin": dict(pin, source_digest="old"), "graph_digest": digest, "verdict": "PASS",
        "observed": "render", "legs": {"warm": {"outcome": "render"}}}))
    ordered, pulled, blocked = run.with_stale_references([single, dep_a, dep_c], cells, tmp_path, pin)
    assert [cell["id"] for cell in ordered] == ["single01", "cfg01", "single02", "lever01"]
    assert pulled == ["single02"]
    assert blocked == {}
    # The chain is walked to its root: a lever cell reads a cfg cell, which
    # reads its single, and a stale single comes first of all.
    chain_single = dict(base, id="one01")
    chain_cfg = dict(base, id="cfg09", reference={"kind": "cell", "cell": "one01", "bar": "x"})
    chain_lever = dict(base, id="lever09", reference={"kind": "cell", "cell": "cfg09", "bar": "x"})
    ordered, pulled, blocked = run.with_stale_references(
        [chain_lever], [*cells, chain_single, chain_cfg, chain_lever], tmp_path, pin)
    assert [cell["id"] for cell in ordered] == ["one01", "cfg09", "lever09"]
    assert pulled == ["one01", "cfg09"]
    assert blocked == {}


def test_skip_crashed_references_leaves_a_deterministic_crash_unpulled(tmp_path):
    # By design a stale reference joins the run once per session, whatever it
    # last answered (decided 2026-09-28). A caller that isolates OOM-prone
    # cells one per session then meets the same crash on every dependent's
    # session, so --skip-crashed-references leaves a deterministic crash on
    # this pin unpulled instead.
    pin = {"comfy_commit": "c", "source_digest": "d", "torch": "t", "nccl": "n", "world": 2}
    (tmp_path / "graphs").mkdir()
    (tmp_path / "graphs" / "flux2.json").write_text("{}")
    digest = run.graph_digest(tmp_path, "flux2")
    base = {"session": "cluster", "template": "flux2", "sampled_out": False, "label": "render",
            "reference": {"kind": "resident-leg"}}
    single = dict(base, id="single01")
    dep_a = dict(base, id="cfg01", reference={"kind": "cell", "cell": "single01", "bar": "x"})
    dep_b = dict(base, id="cfg02", reference={"kind": "cell", "cell": "single01", "bar": "x"})
    cells = [single, dep_a, dep_b]
    (tmp_path / "cells").mkdir()

    def write_single(observed: str, **over) -> None:
        record = {"cell": {"id": "single01", "label": "render",
                           "reference": {"kind": "resident-leg"}},
                  "pin": pin, "graph_digest": digest, "verdict": "FINDING",
                  "observed": observed, "legs": {"cold": {"outcome": observed}}}
        record.update(over)
        (tmp_path / "cells" / "single01.json").write_text(json.dumps(record))

    # Flag off: the crash is pulled in like any other stale reference.
    write_single("crash")
    ordered, pulled, blocked = run.with_stale_references([dep_a, dep_b], cells, tmp_path, pin)
    assert [cell["id"] for cell in ordered] == ["single01", "cfg01", "cfg02"]
    assert pulled == ["single01"] and blocked == {}

    # Flag on: the same crash is left unpulled, and both dependents are
    # blocked, naming it as the reason.
    ordered, pulled, blocked = run.with_stale_references(
        [dep_a, dep_b], cells, tmp_path, pin, skip_crashed=True)
    assert ordered == [] and pulled == []
    assert blocked == {"cfg01": "single01", "cfg02": "single01"}

    # A stale reference that is not a crash (here, never recorded) is still
    # pulled under the flag.
    (tmp_path / "cells" / "single01.json").unlink()
    ordered, pulled, blocked = run.with_stale_references(
        [dep_a, dep_b], cells, tmp_path, pin, skip_crashed=True)
    assert [cell["id"] for cell in ordered] == ["single01", "cfg01", "cfg02"]
    assert pulled == ["single01"] and blocked == {}

    # A crash recorded on another build says nothing about this pin.
    write_single("crash", pin=dict(pin, source_digest="old"))
    ordered, pulled, blocked = run.with_stale_references(
        [dep_a, dep_b], cells, tmp_path, pin, skip_crashed=True)
    assert [cell["id"] for cell in ordered] == ["single01", "cfg01", "cfg02"]
    assert blocked == {}

    # A wedged driver's crash is the rig's answer, not the cell's, and would
    # clear on a fresh driver: it is pulled like any other stale record.
    write_single("crash", legs={"cold": {"outcome": "crash",
                                         "message": "replacement is blocked"}})
    ordered, pulled, blocked = run.with_stale_references(
        [dep_a, dep_b], cells, tmp_path, pin, skip_crashed=True)
    assert [cell["id"] for cell in ordered] == ["single01", "cfg01", "cfg02"]
    assert blocked == {}

    # An operator who names the crashed reference still runs it, and its
    # dependents are not blocked.
    write_single("crash")
    ordered, pulled, blocked = run.with_stale_references(
        [single, dep_a], cells, tmp_path, pin, skip_crashed=True)
    assert [cell["id"] for cell in ordered] == ["single01", "cfg01"]
    assert pulled == [] and blocked == {}

    # A crash on the root blocks every link between it and the cell that
    # asked, not only its direct dependent.
    chain_single = dict(base, id="one01")
    chain_cfg = dict(base, id="cfg09", reference={"kind": "cell", "cell": "one01", "bar": "x"})
    chain_lever = dict(base, id="lever09", reference={"kind": "cell", "cell": "cfg09", "bar": "x"})
    (tmp_path / "cells" / "one01.json").write_text(json.dumps({
        "cell": {"id": "one01", "label": "render", "reference": {"kind": "resident-leg"}},
        "pin": pin, "graph_digest": digest, "verdict": "FINDING", "observed": "crash",
        "legs": {"cold": {"outcome": "crash"}}}))
    ordered, pulled, blocked = run.with_stale_references(
        [chain_lever], [chain_single, chain_cfg, chain_lever], tmp_path, pin, skip_crashed=True)
    assert ordered == [] and pulled == []
    assert blocked == {"cfg09": "one01", "lever09": "one01"}


def test_skip_crashed_references_propagates_a_block_between_two_selected_cells(tmp_path):
    # A crashed single01 blocks the selected cfg01, and that block must carry
    # to cfg02, a selected cell that reads cfg01; unblocked, cfg02 would run
    # against a reference that never rendered. It must hold in either order,
    # since `runnable` keeps the order of the operator's `--cells` list.
    pin = {"comfy_commit": "c", "source_digest": "d", "torch": "t", "nccl": "n", "world": 2}
    (tmp_path / "graphs").mkdir()
    (tmp_path / "graphs" / "flux2.json").write_text("{}")
    digest = run.graph_digest(tmp_path, "flux2")
    base = {"session": "cluster", "template": "flux2", "sampled_out": False, "label": "render",
            "waiver": False, "waiver_guards": []}
    single = dict(base, id="single01", reference={"kind": "resident-leg"})
    cfg01 = dict(base, id="cfg01", reference={"kind": "cell", "cell": "single01", "bar": "x"})
    cfg02 = dict(base, id="cfg02", reference={"kind": "cell", "cell": "cfg01", "bar": "x"})
    cells = [single, cfg01, cfg02]
    (tmp_path / "cells").mkdir()
    (tmp_path / "cells" / "single01.json").write_text(json.dumps({
        "cell": {"id": "single01", "label": "render", "reference": {"kind": "resident-leg"}},
        "pin": pin, "graph_digest": digest, "verdict": "FINDING",
        "observed": "crash", "legs": {"cold": {"outcome": "crash"}}}))
    for order in ([cfg01, cfg02], [cfg02, cfg01]):
        ordered, pulled, blocked = run.with_stale_references(
            order, cells, tmp_path, pin, skip_crashed=True)
        assert ordered == [] and pulled == [], order
        assert blocked == {"cfg01": "single01", "cfg02": "single01"}, order


def test_skip_crashed_references_never_blocks_a_dependent_that_cannot_compare(tmp_path):
    # A crashed reference must not block a typed refusal that is expected
    # never to read it. One that can render on a waived leg is blocked like a
    # render-labelled dependent (compares_against_reference says which).
    pin = {"comfy_commit": "c", "source_digest": "d", "torch": "t", "nccl": "n", "world": 2}
    (tmp_path / "graphs").mkdir()
    (tmp_path / "graphs" / "flux2.json").write_text("{}")
    digest = run.graph_digest(tmp_path, "flux2")
    base = {"session": "cluster", "template": "flux2", "sampled_out": False,
            "reference": {"kind": "cell", "cell": "single01", "bar": "x"}}
    single = {"session": "cluster", "template": "flux2", "sampled_out": False, "label": "render",
             "waiver": False, "waiver_guards": [], "reference": {"kind": "resident-leg"},
             "id": "single01"}
    refuses = dict(base, id="cfg01", label="refuse:P", waiver=False, waiver_guards=[])
    waivable_refuses = dict(base, id="cfg02", label="refuse:K", waiver=True, waiver_guards=[])
    guard_discoverable = dict(base, id="cfg03", label="refuse:P", waiver=False,
                              waiver_guards=["ring_pad"])
    cells = [single, refuses, waivable_refuses, guard_discoverable]
    (tmp_path / "cells").mkdir()
    (tmp_path / "cells" / "single01.json").write_text(json.dumps({
        "cell": {"id": "single01", "label": "render", "reference": {"kind": "resident-leg"}},
        "pin": pin, "graph_digest": digest, "verdict": "FINDING",
        "observed": "crash", "legs": {"cold": {"outcome": "crash"}}}))
    ordered, pulled, blocked = run.with_stale_references(
        [refuses, waivable_refuses, guard_discoverable], cells, tmp_path, pin, skip_crashed=True)
    # cfg01 never compares, so it runs unblocked; cfg02 and cfg03 might still
    # render on a waived leg, so they are blocked and excluded.
    assert [cell["id"] for cell in ordered] == ["cfg01"]
    assert pulled == []
    assert blocked == {"cfg02": "single01", "cfg03": "single01"}


def test_compares_against_reference_reads_the_label_and_the_waiver_fields():
    render = {"label": "render", "waiver": False, "waiver_guards": []}
    assert run.compares_against_reference(render)
    waivable = {"label": "refuse:K", "waiver": True, "waiver_guards": []}
    assert run.compares_against_reference(waivable)
    guard_discoverable = {"label": "refuse:P", "waiver": False, "waiver_guards": ["ring_pad"]}
    assert run.compares_against_reference(guard_discoverable)
    plain_refusal = {"label": "refuse:untyped", "waiver": False, "waiver_guards": []}
    assert not run.compares_against_reference(plain_refusal)


def test_crashed_on_pin_reads_the_record_the_way_recoverable_crash_does():
    # crashed_on_pin asks resumable's own question (record_current_for), not
    # RESUME_KEYS alone: a crash against an older graph, a moved label or a
    # moved reference plan says nothing about what this reference would answer
    # now.
    pin = {"comfy_commit": "c", "source_digest": "d", "torch": "t", "nccl": "n", "world": 2}
    reference = {"kind": "cell", "cell": "single01", "bar": "x"}
    crash = {"observed": "crash", "pin": pin, "graph_digest": "g1", "verdict": "FINDING",
            "cell": {"label": "render", "reference": reference},
            "legs": {"cold": {"outcome": "crash"}}}
    assert run.crashed_on_pin(crash, pin, "g1", "render", reference)
    assert not run.crashed_on_pin({**crash, "observed": "refuse:P"}, pin, "g1", "render", reference)
    assert not run.crashed_on_pin({**crash, "pin": dict(pin, world=4)}, pin, "g1", "render", reference)
    assert not run.crashed_on_pin(crash, pin, "g2", "render", reference)
    assert not run.crashed_on_pin(crash, pin, "g1", "refuse:P", reference)
    other_reference = {"kind": "cell", "cell": "single02", "bar": "x"}
    assert not run.crashed_on_pin(crash, pin, "g1", "render", other_reference)
    wedged = {**crash, "legs": {"cold": {"outcome": "crash",
                                         "message": "not reusable after a partial bring-up"}}}
    assert not run.crashed_on_pin(wedged, pin, "g1", "render", reference)
    # A crash a fresh fleet has already answered before is the rig meeting a
    # dirty mesh an earlier cell left behind, not this reference
    # deterministically failing, so it is tried again too.
    for kind in run.RECOVERABLE_CRASHES:
        recoverable = {**crash, "legs": {"cold": {
            "outcome": "crash", "exception_type": f"dgx_monarch.mesh.{kind}"}}}
        assert not run.crashed_on_pin(recoverable, pin, "g1", "render", reference), kind


def test_block_crashed_dependents_never_overwrites_a_resumable_record(tmp_path):
    # A dependent whose own settled record already resumes for this pin keeps
    # it: the blocked stamp is only for a cell that would otherwise re-run.
    pin = {"comfy_commit": "c", "source_digest": "d", "torch": "t", "nccl": "n", "world": 2}
    (tmp_path / "graphs").mkdir()
    (tmp_path / "graphs" / "flux2.json").write_text("{}")
    digest = run.graph_digest(tmp_path, "flux2")
    reference = {"kind": "cell", "cell": "single01", "bar": "x"}
    dep_a = {"id": "cfg01", "template": "flux2", "label": "render", "reference": reference}
    (tmp_path / "cells").mkdir()
    (tmp_path / "cells" / "cfg01.json").write_text(json.dumps({
        "cell": {"id": "cfg01", "label": "render", "reference": reference},
        "pin": pin, "graph_digest": digest, "verdict": "PASS", "observed": "render",
        "legs": {"warm": {"outcome": "render"}}}))
    written = run.block_crashed_dependents(tmp_path, [dep_a], pin, {"cfg01": "single01"})
    assert written == []
    record = json.loads((tmp_path / "cells" / "cfg01.json").read_text())
    assert record["verdict"] == "PASS" and record["observed"] == "render"


def test_block_crashed_dependents_keeps_an_unsettled_attempt(tmp_path):
    # An existing but unsettled record, such as the dependent's own earlier
    # crash, is retired under .attemptN, as main()'s loop does for any other
    # unsettled record.
    pin = {"comfy_commit": "c", "source_digest": "d", "torch": "t", "nccl": "n", "world": 2}
    (tmp_path / "graphs").mkdir()
    (tmp_path / "graphs" / "flux2.json").write_text("{}")
    dep_a = {"id": "cfg01", "template": "flux2", "label": "render",
            "reference": {"kind": "cell", "cell": "single01", "bar": "x"}}
    (tmp_path / "cells").mkdir()
    (tmp_path / "cells" / "cfg01.json").write_text(json.dumps({"observed": "crash"}))
    written = run.block_crashed_dependents(tmp_path, [dep_a], pin, {"cfg01": "single01"})
    assert written == ["cfg01"]
    assert (tmp_path / "cells" / "cfg01.attempt1.json").is_file()
    record = json.loads((tmp_path / "cells" / "cfg01.json").read_text())
    assert record["observed"] == "blocked" and record["verdict"] == "FINDING"
    assert "single01" in record["findings"][0]


def test_report_renders_a_blocked_record():
    from benchmark.sweep import report

    record = run.blocked_record(
        {"id": "cfg01", "label": "render", "template": "flux2", "preset": "cfg2",
         "attention": "TORCH_FLASH", "session": "cluster"},
        {"comfy_commit": "c"}, "g1", "single01")
    payload = report.build([record], "cluster")
    assert payload["rows"][0]["observed"] == "blocked"
    assert any("single01" in line for line in payload["findings"])


def test_skip_crashed_references_blocks_the_dependent_without_dispatch(monkeypatch, tmp_path):
    # End to end through main(), as a validation controller running --cells
    # cfg01 alone (one cell per session) would call it: the dependent must
    # never reach run_cell once its reference is a crash left unpulled.
    out = tmp_path / "out"
    (out / "graphs").mkdir(parents=True)
    (out / "graphs" / "flux2.json").write_text("{}")
    single = {"id": "single01", "template": "flux2", "session": "cluster",
             "sampled_out": False, "label": "render", "klass": "image", "levers": {},
             "preset": "auto", "mode": "cluster", "attention": "TORCH_FLASH",
             "waiver": False, "waiver_guards": [], "audio": False,
             "reference": {"kind": "resident-leg"}}
    dep_a = {"id": "cfg01", "template": "flux2", "session": "cluster",
            "sampled_out": False, "label": "render", "klass": "image", "levers": {},
            "preset": "cfg2", "mode": "cluster", "attention": "TORCH_FLASH",
            "waiver": False, "waiver_guards": [], "audio": False,
            "reference": {"kind": "cell", "cell": "single01", "bar": "x"}}
    (out / "cells.jsonl").write_text(json.dumps(single) + "\n" + json.dumps(dep_a) + "\n")
    (out / "cells").mkdir()
    (out / "cells" / "single01.json").write_text(json.dumps({
        "cell": {"id": "single01", "label": "render", "reference": {"kind": "resident-leg"}},
        "pin": {}, "graph_digest": run.graph_digest(out, "flux2"), "verdict": "FINDING",
        "observed": "crash", "legs": {"cold": {"outcome": "crash"}}}))
    config = tmp_path / "sweep.toml"
    config.write_text(f'out_dir = "{out}"\ncomfy_dir = "{tmp_path}"\n')
    monkeypatch.setattr(run, "session_pin", lambda config: {})
    monkeypatch.setattr(run, "recycle", lambda driver: "ok")
    monkeypatch.setattr(run, "run_cell", lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("a blocked dependent must not be dispatched")))
    monkeypatch.setattr(run.time, "sleep", lambda seconds: None)
    assert run.main(["--config", str(config), "--session", "cluster",
                     "--cells", "cfg01", "--skip-crashed-references"]) == 0
    summary = json.loads((out / "run_summary_cluster.json").read_text())
    assert summary["blocked"] == 1 and summary["ran"] == 0
    pruned = json.loads((out / "run_pruned_cluster.json").read_text())
    assert pruned["blocked_on_crashed_reference"] == ["cfg01"]
    record = json.loads((out / "cells" / "cfg01.json").read_text())
    assert record["observed"] == "blocked" and record["verdict"] == "FINDING"
    assert "single01" in record["findings"][0]


def test_a_cfg_cell_that_did_not_split_reads_check(tmp_path):
    # Only the wall shows a cfg split that never ran (cfg_split_note says why).
    # W3 core measured boogu 50.1 s against a 50.1 s single, ernie 50.1 against
    # 48.2 and omnigen2 26.0 against 26.0, while every split family read 1.7 to
    # 2.4x (2026-09-04).
    record, cell = _split_pair(tmp_path, 50.1, 50.1)
    line = run.cfg_split_note(record, cell, tmp_path)
    assert line.startswith("cfg2 did not split: a 50.1 s warm leg against a 50.1 s single")
    assert "single01" in line
    # settle_verdict alone files the line as a fault, so the fidelity scorer
    # sets CHECK before filing it (tested below).
    scored = {"verdict": "PASS", "findings": [line]}
    assert run.settle_verdict(scored) == "FINDING"


def test_a_wall_just_inside_the_band_still_reads_as_no_split(tmp_path):
    # ernie measured 50.1 against 48.2, which is 1.04x and inside ten percent.
    record, cell = _split_pair(tmp_path, 50.1, 48.2)
    assert run.cfg_split_note(record, cell, tmp_path).startswith("cfg2 did not split")


def test_a_split_that_landed_makes_no_claim(tmp_path):
    # 1.7x is the slowest split this fleet has measured; anything under 0.9 of
    # the single wall is a split that ran.
    record, cell = _split_pair(tmp_path, 29.0, 50.0)
    assert run.cfg_split_note(record, cell, tmp_path) == ""


def test_the_rule_reads_only_a_cfg_cell_against_a_single_reference(tmp_path):
    record, cell = _split_pair(tmp_path, 50.1, 50.1)
    # A cell whose topology carries no cfg degree prices nothing here.
    assert run.cfg_split_note(record, {**cell, "resolved_cfg": 1}, tmp_path) == ""
    # A resident-leg reference is the same cell at another residency, not a
    # single render, so it cannot price one rank's work.
    assert run.cfg_split_note(
        record, {**cell, "reference": {"kind": "resident-leg", "preset": "cfg2"}},
        tmp_path) == ""
    # A lever cell scores against the auto row, which may itself be cfg.
    both_cfg, cell_two = _split_pair(tmp_path, 50.1, 50.1, single_cfg=2)
    assert run.cfg_split_note(both_cfg, cell_two, tmp_path) == ""
    # A missing warm render or reference record gives no reading.
    assert run.cfg_split_note(
        {"legs": {"warm": {"outcome": "crash", "wall_s": 2.0}}}, cell, tmp_path) == ""
    assert run.cfg_split_note(record, {**cell, "reference": {
        "kind": "cell", "cell": "absent"}}, tmp_path) == ""


def test_the_fidelity_scorer_files_the_split_line_at_check(tmp_path, monkeypatch):
    """The line is scored where the other fidelity notes are."""
    record, cell = _split_pair(tmp_path, 50.1, 50.1)
    record.update(verdict="PASS", findings=[], notes=[])
    cell.update(id="cfg01", audio=False, probe=False, reference={
        "kind": "cell", "cell": "single01", "bar": "step-nrms"})
    monkeypatch.setattr(run, "compare_outputs",
                        lambda *args, **kwargs: {"nrms": 0.0, "reference_file": "ref.png"})
    run._fidelity(record, cell, tmp_path, 0.1, tmp_path)
    assert record["verdict"] == "CHECK"
    assert any(line.startswith("cfg2 did not split") for line in record["findings"])


def test_resolved_cfg_degree_reads_the_preset_and_the_auto_row():
    cell = {"preset": "cfg2", "family": "flux", "quant": "bf16", "batch": 1}
    facts = {"megapixels": 1.0, "cfg": 3.5}
    assert matrix.resolved_cfg_degree(cell, facts, 2) == 2
    assert matrix.resolved_cfg_degree({**cell, "preset": "uly2"}, facts, 2) == 1
    # An auto row answers from the table, not from the preset name.
    auto = matrix.resolved_cfg_degree({**cell, "preset": "auto"}, facts, 2)
    assert auto in (1, 2)
    # A preset the runtime refuses prices nothing; label_cell already scored it.
    assert matrix.resolved_cfg_degree({**cell, "preset": "nonsense"}, facts, 2) == 1


def _use_row(guard: str = "shard_quant_scale:chroma", action: str = "use", *,
             run_id: str = "this-cold-run") -> dict:
    """One WAIVER row as the gate ledger writes it when a guard spends a grant."""
    return {"waiver_kind": "waive-known-wrong:shard-quant", "waiver_class": "K",
            "target_guard": guard, "action": action, "verdict": "WAIVER",
            "run_id": run_id,
            "memo_context": '{"combo_key":"c","topology":"cfg2","world":"2"}'}


def _carried_grant_record(rows=None, cold: str = "render") -> dict:
    """A cell that never refused: the cold leg rendered under a live grant.

    It has the shape docs/VALIDATION.md records for three sweep runs
    on 2026-09-06: no waived leg and no consent
    this cell granted, because the guard passed under a grant an earlier cell left
    live and the cold leg never refused.
    """
    return {"verdict": "PASS", "findings": [], "notes": [], "observed": "render",
            "legs": {"cold": {"outcome": cold, "waiver_run_ids": ["this-cold-run"]},
                     "warm": {"outcome": "render"},
                     "probe": {"outcome": "render"}},
            "ledger_rows": [_use_row()] if rows is None else rows}


def test_a_cold_leg_that_rendered_under_a_carried_grant_takes_the_waived_bar(
        monkeypatch, tmp_path):
    # carried_grant_render says why such a cell never refuses and what proves
    # the guard fired.
    _waived_leg_compare(monkeypatch)
    record = _carried_grant_record()
    run._fidelity(record, _waived_leg_cell(), tmp_path, 0.10)
    assert record["fidelity"]["bar"] == matrix.WAIVED_BAR
    assert record["verdict"] == "PASS" and run.settle_verdict(record) == "PASS"
    assert record["findings"] == []
    line = record["waived_render"]
    assert record["notes"] == [line] and line.startswith(f"{matrix.WAIVED_BAR} 0.120494 ")
    # The reason names the grant that was already live. This runner granted no
    # card in this cell, so claiming it did would name the wrong event.
    assert "already live" in line and "shard_quant_scale:chroma" in line
    assert "the runner granted" not in line and "no floor applies" in line


def test_a_dynamic_render_label_can_carry_a_matching_live_grant(monkeypatch, tmp_path):
    _waived_leg_compare(monkeypatch)
    cell = _waived_leg_cell(label="render", waiver=False, waiver_guards=["ring_pad"])
    record = _carried_grant_record(rows=[_use_row(guard="ring_pad")])
    run._fidelity(record, cell, tmp_path, 0.10)
    assert record["fidelity"]["bar"] == matrix.WAIVED_BAR
    assert "already live" in record["waived_render"]


def test_a_waiver_use_row_must_name_the_rendered_leg(monkeypatch, tmp_path):
    """An interleaved prompt spending ring_pad cannot change this cell's bar."""
    _waived_leg_compare(monkeypatch)
    cell = _waived_leg_cell(label="render", waiver=False, waiver_guards=["ring_pad"])
    foreign = _use_row(guard="ring_pad", run_id="another-prompt")

    fresh = _waived_leg_record(rows=[foreign])
    run._fidelity(fresh, cell, tmp_path, 0.10)
    assert fresh["fidelity"]["bar"] == "step-nrms"
    assert run.settle_verdict(fresh) == "CHECK"

    carried = _carried_grant_record(rows=[foreign])
    run._fidelity(carried, cell, tmp_path, 0.10)
    assert carried["fidelity"]["bar"] == "step-nrms"
    assert run.settle_verdict(carried) == "CHECK"


def test_interleaved_waiver_rows_keep_the_matching_render_proof(monkeypatch, tmp_path):
    _waived_leg_compare(monkeypatch)
    cell = _waived_leg_cell(label="render", waiver=False, waiver_guards=["ring_pad"])
    foreign = _use_row(guard="ring_pad", run_id="another-prompt")

    fresh = _waived_leg_record(rows=[foreign, _use_row(
        guard="ring_pad", run_id="this-waived-run")])
    run._fidelity(fresh, cell, tmp_path, 0.10)
    assert fresh["fidelity"]["bar"] == matrix.WAIVED_BAR

    carried = _carried_grant_record(rows=[foreign, _use_row(
        guard="ring_pad", run_id="this-cold-run")])
    run._fidelity(carried, cell, tmp_path, 0.10)
    assert carried["fidelity"]["bar"] == matrix.WAIVED_BAR


@pytest.mark.parametrize("row", [
    {"verdict": "WAIVER", "waiver_class": "K", "action": "use", "target_guard": "ring_pad"},
    {"verdict": "WAIVER", "waiver_class": "K", "action": "use", "target_guard": "ring_pad",
     "run_id": 7},
    {"verdict": "WAIVER", "waiver_class": "K", "action": "use", "target_guard": ["ring_pad"],
     "run_id": "this-waived-run"},
])
def test_a_waiver_use_without_a_valid_render_identity_keeps_the_bar(monkeypatch, tmp_path, row):
    _waived_leg_compare(monkeypatch)
    cell = _waived_leg_cell(label="render", waiver=False, waiver_guards=["ring_pad"])
    record = _waived_leg_record(rows=[row])
    run._fidelity(record, cell, tmp_path, 0.10)
    assert record["fidelity"]["bar"] == "step-nrms"
    assert run.settle_verdict(record) == "CHECK"


def test_sampler_ui_dispatch_ids_belong_to_its_prompt_only():
    entry = {"outputs": {
        "7": {"dgxm_waiver_runs": ["this-run", "this-run", 7, ""]},
        "8": {"dgxm_waiver_runs": "not-a-list"},
        "9": "not-an-output",
    }}
    assert waiver_run_ids(entry) == ["this-run"]


def test_sampler_uses_its_stamped_dispatch_id_as_prompt_ui_evidence():
    output = {STAMPED_RESULT_KEY: [
        {"run_id": "this-run", "guard": "ring_pad"},
        {"run_id": "ancestor-run", "guard": "ring_pad", "inherited": True},
    ]}
    result = _sampler_result(output)
    assert result["ui"] == {"dgxm_waiver_runs": ["this-run"]}
    assert result["result"] == (output,)
    inherited_only = {STAMPED_RESULT_KEY: [
        {"run_id": "ancestor-run", "guard": "ring_pad", "inherited": True},
    ]}
    assert _sampler_result(inherited_only) == (inherited_only,)


def test_the_carried_grant_is_read_from_the_ledger_rows_and_nothing_else():
    # This run has no reference comparison, so the bar never runs on
    # it and this predicate is the whole of what the verdict reads.
    cell = _waived_leg_cell(reference={"kind": "none", "bar": "reference"})
    assert run.carried_grant_render(_carried_grant_record(), cell) is True

    # A cell that did refuse takes the ordinary waived-leg path, not this one.
    assert run.carried_grant_render(
        _carried_grant_record(cold="refuse:K"), cell) is False
    # No row, a row for a guard this cell's label does not carry, and a grant
    # row rather than a spend: none of the three is evidence the guard fired.
    assert run.carried_grant_render(_carried_grant_record(rows=[]), cell) is False
    assert run.carried_grant_render(
        _carried_grant_record(rows=[_use_row(guard="ring_pad")]), cell) is False
    assert run.carried_grant_render(
        _carried_grant_record(rows=[_use_row(action="grant")]), cell) is False
    # A cell the matrix never labelled refuse:K is an ordinary render.
    assert run.carried_grant_render(
        _carried_grant_record(),
        _waived_leg_cell(label="render", waiver=False, waiver_guards=[])) is False


def _sampled(available: float, **over) -> dict:
    """One memory sample, both boxes, with the head the lower reading."""
    leg = {"outcome": "render", "wall_s": 40.0, "memory": {
        "head": {"MemAvailable": {"min": available, "peak": 110.0, "last": available},
                 "AnonPages": {"min": 3.0, "peak": 30.0, "last": 20.0}},
        "second": {"MemAvailable": {"min": available + 2, "peak": 110.0,
                                    "last": available + 2}}}}
    leg.update(over)
    return leg


def _last(observed: str = "render", load: str = "", available: float | None = 100.0,
          template: str = "hunyuan", managed: str = "off") -> run.LastCell:
    return run.LastCell(template, managed, observed, load, available)


def _next_cell(gib: float = 16.2, template: str = "hunyuan", managed: str = "off") -> dict:
    return {"template": template, "levers": {"comfy_managed": managed},
            "checkpoint_gib": gib}


def test_the_box_a_cell_hands_on_is_its_last_leg_lowest_reading():
    # The last leg ran nearest the handover, and the lower box rules: a cell
    # runs on both, so the one with less room is the one that answers.
    record = {"legs": {"cold": _sampled(90.0), "warm": _sampled(61.4)}}
    assert run.final_available_gib(record) == 61.4
    # No sample, an unreadable sample and a host that gave no reading all
    # answer None. A missing reading is not a full box.
    assert run.final_available_gib({"legs": {}}) is None
    assert run.final_available_gib({"legs": {"cold": {"outcome": "render"}}}) is None
    assert run.final_available_gib(
        {"legs": {"cold": _sampled(90.0),
                  "warm": {"memory": {"memory_error": "the sampler thread did not stop"}}}}
    ) is None
    torn = {"legs": {"cold": _sampled(90.0)}}
    torn["legs"]["cold"]["memory"]["second"] = {"read_error": "rc 255: transport lost"}
    assert run.final_available_gib(torn) is None


def test_the_memory_a_cell_needs_is_the_runner_floor_or_its_own_price():
    # The floor rules a small checkpoint and the stock price rules above it
    # (fleet_need_gib says why the harness charges stock).
    from dgx_monarch.capacity_fit import stock_required_bytes

    price_60 = round(stock_required_bytes(60 * (1 << 30)) / (1 << 30), 1)
    price_32_5 = round(stock_required_bytes(int(32.5 * (1 << 30))) / (1 << 30), 1)

    assert run.fleet_need_gib({"checkpoint_gib": 16.2}, 60.0) == 60.0
    assert run.fleet_need_gib({"checkpoint_gib": 60.0}, 60.0) == price_60
    assert run.fleet_need_gib({}, 60.0) == 60.0
    assert run.fleet_need_gib({"checkpoint_gib": 60.0}, 0.0) == price_60
    # The file plus the host floor alone is what a slab load pays, and a cell
    # priced at that would be admitted onto a box a stock load refuses.
    assert run.fleet_need_gib({"checkpoint_gib": 32.5}, 60.0) == price_32_5


def test_a_rendered_cell_keeps_its_fleet_only_where_the_residue_cannot_answer():
    from dgx_monarch.capacity_fit import stock_required_bytes

    price_32_5 = round(stock_required_bytes(int(32.5 * (1 << 30))) / (1 << 30), 1)
    price_60 = round(stock_required_bytes(60 * (1 << 30)) / (1 << 30), 1)

    # W3c ran 390 cells and replaced the fleet before 380 of them (recorded
    # 2026-09-09). A render that left the box clear of both bars cannot move
    # the next cell's guard, which prices its checkpoint against MemAvailable
    # and nothing else.
    assert run.needs_fleet_reset(_last(available=105.6), _next_cell(), 60.0) == ""
    # A measured pair that needs a model-specific rule: the first run grew
    # anonymous memory on the head by 65.7 GiB during its cold leg and handed on
    # 50.95 GiB. The next run uses the same 32.5 GiB checkpoint. The
    # runner replaces the fleet.
    short = run.needs_fleet_reset(_last(available=50.95), _next_cell(gib=32.5), 60.0)
    assert short == f"the fleet left 50.95 GiB available, under the {price_32_5} GiB this cell needs"
    # The same pair on a roomier box. The floor is clear and the price is not,
    # so the price alone answers here; the file plus the host floor would not.
    assert run.needs_fleet_reset(_last(available=62.0), _next_cell(gib=32.5), 60.0) == \
        f"the fleet left 62.0 GiB available, under the {price_32_5} GiB this cell needs"
    # A checkpoint bigger than the floor raises the bar well above it.
    assert run.needs_fleet_reset(_last(available=62.0), _next_cell(gib=60.0), 60.0) == \
        f"the fleet left 62.0 GiB available, under the {price_60} GiB this cell needs"
    assert run.needs_fleet_reset(_last(available=62.0), _next_cell(gib=16.2), 60.0) == ""
    # No reading is not a full box.
    assert run.needs_fleet_reset(_last(available=None), _next_cell(), 60.0) == \
        "the previous cell gave no memory reading"


def test_a_kept_fleet_hands_its_own_load_to_the_cell_after_it():
    # A growth signal misses this: cell B inherits A's load, adds nothing of
    # its own and reads no growth. The rule reads the box, so B hands on the
    # whole inherited figure and C is replaced on it.
    inherited = {"legs": {"cold": _sampled(38.0, wall_s=14.0)}}
    inherited["legs"]["cold"]["memory"]["head"]["AnonPages"] = {
        "min": 30.0, "peak": 30.4, "last": 30.2}
    assert run.load_evidence(inherited) == ""
    assert run.final_available_gib(inherited) == 38.0
    kept = run.LastCell("hunyuan", "off", "render", run.load_evidence(inherited),
                        run.final_available_gib(inherited))
    assert run.needs_fleet_reset(kept, _next_cell(), 60.0) == \
        "the fleet left 38.0 GiB available, under the 60.0 GiB this cell needs"


def _session_cell(cell_id: str, gib: float) -> dict:
    return {"id": cell_id, "template": "t", "session": "cluster", "sampled_out": False,
            "label": "render", "klass": "image", "levers": {}, "preset": "auto",
            "mode": "cluster", "attention": "TORCH_FLASH", "waiver": False,
            "checkpoint_gib": gib, "waiver_guards": [],
            "reference": {"kind": "none", "bar": "reference"}}


def _run_two_cells(monkeypatch, tmp_path, available: float, gib: float) -> list:
    """One session, two cells of one template; returns the reset each was handed."""
    out = tmp_path / "out"
    (out / "graphs").mkdir(parents=True)
    (out / "graphs" / "t.json").write_text(json.dumps({}))
    (out / "cells.jsonl").write_text(
        "\n".join(json.dumps(_session_cell(name, gib)) for name in ("aaa111", "bbb222")))
    config = tmp_path / "sweep.toml"
    config.write_text(f'out_dir = "{out}"\ncomfy_dir = "{tmp_path}"\nmem_floor_gib = 60.0\n')
    seen: list = []

    def _cell(cell, config, pin, client_id, args, reset=None, watch=None):
        seen.append(reset)
        return {"observed": "render", "verdict": "PASS", "findings": [],
                "legs": {"cold": _sampled(available)}}

    monkeypatch.setattr(run, "session_pin", lambda config: {})
    monkeypatch.setattr(run, "recycle", lambda driver: "ok")
    monkeypatch.setattr(run.SessionWatch, "mesh_block", lambda self: {})
    monkeypatch.setattr(run, "probe_transport", lambda *args, **kwargs: "cluster")
    monkeypatch.setattr(run, "reset_fleet", lambda *args, **kwargs: {"recycle": "ok"})
    monkeypatch.setattr(run, "run_cell", _cell)
    monkeypatch.setattr(run.time, "sleep", lambda seconds: None)
    assert run.main(["--config", str(config), "--session", "cluster"]) == 0
    return seen


def test_the_session_keeps_the_fleet_after_a_render_that_left_the_box_clear(
        monkeypatch, tmp_path):
    # The first cell of a session has nothing before it. The second inherits a
    # box with 96 GiB left against a 60 GiB floor, so it runs on the same
    # fleet and its record says so.
    first, second = _run_two_cells(monkeypatch, tmp_path, available=96.0, gib=16.2)
    assert first is None
    assert second == {"fleet_kept": True, "available_gib": 96.0, "need_gib": 60.0}


def test_the_session_still_replaces_the_fleet_where_the_residue_could_answer(
        monkeypatch, tmp_path):
    # Same two cells, and a first cell that handed on 41 GiB. That is under the
    # floor the second cell must clear, so the runner first replaces the fleet.
    _first, second = _run_two_cells(monkeypatch, tmp_path, available=41.0, gib=16.2)
    assert second == {"recycle": "ok", "reason": "the fleet left 41.0 GiB available, "
                                                 "under the 60.0 GiB this cell needs"}
