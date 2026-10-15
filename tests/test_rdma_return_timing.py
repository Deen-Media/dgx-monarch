"""The qualifier's --timing mode: plan, private evidence, and the paired summary.

The hardware run reuses the default mode's admission, return path and
ownership checks; these tests cover what can run without a fabric.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import os
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
SHA, MANIFEST = "a" * 40, "b" * 64


def _bench():
    spec = importlib.util.spec_from_file_location(
        "dgxm_rdma_timing_under_test", REPO / "benchmark" / "rdma_return_bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config():
    return SimpleNamespace(
        client_bind="tcp://192.0.2.1:0",
        hosts=[SimpleNamespace(name="spark-one", address="tcp://192.0.2.1:26600"),
               SimpleNamespace(name="spark-two", address="tcp://192.0.2.2:26600")])


def test_default_mode_parses_without_timing():
    args = _bench().parse_args(["--expect-sha", SHA, "--expect-manifest", MANIFEST])
    assert args.timing is False and args.timing_summary is None
    assert args.timing_sizes_mib == (8, 32, 64) and args.timing_repeats == 7


def test_the_timing_plan_leads_with_first_use_then_warm_repeats_per_size():
    module = _bench()
    plan = module.timing_plan((8, 32, 64), 7)
    assert plan[0] == ("first_use", module.THRESHOLD, 0)
    assert len(plan) == 22
    for size in (8, 32, 64):
        assert [r for phase, s, r in plan if phase == "warm" and s == size * module.MIB] == list(range(7))


@pytest.mark.parametrize("mode,position,order", [
    ("message", "first", "message-first"), ("native", "second", "message-first"),
    ("native", "first", "native-first"), ("message", "second", "native-first")])
def test_each_run_names_its_campaign_order(mode, position, order):
    assert _bench().timing_order(mode, position) == order


def _parse(module, *extra):
    return module.parse_args(["--expect-sha", SHA, "--expect-manifest", MANIFEST, *extra])


def test_the_default_plan_fits_the_default_staging_budget():
    module = _bench()
    args = _parse(module, "--timing", "--timing-position", "first")
    assert args.timing_budget_gib == 8
    assert module.timing_staged_bytes(args.timing_sizes_mib, args.timing_repeats) == 2944 * module.MIB
    module.admit_timing_args(args)


def test_a_plan_above_the_staging_budget_refuses_until_the_budget_is_raised():
    module = _bench()
    plan = ("--timing", "--timing-position", "first",
            "--timing-sizes-mib", "64,128,256", "--timing-repeats", "20")
    with pytest.raises(module.QualificationRefused, match="budget"):
        module.admit_timing_args(_parse(module, *plan))
    module.admit_timing_args(_parse(module, *plan, "--timing-budget-gib", "36"))
    with pytest.raises(module.QualificationRefused, match="at least 1"):
        module.admit_timing_args(_parse(module, *plan, "--timing-budget-gib", "0"))


@pytest.mark.parametrize("flag", [
    ("--timing-position", "first"), ("--timing-sizes-mib", "8,32"),
    ("--timing-repeats", "7"), ("--timing-budget-gib", "8")])
def test_a_timing_flag_without_timing_refuses(flag):
    module = _bench()
    args = _parse(module, *flag)
    assert args.timing_flags_given == [flag[0]]
    with pytest.raises(module.QualificationRefused, match="need --timing"):
        module.admit_timing_args(args)
    module.admit_timing_args(_parse(module))  # the default mode still passes


@pytest.mark.parametrize("flag", [
    ("--timing",), ("--timing-repeats", "7"), ("--config", "cluster.toml"),
    ("--output", "run.json"), ("--preflight-only",)])
def test_the_summary_refuses_run_flags_it_would_ignore(flag):
    module = _bench()
    with pytest.raises(module.QualificationRefused, match="reports only"):
        module.main(["--expect-sha", SHA, "--expect-manifest", MANIFEST,
                     "--timing-summary", "unread.json", *flag])


class _Root:
    """A guard or future stand-in that can be weakly referenced."""


def _timed_fakes(module, events):
    """Fake dispatch, consume and facts that record the hook's calls."""
    import torch

    row, refs = {}, []

    def dispatch(size, seed, depth, roots):
        guard, future = _Root(), _Root()
        tensor = torch.zeros(4, dtype=torch.uint8)
        result = {"latent": {"kind": "message", "tensor": tensor}}
        roots += [guard, future, result]
        refs.append((weakref.ref(guard), weakref.ref(future), weakref.ref(tensor)))
        events.append(("dispatch", size // module.MIB, seed, depth))
        return guard, result["latent"], size, seed, 0.0

    def consume(item, kind, label, bucket, extra):
        events.append(("consume", kind, label, bucket))
        row.setdefault(bucket, []).append({"label": label, **extra})

    def facts(expected_live):
        events.append(("facts", expected_live))
    return row, refs, dispatch, consume, facts


def test_the_timing_hook_runs_the_plan_in_order_and_checks_after_each_call(monkeypatch):
    module = _bench()
    monkeypatch.setattr(module, "OWNED", [])
    events = []
    row, _refs, dispatch, consume, facts = _timed_fakes(module, events)
    module.timed_returns(module.timing_plan((8, 32), 2), "message", row, dispatch, consume, facts)
    assert events == [
        ("dispatch", 8, 600, 2), ("consume", "message", "timing_first_use", "timing_calls"), ("facts", 0),
        ("dispatch", 8, 601, 2), ("consume", "message", "timing_warm", "timing_calls"), ("facts", 0),
        ("dispatch", 8, 602, 2), ("consume", "message", "timing_warm", "timing_calls"), ("facts", 0),
        ("dispatch", 32, 603, 2), ("consume", "message", "timing_warm", "timing_calls"), ("facts", 0),
        ("dispatch", 32, 604, 2), ("consume", "message", "timing_warm", "timing_calls"), ("facts", 0)]
    assert [(c["phase"], c["repeat"], c["size_mib"]) for c in row["timing_calls"]] == [
        ("first_use", 0, 8), ("warm", 0, 8), ("warm", 1, 8), ("warm", 0, 32), ("warm", 1, 32)]


def test_a_passed_timed_message_return_frees_its_guard_future_and_tensor(monkeypatch):
    module = _bench()
    monkeypatch.setattr(module, "OWNED", [])
    row, refs, dispatch, consume, facts = _timed_fakes(module, [])
    module.timed_returns(module.timing_plan((8,), 3), "message", row, dispatch, consume, facts)
    gc.collect()
    assert len(refs) == 4
    assert all(ref() is None for call in refs for ref in call)


def test_native_timed_returns_and_a_failed_message_return_stay_rooted(monkeypatch):
    module = _bench()
    monkeypatch.setattr(module, "OWNED", [])
    row, refs, dispatch, consume, facts = _timed_fakes(module, [])
    module.timed_returns(module.timing_plan((8,), 1), "rdma", row, dispatch, consume, facts)

    def failing(item, kind, label, bucket, extra):
        raise module.QualificationRefused("CPU tensor exact integrity mismatch")
    with pytest.raises(module.QualificationRefused):
        module.timed_returns(module.timing_plan((8,), 1), "message", row, dispatch, failing, facts)
    gc.collect()
    assert len(refs) == 3
    assert all(ref() is not None for call in refs for ref in call)
    rooted = [root for roots in module.OWNED for root in roots]
    assert all(any(root is ref() for root in rooted) for call in refs for ref in call[:2])


@pytest.mark.parametrize("sizes", ["4", "8,8", "8,512", "eight", ""])
def test_timing_sizes_refuse_values_outside_the_measured_range(sizes):
    with pytest.raises(SystemExit):
        _bench().parse_args(["--expect-sha", SHA, "--expect-manifest", MANIFEST,
                             "--timing", "--timing-sizes-mib", sizes])


@pytest.mark.parametrize("leak", [
    {"note": "spark-one"}, {"note": "spark-two.local"}, {"peer": "192.0.2.2"},
    {"path": "/home/operator/evidence"}, {"retained_driver_pid": 4242},
    {"note": "this-box"}])
def test_timing_evidence_refuses_host_identity_before_writing(tmp_path, leak):
    module = _bench()
    guard = module.disclosure_guard(_config(), home="/srv/operator", hostname="this-box")
    with pytest.raises(module.QualificationRefused, match="disclose"):
        module.Evidence(tmp_path / "timing.json", {"result": "RUNNING", **leak}, guard=guard)
    assert not (tmp_path / "timing.json").exists()


def test_timing_evidence_allows_policy_snapshots_and_lookalike_words(tmp_path):
    module = _bench()
    guard = module.disclosure_guard(_config(), home="/srv/operator", hostname="this-box")
    evidence = module.Evidence(tmp_path / "timing.json", {
        "result": "RUNNING", "driver_policy": {"rdma": {"target": "nic:mlx5_0"}},
        "note": "spark-ones and respark-one stay distinct"}, guard=guard)
    try:
        evidence.data["later"] = "192.0.2.2"
        with pytest.raises(module.QualificationRefused):
            evidence.save()
        assert "192.0.2.2" not in (tmp_path / "timing.json").read_text()
    finally:
        os.close(evidence.fd)


def _hold(module, monkeypatch, evidence, **kwargs):
    class EndObservation(BaseException):
        pass

    class Event:
        def wait(self, _interval):
            raise EndObservation()
    monkeypatch.setattr(module.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(module.threading, "Event", Event)
    with pytest.raises(EndObservation):
        module.park(evidence, "settlement unknown", **kwargs)


def test_timing_hold_omits_the_pid_and_the_default_hold_keeps_it(monkeypatch, tmp_path):
    module = _bench()
    guard = module.disclosure_guard(_config(), home="/srv/operator", hostname="this-box")
    timing = module.Evidence(tmp_path / "timing.json", {"result": "RUNNING"}, guard=guard)
    default = module.Evidence(tmp_path / "default.json", {"result": "RUNNING"})
    try:
        _hold(module, monkeypatch, timing, record_pid=False)
        _hold(module, monkeypatch, default)
        held = json.loads((tmp_path / "timing.json").read_text())
        assert held["result"] == "HOLD" and held["retained_owner"] is True
        assert "retained_driver_pid" not in held
        assert json.loads((tmp_path / "default.json").read_text())["retained_driver_pid"] == os.getpid()
    finally:
        os.close(timing.fd)
        os.close(default.fd)


def _calls(owner, first_use, warm):
    calls = [{"owner_rank": owner, "phase": "first_use", "size_mib": 8, "repeat": 0,
              "pack_rpc_s": first_use, "read_drop_ack_s": 0.0}]
    for size, totals in warm.items():
        calls += [{"owner_rank": owner, "phase": "warm", "size_mib": size, "repeat": index,
                   "pack_rpc_s": total / 2, "read_drop_ack_s": total / 2}
                  for index, total in enumerate(totals)]
    return calls


def _report(module, mode, position, first_use, warm):
    cycles = [{"cycle": cycle, "timing_calls": _calls(0, first_use, warm) + _calls(1, first_use, warm)}
              for cycle in (0, 1)]
    return {"schema": 1, "result": "PASS", "return_mode": mode, "script_sha256": "c" * 64,
            "driver": {"bound_source_sha": SHA, "source_manifest_sha256": MANIFEST},
            "timing": {"schema": module.TIMING_SCHEMA, "position": position,
                       "order": module.timing_order(mode, position),
                       "sizes_mib": [8, 32], "warm_repeats": 3},
            "timing_summary": module.timing_summary(cycles)}


def _campaign(module, tmp_path):
    message = {8: [0.030, 0.031, 0.032], 32: [0.120, 0.110, 0.130]}
    native = {8: [0.031, 0.032, 0.033], 32: [0.080, 0.090, 0.085]}
    paths = []
    for name, mode, position, first, warm in (
            ("mm", "message", "first", 0.05, message), ("mn", "native", "second", 2.0, native),
            ("nn", "native", "first", 2.1, native), ("nm", "message", "second", 0.05, message)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(_report(module, mode, position, first, warm)))
        paths.append(str(path))
    return paths


def test_the_summary_pairs_orders_and_reports_warmed_medians(tmp_path):
    module = _bench()
    result = module.compare_timing(module.load_timing_reports(_campaign(module, tmp_path), SHA, MANIFEST))
    assert len(result["strata"]) == 8
    assert {row["order"] for row in result["strata"]} == {"message-first", "native-first"}
    row = result["strata"][0]["warm"]["32"]
    assert row["message_median_s"] == pytest.approx(0.120)
    assert row["native_median_s"] == pytest.approx(0.085)
    assert row["native_saving_ms"] == pytest.approx(35.0)
    assert row["message_to_native_ratio"] == pytest.approx(0.120 / 0.085)
    assert result["warm_ranges"]["8"]["native_saving_ms"][1] < 0  # native slower at 8 MiB
    assert result["native_first_use_s"] == pytest.approx([2.0, 2.1])


def test_the_summary_refuses_unbound_or_unpaired_reports(tmp_path):
    module = _bench()
    paths = _campaign(module, tmp_path)
    with pytest.raises(module.QualificationRefused, match="source differs"):
        module.load_timing_reports(paths, "d" * 40, MANIFEST)
    with pytest.raises(module.QualificationRefused, match="message and a native"):
        module.compare_timing(module.load_timing_reports(paths[:1] + paths[2:3], SHA, MANIFEST))
    report = json.loads(Path(paths[0]).read_text())
    report["result"] = "HOLD"
    Path(paths[0]).write_text(json.dumps(report))
    with pytest.raises(module.QualificationRefused, match="PASS timing reports"):
        module.load_timing_reports(paths, SHA, MANIFEST)


def test_the_summary_refuses_one_complete_pair_and_names_the_missing_order(tmp_path, capsys):
    module = _bench()
    paths = _campaign(module, tmp_path)  # mm, mn: message-first; nn, nm: native-first
    for pair, missing in ((paths[:2], "native-first"), (paths[2:], "message-first")):
        with pytest.raises(module.QualificationRefused, match=f"both orders; missing {missing}"):
            module.compare_timing(module.load_timing_reports(pair, SHA, MANIFEST))
        assert module.main(["--expect-sha", SHA, "--expect-manifest", MANIFEST,
                            "--timing-summary", *pair]) == 2
        captured = capsys.readouterr()
        assert captured.out == "" and f"missing {missing}" in captured.err
    assert len(module.compare_timing(module.load_timing_reports(paths, SHA, MANIFEST))["strata"]) == 8


def test_the_summary_command_imports_no_native_runtime(monkeypatch, tmp_path, capsys):
    import builtins

    module = _bench()
    paths = _campaign(module, tmp_path)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("monarch"):
            raise AssertionError("the timing summary imported a native runtime")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert module.main(["--expect-sha", SHA, "--expect-manifest", MANIFEST,
                        "--timing-summary", *paths]) == 0
    assert len(json.loads(capsys.readouterr().out)["strata"]) == 8


def test_the_first_use_phase_split_reaches_the_summary(tmp_path):
    module = _bench()
    split = {"worker": {"schema": 1, "total_s": 1.95, "resolution_s": 0.001, "complete": True,
                        "phases_s": {"rdma_preflight": 1.8, "rdma_register": 0.1,
                                     "unattributed": 0.05}},
             "driver": {"schema": 1, "total_s": 0.04, "resolution_s": 0.001, "complete": True,
                        "phases_s": {"rdma_read": 0.02, "rdma_drop": 0.005, "ack": 0.01,
                                     "unattributed": 0.005}}}
    calls = _calls(0, 2.0, {8: [0.03]}) + _calls(1, 2.0, {8: [0.03]})
    for call in calls:
        if call["phase"] == "first_use":
            call["return_phases"] = split
    strata = module.timing_summary([{"cycle": 0, "timing_calls": calls}])
    assert [entry["first_use_phases"] for entry in strata] == [split, split]
    # Reports from before the split carry no phases and still pair.
    paths = _campaign(module, tmp_path)
    native = json.loads(Path(paths[1]).read_text())
    for entry in native["timing_summary"]:
        entry["first_use_phases"] = split
    Path(paths[1]).write_text(json.dumps(native))
    result = module.compare_timing(module.load_timing_reports(paths, SHA, MANIFEST))
    with_split = [row for row in result["strata"] if "native_first_use_phases" in row]
    assert len(with_split) == 4 and with_split[0]["native_first_use_phases"] == split
