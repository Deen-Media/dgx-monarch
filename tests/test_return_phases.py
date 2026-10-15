"""The phase split of one native return: record shape, sums and the log line."""
from __future__ import annotations

import subprocess
import threading
import time
from types import SimpleNamespace

import pytest
import torch

import dgx_monarch.transfer as transfer
from dgx_monarch import return_phases, runtime_diagnostics, transfer_utils
from dgx_monarch.actor.rdma_handoff import RDMAHandoffMixin
from dgx_monarch.transfer import LatentReturn, read_latent_result
from transfer_helpers import _isolate_process_lifetime_rdma_poison  # noqa: F401


@pytest.fixture(autouse=True)
def fresh_ring(monkeypatch):
    monkeypatch.setattr(runtime_diagnostics, "_STORE", runtime_diagnostics.PhaseMeasurements())


def _assert_sums_to_total(record):
    """The recorded phases add up to the total within the recorded rounding."""
    phases = record["phases_s"]
    assert record["complete"] is True
    assert record["resolution_s"] == 0.001
    tolerance = record["resolution_s"] / 2 * (len(phases) + 1) + 1e-9
    assert abs(sum(phases.values()) - record["total_s"]) <= tolerance
    assert all(value >= 0 for value in phases.values())


def _sleep_phase(name, seconds):
    with runtime_diagnostics.phase(name, return_mode="rdma"):
        time.sleep(seconds)


def test_first_use_shape_names_each_phase_and_the_unattributed_rest():
    started = return_phases.mark()
    _sleep_phase("rdma_preflight", 0.03)
    _sleep_phase("rdma_register", 0.01)
    with runtime_diagnostics.phase("collect", return_mode="message"):  # not a return phase
        time.sleep(0.005)
    record = return_phases.split(started, return_phases.WORKER_PHASES)
    assert record["schema"] == 1
    assert list(record["phases_s"]) == ["rdma_preflight", "rdma_register", "unattributed"]
    assert record["phases_s"]["rdma_preflight"] >= 0.03
    assert record["phases_s"]["unattributed"] >= 0.005
    _assert_sums_to_total(record)


def test_warm_shape_has_no_preflight_and_message_returns_have_no_record():
    started = return_phases.mark()
    _sleep_phase("rdma_register", 0.005)
    record = return_phases.split(started, return_phases.WORKER_PHASES)
    assert list(record["phases_s"]) == ["rdma_register", "unattributed"]
    _assert_sums_to_total(record)
    assert return_phases.split(return_phases.mark(), return_phases.WORKER_PHASES) is None
    forced = return_phases.split(return_phases.mark(), return_phases.DRIVER_PHASES, always=True)
    assert list(forced["phases_s"]) == ["unattributed"]


def test_an_evicted_window_or_phases_longer_than_the_total_read_partial():
    started = return_phases.mark()
    for _ in range(runtime_diagnostics.MAX_RECENT + 2):
        runtime_diagnostics.record_phase("rdma_read", 0.0, return_mode="rdma")
    assert return_phases.split(started, return_phases.DRIVER_PHASES)["complete"] is False
    started = return_phases.mark()
    runtime_diagnostics.record_phase("ack", 30.0, return_mode="rdma")
    record = return_phases.split(started, return_phases.DRIVER_PHASES)
    assert record["complete"] is False and record["phases_s"]["unattributed"] == 0.0


class _Worker(RDMAHandoffMixin):
    def __init__(self, threshold=1):
        self.rank = 0
        self._setup_generation = 3
        self._latent_return = LatentReturn("rdma", min_bytes=threshold)
        self._init_rdma_handoffs()

    def pack(self):
        return self._pack_latent_result({}, {"samples": torch.ones(64)}, {})


def _child(monkeypatch, returncode=0, seconds=0.02):
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        time.sleep(seconds)
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(transfer_utils.subprocess, "run", run)
    return calls


def test_the_worker_result_row_carries_the_pack_split(monkeypatch):
    monkeypatch.setattr(transfer, "_qp_split", lambda: 1)
    _child(monkeypatch)
    monkeypatch.setattr(transfer, "_rdma_usable", transfer_utils.native_rdma_preflight)
    worker = _Worker()
    first = worker.pack()
    assert first["latent"]["kind"] == "rdma"
    record = first[return_phases.RESULT_KEY]
    assert set(record["phases_s"]) == {"rdma_preflight", "rdma_register", "unattributed"}
    assert record["phases_s"]["rdma_preflight"] >= 0.02
    _assert_sums_to_total(record)
    # Later packs in the process skip the preflight (transfer caches its result).
    monkeypatch.setattr(transfer, "_rdma_usable", lambda: True)
    worker._latent_handoffs = type(worker._latent_handoffs)()
    warm = worker.pack()[return_phases.RESULT_KEY]
    assert set(warm["phases_s"]) == {"rdma_register", "unattributed"}
    _assert_sums_to_total(warm)


def test_a_message_pack_below_the_threshold_adds_nothing_to_the_row(monkeypatch):
    monkeypatch.setattr(transfer, "_rdma_usable", lambda: True)
    result = _Worker(threshold=1 << 30).pack()
    assert result["latent"]["kind"] == "message"
    assert return_phases.RESULT_KEY not in result


def test_a_failed_preflight_is_an_error_phase_and_still_returns_false(monkeypatch):
    _child(monkeypatch, returncode=1, seconds=0.0)
    assert transfer_utils.native_rdma_preflight() is False
    [row] = runtime_diagnostics.snapshot()["phases"]
    assert (row["phase"], row["outcome"], row["return_mode"]) == ("rdma_preflight", "error", "rdma")

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("python", 15)

    monkeypatch.setattr(transfer_utils.subprocess, "run", timeout)
    assert transfer_utils.native_rdma_preflight() is False
    outcomes = {row["outcome"] for row in runtime_diagnostics.snapshot()["phases"]
                if row["phase"] == "rdma_preflight"}
    assert outcomes == {"error", "timeout"}  # a hung child is a timeout, not a crash
    _child(monkeypatch, returncode=0, seconds=0.0)
    assert transfer_utils.native_rdma_preflight() is True


def test_the_driver_read_split_covers_read_and_drop(monkeypatch):
    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "_qp_split", lambda: 2)
    descriptor = LatentReturn("rdma", min_bytes=1).pack("samples", torch.randn(4, 64))
    assert descriptor["kind"] == "rdma" and len(descriptor["parts"]) == 2
    started = return_phases.mark()
    read_latent_result(descriptor)
    record = return_phases.split(started, return_phases.DRIVER_PHASES, always=True)
    assert set(record["phases_s"]) == {"rdma_read", "rdma_drop", "unattributed"}
    _assert_sums_to_total(record)


def test_overlapping_part_reads_count_once_and_a_later_drop_adds_its_own_time():
    started = return_phases.mark()
    # Parts of one read overlap. Staggered ends keep each record clear of the
    # ring's non-blocking lock, so none lands in unattributed by contention.
    parts = [threading.Thread(target=_sleep_phase, args=("rdma_read", seconds))
             for seconds in (0.10, 0.11, 0.12, 0.13)]
    for part in parts:
        part.start()
    for part in parts:
        part.join()
    _sleep_phase("rdma_drop", 0.02)
    record = return_phases.split(started, return_phases.DRIVER_PHASES)
    assert set(record["phases_s"]) == {"rdma_read", "rdma_drop", "unattributed"}
    # The longest part, not the 0.46 s sum of all four.
    assert 0.13 <= record["phases_s"]["rdma_read"] < 0.4
    assert record["phases_s"]["rdma_drop"] >= 0.02
    _assert_sums_to_total(record)


def _total_s(described):
    return float(described.split("s (", 1)[0])


def test_the_render_logs_one_line_per_native_return():
    lines = []
    log = SimpleNamespace(info=lambda *args: lines.append(args))
    descriptor = {"kind": "rdma", "parts": []}
    started = return_phases.mark()
    _sleep_phase("rdma_read", 0.002)
    _sleep_phase("ack", 0.001)
    worker = {"schema": 1, "total_s": 2.051, "resolution_s": 0.001, "complete": True,
              "phases_s": {"rdma_preflight": 1.912, "rdma_register": 0.081,
                           "unattributed": 0.058}}
    rows = [{"rank": 1, "host": "h", "latent": None},
            {"rank": 0, "host": "h", "latent": descriptor, return_phases.RESULT_KEY: worker}]
    return_phases.log_read(log, descriptor, rows, started)
    return_phases.log_read(log, {"kind": "message"}, rows, started)
    [(fmt, rank, host, pack, read)] = lines
    assert fmt == "rank %s/%s native return: worker pack %s; driver read %s"
    assert (rank, host) == (0, "h")
    assert pack == ("2.051s (rdma_preflight 1.912s, rdma_register 0.081s, "
                    "unattributed 0.058s)")
    assert _total_s(read) >= 0.003 and "rdma_read" in read and "ack" in read


class _Log:
    """Collects INFO lines; other levels are ignored."""

    def __init__(self):
        self.lines = []

    def info(self, fmt, *args):
        self.lines.append((fmt, args))

    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None

    def native(self):
        return [args for fmt, args in self.lines if "native return:" in fmt]


def test_finish_render_logs_the_native_split_and_a_message_return_logs_none(monkeypatch):
    from dgx_monarch.nodes import render_result

    log = _Log()
    monkeypatch.setattr(render_result, "log", log)
    monkeypatch.setattr(render_result, "verify_cross_rank_signatures", lambda *_args: None)
    rendered = torch.ones(1, 4, 2, 2)

    def read(descriptor):
        if descriptor.get("kind") == "rdma":
            _sleep_phase("rdma_read", 0.002)
            _sleep_phase("ack", 0.001)
        return rendered

    monkeypatch.setattr(render_result, "read_latent_result", read)
    worker = {"schema": 1, "total_s": 0.09, "resolution_s": 0.001, "complete": True,
              "phases_s": {"rdma_register": 0.081, "unattributed": 0.009}}
    native = {"rank": 0, "dp_rank": 0, "host": "h", "latent": {"kind": "rdma", "parts": []},
              return_phases.RESULT_KEY: worker}
    latent = {"samples": torch.zeros(1, 4, 1, 1)}
    out = render_result._finish_render([native], SimpleNamespace(dp=1), latent)
    assert out["samples"] is rendered
    [(rank, host, pack, driver)] = log.native()
    assert (rank, host) == (0, "h")
    assert pack == return_phases.describe(worker)  # the Worker row's split
    assert _total_s(driver) >= 0.003 and "rdma_read" in driver and "ack" in driver

    log.lines.clear()
    message = {"rank": 0, "dp_rank": 0, "host": "h", "latent": {"kind": "message"}}
    assert render_result._finish_render([message], SimpleNamespace(dp=1), latent)["samples"] is rendered
    assert log.native() == []


@pytest.mark.parametrize("record", [
    None, {"schema": 2}, {"schema": 1, "total_s": float("nan"), "phases_s": {}},
    {"schema": 1, "total_s": 1.0, "phases_s": {"bad name": 1.0}},
    {"schema": 1, "total_s": 1.0, "phases_s": {"rdma_read": -1}},
    {"schema": 1, "total_s": 1.0, "phases_s": {f"p{i}": 0.1 for i in range(9)}},
])
def test_an_older_or_malformed_worker_record_reads_not_recorded(record):
    assert return_phases.describe(record) == "not recorded"


def test_a_partial_record_says_so():
    record = {"schema": 1, "total_s": 1.0, "complete": False, "phases_s": {"ack": 1.0}}
    assert return_phases.describe(record) == "1.000s (ack 1.000s, partial)"


def test_diagnostic_failures_never_reach_the_pack_or_read(monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("ring unreadable")

    monkeypatch.setattr(runtime_diagnostics, "events_since", broken)
    result = {}
    return_phases.attach(result, return_phases.mark(), {"kind": "rdma"})
    assert result == {}
    return_phases.log_read(SimpleNamespace(info=broken), {"kind": "rdma"}, [], return_phases.mark())
    monkeypatch.setattr(runtime_diagnostics, "sequence", broken)
    assert return_phases.mark()[0] is None


def test_marking_takes_no_store_lock_and_an_idle_window_reads_nothing(monkeypatch):
    store = runtime_diagnostics._STORE
    with store._lock:  # a concurrent record or snapshot holds it
        started = return_phases.mark()
    assert started[0] == store._sequence

    def forbidden(*_args, **_kwargs):
        raise AssertionError("copied the ring for an idle window")

    monkeypatch.setattr(runtime_diagnostics, "events_since", forbidden)
    assert return_phases.split(started, return_phases.WORKER_PHASES) is None
