"""Poller.tick() on malformed telemetry: a field of the wrong type is dropped and
the rest of the tick kept, and anything that still raises becomes an UNKNOWN
tick (the `except Exception` in Poller.tick), so no AttributeError or TypeError
reaches Textual's polling timer. The HTTP fetch is stubbed at
Poller._driver_telemetry."""
from __future__ import annotations

import socket
import types

from dgx_monarch.tui.braille import braille_rows
from dgx_monarch.tui.data import Poller


def _poller_with_telemetry(monkeypatch, telemetry: dict) -> Poller:
    poller = Poller(loops=())
    monkeypatch.setattr(poller, "_driver_telemetry", lambda: telemetry)
    return poller


def test_tick_survives_non_dict_driver_host(monkeypatch):
    poller = _poller_with_telemetry(monkeypatch, {"driver_host": "not-a-dict"})
    tick = poller.tick()
    assert isinstance(tick, dict)
    assert tick["hosts"] == {}


def test_tick_survives_non_list_events(monkeypatch):
    poller = _poller_with_telemetry(monkeypatch, {"events": "not-a-list"})
    tick = poller.tick()
    assert tick["events"] == []


def test_tick_survives_non_list_workers(monkeypatch):
    poller = _poller_with_telemetry(monkeypatch, {"workers": True})
    tick = poller.tick()
    assert isinstance(tick, dict)
    assert tick["hosts"] == {}
    assert tick["workers"] == {}


def test_tick_passes_through_canonical_readiness_without_another_observation(monkeypatch):
    report = {
        "schema_version": 1,
        "overall": "ready",
        "lifecycle": {},
        "actions": [],
    }
    poller = _poller_with_telemetry(monkeypatch, {"readiness": report})
    assert poller.tick()["readiness"] is report


def test_tick_survives_non_list_events_inside_a_worker(monkeypatch):
    poller = _poller_with_telemetry(monkeypatch, {
        "workers": [{"host": {"host": "spark-b"}, "events": "not-a-list"}],
    })
    tick = poller.tick()
    assert tick["events"] == []
    assert "spark-b" in tick["hosts"]


def test_tick_rejects_wrong_type_render(monkeypatch):
    poller = _poller_with_telemetry(monkeypatch, {"render": [1, 2, 3]})
    tick = poller.tick()
    assert tick["render"] == {"active": False}


def test_tick_survives_every_malformed_field_at_once(monkeypatch):
    poller = _poller_with_telemetry(monkeypatch, {
        "driver_host": "not-a-dict",
        "events": "not-a-list",
        "render": [1, 2, 3],
        "ledger": 5,
        "workers": "also-not-a-list",
    })
    tick = poller.tick()
    assert isinstance(tick, dict)
    assert tick["hosts"] == {}
    assert tick["events"] == []
    assert tick["workers"] == {}


def test_tick_filters_malformed_event_rows_and_rail_counters(monkeypatch):
    poller = _poller_with_telemetry(monkeypatch, {
        "events": [1, None, {"t": 2, "kind": "ok"}],
        "driver_host": {
            "host": "driver",
            "rails": {
                "bad-row": "oops",
                "bad-number": {"rx_bytes": "one", "tx_bytes": 2},
                "good": {"rx_bytes": 3, "tx_bytes": 4},
            },
        },
    })
    tick = poller.tick()
    assert tick["events"] == [{"t": 2, "kind": "ok"}]
    assert tick["hosts"]["driver"]["rails"] == {
        "good": {"rx_gbs": 0.0, "tx_gbs": 0.0},
    }


def test_non_numeric_event_times_do_not_erase_known_blocked_readiness(monkeypatch):
    report = {"schema_version": 1, "overall": "blocked", "lifecycle": {}, "actions": []}
    poller = _poller_with_telemetry(monkeypatch, {
        "readiness": report,
        "events": [
            {"t": "not-a-number", "kind": "notice", "note": "first"},
            {"t": [1], "kind": "notice", "note": "second"},
            {"t": float("nan"), "kind": "notice", "note": "third"},
            {"t": 3.0, "kind": "notice", "note": "last"},
        ],
    })

    tick = poller.tick()

    assert tick["driver_up"] is True
    assert tick["readiness"] is report
    assert "telemetry_error" not in tick
    assert [event["note"] for event in tick["events"]] == [
        "first", "second", "third", "last"]


def test_nonfinite_rail_rates_are_zeroed_before_labels_and_braille(monkeypatch):
    poller = Poller(loops=())
    poller._prev_rails[("driver", "rail0")] = {
        "t": 0.0, "rx": -float.fromhex("0x1.fffffffffffffp+1023"), "tx": 0.0,
    }
    rates = poller._rail_rates(
        "driver",
        {"rail0": {"rx_bytes": float.fromhex("0x1.fffffffffffffp+1023"), "tx_bytes": 1.0}},
        1.0,
    )

    assert rates == {"rail0": {"rx_gbs": 0.0, "tx_gbs": 1e-09}}
    assert braille_rows([float("nan"), float("inf"), -float("inf"), 1.0], 2, 1)


def test_retained_pool_scan_keeps_last_complete_value_when_budget_expires(monkeypatch):
    from dgx_monarch import telemetry

    monkeypatch.setattr(telemetry, "_POOL_CACHE", {"t": 0.0, "gib": 12.5})
    monkeypatch.setattr(telemetry, "_POOL_SCAN_BUDGET_S", 0.0)
    monkeypatch.setattr(telemetry.time, "time", lambda: 100.0)
    monkeypatch.setattr(telemetry.time, "monotonic", lambda: 1.0)
    monkeypatch.setattr(
        telemetry.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(stdout="123\n"),
    )
    monkeypatch.setattr(
        "builtins.open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("expired scan must not open smaps")),
    )

    assert telemetry._retained_pool_gib() == 12.5
    assert telemetry._POOL_CACHE == {"t": 100.0, "gib": 12.5}


def test_tick_publishes_unknown_if_normalization_raises(monkeypatch):
    poller = _poller_with_telemetry(monkeypatch, {"driver_host": {"host": "driver"}})
    monkeypatch.setattr(
        poller, "_rail_rates",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("private detail")),
    )
    tick = poller.tick()
    assert tick["driver_up"] is False
    assert tick["readiness"] is None
    assert tick["telemetry_error"] == "RuntimeError"
    assert "private detail" not in str(tick)


def test_tick_prefers_driver_reported_loop_state(monkeypatch):
    poller = Poller(loops=("tcp://10.0.0.2:26600", "tcp://[fd00::2]:26600"))
    monkeypatch.setattr(poller, "_driver_telemetry", lambda: {
        "loops": [
            {"address": "10.0.0.2:26600", "up": True},
            {"addr": "tcp://[fd00::2]:26600", "healthy": False},
        ],
    })

    assert poller.tick()["loops"] == [
        {"addr": "tcp://10.0.0.2:26600", "up": True},
        {"addr": "tcp://[fd00::2]:26600", "up": False},
    ]


def test_tick_reports_unknown_without_driver_attestation(monkeypatch):
    from dgx_monarch import telemetry

    poller = Poller(loops=("tcp://10.0.0.2:26600",))
    monkeypatch.setattr(poller, "_driver_telemetry", lambda: None)
    monkeypatch.setattr(telemetry, "host_stats", lambda: {"host": "driver", "rails": {}})
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("the TUI must not connect to Monarch worker sockets")
        ),
    )

    tick = poller.tick()
    assert tick["driver_up"] is False
    assert tick["loops"] == [{"addr": "tcp://10.0.0.2:26600", "up": None}]


def test_tick_reports_observation_errors_as_unknown(monkeypatch):
    poller = Poller(loops=("tcp://10.0.0.2:26600",))
    monkeypatch.setattr(poller, "_driver_telemetry", lambda: {
        "loops": [{"address": "10.0.0.2:26600", "status_error": "TimeoutError"}],
    })

    assert poller.tick()["loops"] == [
        {"addr": "tcp://10.0.0.2:26600", "up": None},
    ]
