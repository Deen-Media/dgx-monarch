"""RDMA rail readings: byte counters and link state from a faked sysfs tree.

Pure stdlib: no comfy, no torch, no real RDMA device.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from dgx_monarch import rails, telemetry


def _port(root: Path, dev: str, *, state: str | None = "4: ACTIVE",
          phys: str | None = "5: LinkUp", rx: str | None = None,
          tx: str | None = None, number: str = "1") -> None:
    port = root / dev / "ports" / number
    port.mkdir(parents=True)
    if state is not None:
        (port / "state").write_text(f"{state}\n")
    if phys is not None:
        (port / "phys_state").write_text(f"{phys}\n")
    if rx is not None or tx is not None:
        counters = port / "counters"
        counters.mkdir()
        if rx is not None:
            (counters / "port_rcv_data").write_text(f"{rx}\n")
        if tx is not None:
            (counters / "port_xmit_data").write_text(f"{tx}\n")


@pytest.fixture
def sysfs(tmp_path, monkeypatch):
    root = tmp_path / "infiniband"
    root.mkdir()
    monkeypatch.setattr(rails, "_SYSFS_IB", str(root))
    return root


def test_telemetry_keeps_the_names_it_moved_out():
    # Callers and monkeypatch targets that name telemetry.rail_counters keep
    # working after the move to the leaf module.
    assert telemetry.rail_counters is rails.rail_counters
    assert telemetry.rail_state is rails.rail_state


def test_counters_read_four_byte_lanes_and_skip_an_unused_rail(sysfs):
    _port(sysfs, "rail0", rx="10", tx="20")
    _port(sysfs, "rail1", rx="0", tx="0")            # cabled elsewhere or unused
    _port(sysfs, "rail2", rx="7", tx="garbage")      # unreadable reading
    _port(sysfs, "rail3", rx="3")                    # one counter file missing
    _port(sysfs, "rail4", rx="0", tx="5")            # one direction counts

    assert rails.rail_counters() == {
        "rail0": {"rx_bytes": 40, "tx_bytes": 80},
        "rail4": {"rx_bytes": 0, "tx_bytes": 20},
    }


def test_state_mirrors_the_doctor_row_and_keeps_raw_text_otherwise(sysfs):
    _port(sysfs, "up", rx="10", tx="10")
    # A rail that carried traffic and then failed: down, with counters.
    _port(sysfs, "down", state="1: DOWN", phys="3: Disabled", rx="4", tx="4")
    _port(sysfs, "trained_down", state="1: DOWN", phys="5: LinkUp")
    _port(sysfs, "polling", state="1: DOWN", phys="2: Polling")   # uncabled: left out
    _port(sysfs, "no_phys", phys=None, rx="1", tx="1")  # unreadable state: counters only

    assert rails.rail_state() == {
        "up": {"state": "ACTIVE", "phys_state": "5: LinkUp"},
        "down": {"state": "1: DOWN", "phys_state": "3: Disabled"},
        "trained_down": {"state": "1: DOWN", "phys_state": "5: LinkUp"},
    }
    assert set(rails.rail_counters()) == {"up", "down", "no_phys"}


def test_state_covers_cabled_ports_the_counters_skip(sysfs):
    # A cabled port with zero counters has no "rails" row, but its state must
    # still reach the payload. "linked_down" is cabled with its link up and
    # its port down, which the panel lists, and "idle" is this rig's unused
    # ACTIVE function. "never_trained" is a rail that failed link training at
    # boot: DOWN with phys Disabled and zero counters, read the same as an
    # uncabled port, so it is left out of both maps and `dgxm doctor` reports it.
    _port(sysfs, "carrying", rx="8", tx="8")
    _port(sysfs, "idle", rx="0", tx="0")
    _port(sysfs, "linked_down", state="1: DOWN", phys="5: LinkUp", rx="0", tx="0")
    _port(sysfs, "never_trained", state="1: DOWN", phys="3: Disabled", rx="0", tx="0")

    assert set(rails.rail_counters()) == {"carrying"}
    assert set(rails.rail_state()) == {"carrying", "idle", "linked_down"}


def test_a_device_with_one_cabled_port_is_keyed_by_its_name(sysfs):
    # Port 2 is uncabled (no link, zero counters), so the device has one rail
    # and keeps the device key, as every ConnectX-7 function on this rig does.
    _port(sysfs, "mlx5_0", rx="10", tx="20")
    _port(sysfs, "mlx5_0", state="1: DOWN", phys="2: Polling", rx="0", tx="0", number="2")

    readings = rails.rail_readings()

    assert readings["rails"] == {"mlx5_0": {"rx_bytes": 40, "tx_bytes": 80}}
    assert readings["rail_state"] == {"mlx5_0": {"state": "ACTIVE", "phys_state": "5: LinkUp"}}


def test_two_cabled_ports_on_one_device_are_keyed_by_device_and_port(sysfs):
    # Before, the second port overwrote the first in both maps: a DOWN port
    # could mark an ACTIVE rail down, and the counters could belong to another
    # port than the state beside them. Each port now has its own key, the
    # same in both maps.
    _port(sysfs, "mlx5_0", rx="10", tx="20")
    _port(sysfs, "mlx5_0", state="1: DOWN", phys="3: Disabled", rx="5", tx="6", number="2")
    _port(sysfs, "mlx5_1", rx="1", tx="1")
    _port(sysfs, "mlx5_1", state="1: DOWN", phys="5: LinkUp", rx="0", tx="0", number="2")

    readings = rails.rail_readings()

    assert readings["rails"] == {
        "mlx5_0/1": {"rx_bytes": 40, "tx_bytes": 80},
        "mlx5_0/2": {"rx_bytes": 20, "tx_bytes": 24},
        "mlx5_1/1": {"rx_bytes": 4, "tx_bytes": 4},
    }
    assert readings["rail_state"] == {
        "mlx5_0/1": {"state": "ACTIVE", "phys_state": "5: LinkUp"},
        "mlx5_0/2": {"state": "1: DOWN", "phys_state": "3: Disabled"},
        "mlx5_1/1": {"state": "ACTIVE", "phys_state": "5: LinkUp"},
        "mlx5_1/2": {"state": "1: DOWN", "phys_state": "5: LinkUp"},
    }
    assert set(readings["rails"]) <= set(readings["rail_state"])
    assert rails.rail_counters() == readings["rails"]
    assert rails.rail_state() == readings["rail_state"]


def test_the_stamp_is_taken_when_the_counters_are_read(sysfs, monkeypatch):
    # A reader times a rate by the interval between two monotonic stamps, so
    # both stamps must follow the counter reads, not the slower readings
    # around them.
    _port(sysfs, "rail0", rx="1", tx="1")
    _port(sysfs, "rail1", rx="2", tx="2")
    seen = []
    counters, link = rails._counters, rails._link

    def mark(kind):
        seen.append((kind, time.time(), time.monotonic()))

    monkeypatch.setattr(rails, "_counters", lambda port: (mark("c"), counters(port))[1])
    monkeypatch.setattr(rails, "_link", lambda port: (mark("l"), link(port))[1])

    readings = rails.rail_readings()

    assert [kind for kind, _, _ in seen] == ["c", "c", "l", "l"]
    for key, clock in (("rails_t", 1), ("rails_m", 2)):
        stamp = readings[key]
        assert isinstance(stamp, float)
        assert (max(row[clock] for row in seen if row[0] == "c") <= stamp
                <= min(row[clock] for row in seen if row[0] == "l"))


def test_no_rdma_devices_reads_empty(sysfs):
    assert rails.rail_counters() == {}
    assert rails.rail_state() == {}


def test_host_stats_publishes_link_state_beside_the_counters(sysfs, monkeypatch):
    # Stub the readings that shell out to nvidia-smi; this test reads the rails.
    monkeypatch.setattr(telemetry, "_gpu_stats", dict)
    monkeypatch.setattr(telemetry, "_gpu_proc_gib", lambda: 0.0)
    monkeypatch.setattr(telemetry, "_retained_pool_gib", lambda: 0.0)
    _port(sysfs, "rail0", rx="1", tx="2")
    _port(sysfs, "rail1", state="1: DOWN", phys="5: LinkUp", rx="0", tx="0")
    _port(sysfs, "rail2", state="1: DOWN", phys="3: Disabled", rx="0", tx="0")  # uncabled

    before, before_mono = time.time(), time.monotonic()
    stats = telemetry.host_stats()

    assert stats["rails"] == {"rail0": {"rx_bytes": 4, "tx_bytes": 8}}
    assert stats["rail_state"] == {
        "rail0": {"state": "ACTIVE", "phys_state": "5: LinkUp"},
        "rail1": {"state": "1: DOWN", "phys_state": "5: LinkUp"},
    }
    assert before <= stats["rails_t"] <= time.time()
    assert before_mono <= stats["rails_m"] <= time.monotonic()


def test_the_rails_module_is_a_stdlib_leaf():
    import ast

    tree = ast.parse(Path(rails.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "rails.py must not import from the package"
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"__future__", "glob", "os", "time"}


def test_dgxm_top_reads_the_counters_and_ignores_the_link_state(monkeypatch):
    from dgx_monarch.tui.data import Poller

    poller = Poller(loops=())
    monkeypatch.setattr(poller, "_driver_telemetry", lambda: {"driver_host": {
        "host": "driver",
        "rails": {"rail0": {"rx_bytes": 3, "tx_bytes": 4}},
        "rail_state": {"rail0": {"state": "ACTIVE", "phys_state": "5: LinkUp"},
                       "rail1": {"state": "1: DOWN", "phys_state": "3: Disabled"}},
    }})
    assert poller.tick()["hosts"]["driver"]["rails"] == {
        "rail0": {"rx_gbs": 0.0, "tx_gbs": 0.0}}


def test_dgxm_top_times_a_rate_by_the_reading_stamps():
    # The poller's clock says when it saw a reading; rails_m says when the box
    # read its counters. With both stamps the rate uses the box's interval.
    from dgx_monarch.tui.data import Poller

    poller = Poller(loops=())
    rates = []
    for at, read_mono, rx in [
        (100.0, 50.0, 0),        # first sample: no interval yet
        (104.0, 52.5, 5e9),      # waited in the cache: 2.5 s by the box, 4 s by the poller
        (106.0, None, 7e9),      # an older driver: the poller's clock
        (107.0, 60.0, 8e9),      # the previous sample had no stamp: the poller's clock
        (108.0, 60.0, 8e9),      # the same reading again: the stamp did not move
        (110.0, 62.0, 12e9),     # timed from the last stamp that moved, 2 s
    ]:
        sample = poller._rail_rates("box", {"mlx5_0/1": {"rx_bytes": rx, "tx_bytes": 0}},
                                    at, read_mono=read_mono)
        rates.append(sample["mlx5_0/1"]["rx_gbs"])

    assert rates == [0.0, 2.0, 1.0, 1.0, 0.0, 2.0]


def test_dgxm_top_reads_the_stamp_from_the_payload(monkeypatch):
    from dgx_monarch.tui import data
    from dgx_monarch.tui.data import Poller

    poller = Poller(loops=())
    payloads = iter([
        (10.0, {"host": "box", "rails_t": 1000.0, "rails_m": 500.0,
                "rails": {"r0": {"rx_bytes": 0, "tx_bytes": 0}}}),
        (13.0, {"host": "box", "rails_t": 1001.0, "rails_m": 501.0,
                "rails": {"r0": {"rx_bytes": 3e9, "tx_bytes": 0}}}),
    ])
    clock = {"now": 0.0}

    def telemetry():
        clock["now"], stats = next(payloads)
        return {"workers": [{"host": stats}]}

    monkeypatch.setattr(poller, "_driver_telemetry", telemetry)
    monkeypatch.setattr(data, "time", type("Clock", (), {
        "time": staticmethod(lambda: clock["now"]),
        "monotonic": staticmethod(lambda: clock["now"])}))
    poller.tick()

    # 3e9 bytes over the box's 1 s, not over the poller's 3 s.
    assert poller.tick()["hosts"]["box"]["rails"] == {"r0": {"rx_gbs": 3.0, "tx_gbs": 0.0}}


def test_a_device_and_port_key_reaches_dgxm_top_and_the_metrics_route():
    from dgx_monarch.nodes.metrics_route import _metrics_text
    from dgx_monarch.tui.data import Poller

    counters = {"mlx5_0/1": {"rx_bytes": 8, "tx_bytes": 9}, "mlx5_0/2": {"rx_bytes": 1, "tx_bytes": 2}}
    assert set(Poller(loops=())._rail_rates("box", counters, 1.0)) == {"mlx5_0/1", "mlx5_0/2"}

    text = _metrics_text({"workers": [{"host": {"host": "box", "rails": counters}}]})
    assert 'dgxm_rail_rx_bytes_total{host="box",rail="mlx5_0/1"} 8' in text
    assert 'dgxm_rail_tx_bytes_total{host="box",rail="mlx5_0/2"} 2' in text
