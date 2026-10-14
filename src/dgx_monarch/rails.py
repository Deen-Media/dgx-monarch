"""RDMA rail readings from sysfs for the telemetry payload (stdlib leaf).

rail_readings() walks ``/sys/class/infiniband/<dev>/ports/<n>`` once and
returns the four keys host_stats() publishes:

* ``rails``, the cumulative RX and TX bytes per rail. A rail whose two byte
  counters are both zero has no row.
* ``rails_t``, the wall-clock time (time.time()) at which those counters were
  read, for display. No reader times a rate by it, because NTP can step the
  wall clock between two readings.
* ``rails_m``, the monotonic time (time.monotonic()) taken at the same
  instant. A reader times a rate by the interval between two of these stamps
  from the same box, so a reading that waited in a cache, or behind the
  slower readings in host_stats(), is not timed by when the reader saw it,
  and a wall-clock step cannot stretch or shrink the interval. On Linux the
  monotonic clock counts from boot for every process on the box, so stamps
  from the driver and a worker on one box compare. When a stamp is missing
  (an older driver) or did not move forward (the same reading again, or a
  reboot), the reader times the rate by its own monotonic clock.
* ``rail_state``, the port state and physical state per rail, read from the
  same files and parsed the same way as the RDMA rails row in ``dgxm doctor``.

A rail is a cabled port: one whose byte counters are not both zero, or whose
physical state reads LinkUp. A port that is neither appears in no map. A rail
that failed link training at boot reads that way too (zero counters, no link),
so it is left out with the uncabled ports; ``dgxm doctor`` reports it. A rail
that is down keeps its state row while its counters or its link show it is
cabled.

Both maps key a rail the same way, so a reader can match its counters to its
state. A device with one rail is keyed by its name, such as ``rocep1s0f0``;
each ConnectX-7 function has one port, so that is every rail on a DGX Spark.
A device with more than one rail keys each one as ``<device>/<port>``, such as
``mlx5_0/2``, so no port hides another. When a device gains or loses its
second rail, its keys switch between the two forms and readers start that
rail's rate over.

rail_counters() and rail_state() return one map each, for callers that need
only one.
"""
from __future__ import annotations

import glob
import os
import time

_SYSFS_IB = "/sys/class/infiniband"
_RAIL_LANE_BYTES = 4  # infiniband port_{rcv,xmit}_data counters tick in 4-byte lanes


def _read(path: str) -> str:
    with open(path) as handle:
        return handle.read().strip()


def _counters(port: str) -> tuple[int, int] | None:
    """RX and TX bytes of one port, or None when a counter cannot be read."""
    try:
        rx = int(_read(os.path.join(port, "counters", "port_rcv_data")))
        tx = int(_read(os.path.join(port, "counters", "port_xmit_data")))
    except (OSError, ValueError):
        return None
    return rx * _RAIL_LANE_BYTES, tx * _RAIL_LANE_BYTES


def _link(port: str) -> dict[str, str] | None:
    """``state`` reads "ACTIVE" when the state file names it, as doctor's test
    does, and the raw text otherwise, such as "1: DOWN". ``phys_state`` is the
    raw text, such as "5: LinkUp". None when either file cannot be read."""
    try:
        state = _read(os.path.join(port, "state"))
        phys = _read(os.path.join(port, "phys_state"))
    except (OSError, ValueError):
        return None
    return {"state": "ACTIVE" if "ACTIVE" in state else state, "phys_state": phys}


def _walk() -> tuple[dict[str, dict[str, int]], float, float, dict[str, dict[str, str]]]:
    ports = sorted(glob.glob(os.path.join(_SYSFS_IB, "*", "ports", "*")))
    counters = {port: _counters(port) for port in ports}
    read_at, read_mono = time.time(), time.monotonic()
    links = {port: _link(port) for port in ports}
    cabled = []
    for port in ports:
        count, link = counters[port], links[port]
        if (count is not None and any(count)) or (link is not None and "LinkUp" in link["phys_state"]):
            dev, _, number = os.path.relpath(port, _SYSFS_IB).split(os.sep)
            cabled.append((port, dev, number))
    per_device: dict[str, int] = {}
    for _, dev, _ in cabled:
        per_device[dev] = per_device.get(dev, 0) + 1
    rails: dict[str, dict[str, int]] = {}
    state: dict[str, dict[str, str]] = {}
    for port, dev, number in cabled:
        key = dev if per_device[dev] == 1 else f"{dev}/{number}"
        count, link = counters[port], links[port]
        if count is not None and any(count):
            rails[key] = {"rx_bytes": count[0], "tx_bytes": count[1]}
        if link is not None:
            state[key] = link
    return rails, read_at, read_mono, state


def rail_readings() -> dict[str, object]:
    """The rail counters, the wall-clock and monotonic times they were read, and
    the link state, from one walk."""
    rails, read_at, read_mono, state = _walk()
    return {"rails": rails, "rails_t": read_at, "rails_m": read_mono, "rail_state": state}


def rail_counters() -> dict[str, dict[str, int]]:
    """Cumulative RX/TX bytes per rail (sysfs; cheap)."""
    return _walk()[0]


def rail_state() -> dict[str, dict[str, str]]:
    """Port state and physical state per rail (sysfs; cheap)."""
    return _walk()[3]
