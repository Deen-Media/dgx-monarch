"""The ``dgxm doctor`` row for how the driver's ComfyUI reads free memory.

On a GPU that shares the host's memory, ComfyUI's own free-memory reading
follows MemFree, and the pack replaces it in the driver with MemAvailable less
a floor (``driver_uma_memory``). This row says which of the two the running
driver reads, from the driver's own numbers: ComfyUI's stock ``/system_stats``
route reports the device's free figure and the host's available memory from
one moment in one process, so the row reads that route over HTTP, as the
frontend row does. Doctor is passive and runs beside a live render, so it
makes no CUDA context of its own and never attaches a mesh.

The classifier takes the parsed payload and needs no driver, CUDA or network.
"""
from __future__ import annotations

import json
import urllib.request
from collections.abc import Mapping
from typing import Any

from ..driver_uma_memory import DISABLE_ENV, floor_bytes
from . import comfy_ports
from .probe_certainty import OK, WARN

NAME = "driver memory reading"
ENTRY = "docs/TROUBLESHOOTING.md #12"
_TIMEOUT_S = 1.5
_GIB = 2 ** 30
# The device and the host report one pool when their totals agree this closely
# (a Spark logs "Total VRAM 124543 MB, total RAM 124543 MB").
_ONE_POOL_RATIO = 0.01
# The route reads the host figure and the device figure microseconds apart.
_AGREE_BYTES = _GIB


def fetch(candidates: tuple[str, ...] | None = None) -> list[tuple[str, Mapping[str, Any]]]:
    """``(host:port, /system_stats payload)`` for every local ComfyUI that serves this pack and answers."""
    probes = comfy_ports.driver_candidates() if candidates is None else candidates
    answers: list[tuple[str, Mapping[str, Any]]] = []
    for candidate in probes:
        if comfy_ports._confirmed_driver(candidate) is False:
            continue  # not this pack's driver: ComfyUI's own reading is the user's choice there
        try:
            with urllib.request.urlopen(f"http://{candidate}/system_stats", timeout=_TIMEOUT_S) as response:
                payload = json.load(response)
        except Exception:
            continue
        if isinstance(payload, Mapping):
            answers.append((candidate, payload))
    return answers


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def classify(payload: Mapping[str, Any], machine_total: float | None = None) -> tuple[str, str]:
    """``(status, detail)`` for one driver's ``/system_stats`` payload.

    ``machine_total`` is this box's memory, which doctor shares with the driver. Under a cgroup
    limit ComfyUI reports the limit as its host total, so one pool is judged against the box.
    """
    system = payload.get("system")
    devices = payload.get("devices")
    device = devices[0] if isinstance(devices, list) and devices and isinstance(devices[0], Mapping) else None
    if not isinstance(system, Mapping) or device is None:
        return WARN, "its /system_stats answer names no device, so the reading is unobserved"
    if device.get("type") != "cuda":
        return OK, f"device type {device.get('type')!r}: ComfyUI's own reading, nothing to correct"
    total = _number(device.get("vram_total"))
    free = _number(device.get("vram_free"))
    cached = _number(device.get("torch_vram_free"))
    host_total = _number(system.get("ram_total"))
    available = _number(system.get("ram_free"))
    if total is None or free is None or cached is None or host_total is None or available is None or total <= 0:
        return WARN, "its /system_stats answer lacks a memory figure, so the reading is unobserved"
    pool = machine_total if machine_total is not None else host_total
    if abs(total - pool) > _ONE_POOL_RATIO * total:
        return OK, (f"the GPU has memory of its own ({total / _GIB:.0f} GiB beside {pool / _GIB:.0f} GiB "
                    "of host memory): ComfyUI's own reading, nothing to correct")
    floor = floor_bytes(int(total))
    device_free = free - cached
    corrected = max(0.0, available - floor)
    if abs(device_free - corrected) <= _AGREE_BYTES:
        return OK, (f"reads the box as it is: {device_free / _GIB:.1f} GiB free on the device, which is the "
                    f"{available / _GIB:.1f} GiB available less a {floor / _GIB:.1f} GiB floor")
    return WARN, (f"reads {device_free / _GIB:.1f} GiB free on a device that shares {available / _GIB:.1f} GiB "
                  "available with the host: ComfyUI's own reading, which follows MemFree, so its VAE batches "
                  "move with the page cache. This is one reading, and a render allocating at that moment can "
                  "show the same, so run doctor again before acting; then restart the driver on this pack version "
                  f"without {DISABLE_ENV} ({ENTRY})")


def verdict(answers: list[tuple[str, Mapping[str, Any]]],
            machine_total: float | None = None) -> tuple[str, str, str]:
    """The row for every driver that answered; the worst one leads."""
    if not answers:
        return OK, NAME, ("no ComfyUI driver with this pack answered on this box, so there is no reading "
                          "to check; start the driver and run doctor again. On a GPU that shares the "
                          f"host's memory it corrects the reading at node import unless {DISABLE_ENV}=1")
    verdicts = [(candidate, *classify(payload, machine_total)) for candidate, payload in answers]
    worst = next((item for item in verdicts if item[1] == WARN), verdicts[0])
    others = [f"{candidate}: {status}" for candidate, status, _detail in verdicts if candidate != worst[0]]
    detail = f"{worst[0]}: {worst[2]}"
    if others:
        detail += f" (also probed {', '.join(others)})"
    return worst[1], NAME, detail


def doctor_row() -> tuple[str, str, str]:
    """This box's row; never raises."""
    try:
        import psutil

        return verdict(fetch(), float(psutil.virtual_memory().total))
    except Exception as exc:
        return WARN, NAME, f"probe unavailable ({exc!r})"
