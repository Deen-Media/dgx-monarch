"""The ``dgxm doctor`` row for how the driver's ComfyUI reads free memory.

On a GPU that shares the host's memory, ComfyUI's own free-memory reading
follows MemFree, and the pack replaces it in the driver with MemAvailable less
a floor (``driver_uma_memory``). This row says which of the two the running
driver reads, from the driver's own numbers: ComfyUI's stock ``/system_stats``
route reports the device's free figure and the host's available memory from
one moment in one process, so the row reads that route over HTTP, as the
frontend row does. Doctor is passive and runs beside a live render, so it
makes no CUDA context of its own and never attaches a mesh.

On such a GPU the driver also encodes a batch of images one image a call
(``vae_encode_item``). No number shows that from outside, so the driver
publishes the rule's state in ``/dgxm/telemetry`` (``driver_rules``) and the
row repeats it: on, off by the driver's switch, or stepped aside. A state the
driver did not publish is said to be unobserved and changes no verdict.

The classifier takes the parsed answer and needs no driver, CUDA or network.
"""
from __future__ import annotations

import json
import urllib.request
from collections.abc import Mapping
from typing import Any

from .. import vae_encode_item
from ..driver_uma_memory import DISABLE_ENV, floor_bytes
from . import comfy_ports
from .probe_certainty import OK, WARN

NAME = "driver memory reading"
ENTRY = "docs/TROUBLESHOOTING.md #12"
# Where ``fetch`` puts the driver's ``driver_rules`` telemetry block, and the rule's name in it.
RULES_KEY = "dgxm_driver_rules"
ENCODE_RULE = "vae_encode_item"
_TELEMETRY_KEY = "driver_rules"
_KNOWN_STATES = frozenset({vae_encode_item.NOT_LINUX, vae_encode_item.NOT_INTEGRATED, vae_encode_item.NO_COMFY,
                           vae_encode_item.PROBE_FAILED, vae_encode_item.UNSET})
_TIMEOUT_S = 1.5
_GIB = 2 ** 30
# The device and the host report one pool when their totals agree this closely
# (a Spark logs "Total VRAM 124543 MB, total RAM 124543 MB").
_ONE_POOL_RATIO = 0.01
# The route reads the host figure and the device figure microseconds apart.
_AGREE_BYTES = _GIB


def _driver_rules(candidate: str) -> object:
    """The driver's own ``driver_rules`` block from ``/dgxm/telemetry``, or None when it did not say."""
    try:
        # The route's first answer after a driver starts polls the Workers; later ones are cached.
        with urllib.request.urlopen(f"http://{candidate}/dgxm/telemetry",
                                    timeout=comfy_ports._TELEMETRY_TIMEOUT_S) as response:
            telemetry = json.load(response)
    except Exception:
        return None
    return telemetry.get(_TELEMETRY_KEY) if isinstance(telemetry, Mapping) else None


def fetch(candidates: tuple[str, ...] | None = None) -> list[tuple[str, Mapping[str, Any]]]:
    """``(host:port, answer)`` for every local ComfyUI that serves this pack and answers.

    The answer is the driver's ``/system_stats`` payload, with what the driver
    itself published about its rules beside it under ``RULES_KEY``.
    """
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
            answers.append((candidate, {**payload, RULES_KEY: _driver_rules(candidate)}))
    return answers


def encode_clause(answer: Mapping[str, Any]) -> tuple[str | None, str]:
    """``(status, clause)`` for the encode rule in one driver, from what that driver published.

    Doctor is another process and its environment is not the driver's, so the
    state is read, never inferred. With nothing published the status is None:
    an unobserved rule does not change the reading's verdict.
    """
    rules = answer.get(RULES_KEY)
    state = rules.get(ENCODE_RULE) if isinstance(rules, Mapping) else None
    if state == vae_encode_item.INSTALLED:
        return OK, "a VAE encode of several images runs one image a call"
    moves = "so the same images can encode to other bits as memory moves"
    if state == vae_encode_item.OFF:
        return WARN, ("a VAE encode of several images takes ComfyUI's memory-sized batches, because the "
                      f"driver runs with {vae_encode_item.DISABLE_ENV}=1, {moves}")
    if state == vae_encode_item.STEPPED_ASIDE:
        return WARN, ("the one image a call encode rule stepped aside in this driver (one WARNING in its log "
                      f"says why) and ComfyUI's memory-sized batches answer, {moves}; this pack version and "
                      "this ComfyUI do not go together for that call")
    if isinstance(state, str) and state in _KNOWN_STATES:
        return WARN, (f"the one image a call encode rule is not installed in this driver ({state}), {moves}")
    return None, ("whether a VAE encode of several images runs one image a call was not observed (the "
                  "driver's /dgxm/telemetry did not say)")


def _with_encode(status: str, detail: str, answer: Mapping[str, Any]) -> tuple[str, str]:
    encode_status, clause = encode_clause(answer)
    return (WARN if WARN in (status, encode_status) else status), f"{detail}; {clause}"


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
        return _with_encode(OK, (
            f"reads the box as it is: {device_free / _GIB:.1f} GiB free on the device, which is the "
            f"{available / _GIB:.1f} GiB available less a {floor / _GIB:.1f} GiB floor"), payload)
    return _with_encode(WARN, (
        f"reads {device_free / _GIB:.1f} GiB free on a device that shares {available / _GIB:.1f} GiB "
        "available with the host: ComfyUI's own reading, which follows MemFree, so its batch choices "
        "move with the page cache. This is one reading, and a render allocating at that moment can "
        "show the same, so run doctor again before acting; then restart the driver on this pack version "
        f"without {DISABLE_ENV} ({ENTRY})"), payload)


def verdict(answers: list[tuple[str, Mapping[str, Any]]],
            machine_total: float | None = None) -> tuple[str, str, str]:
    """The row for every driver that answered; the worst one leads."""
    if not answers:
        return OK, NAME, ("no ComfyUI driver with this pack answered on this box, so there is no reading "
                          "to check; if this box runs the driver, start it and run doctor again. On a GPU "
                          "that shares the host's memory the driver corrects the reading at node import "
                          f"unless {DISABLE_ENV}=1, "
                          "and encodes a batch of images one image a call unless "
                          f"{vae_encode_item.DISABLE_ENV}=1")
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
