"""The ``dgxm doctor`` row for Worker-assisted VAE decode.

It names the ``[worker_args] worker_vae_decode`` setting and, when it is
``"auto"``, what the last stock VAE decode on this box's driver did: split
over how many ranks, stayed on the driver and why, or fell back to the driver
after a rank failed. The decode record lives in the driver, so the row reads
the driver's telemetry over HTTP, as the mesh health row does, and never
attaches a mesh.
"""
from __future__ import annotations

import json
import urllib.request
from collections.abc import Mapping
from typing import Any

from ..config import ClusterConfig
from . import comfy_ports
from .probe_certainty import OK, WARN

NAME = "worker VAE decode"
ENTRY = "docs/TROUBLESHOOTING.md #140"
_TIMEOUT_S = 1.5


def _setting(config: ClusterConfig | None) -> str:
    worker_args = dict(getattr(config, "worker_args", None) or {})
    value = worker_args.get("worker_vae_decode", "off")
    return value if value in ("off", "auto") else "off"


def fetch_last(candidates: tuple[str, ...] | None = None) -> tuple[bool, Mapping[str, Any] | None]:
    """``(answered, record)``: whether any local driver answered, and its newest decode record."""
    probes = comfy_ports.driver_candidates() if candidates is None else candidates
    answered = False
    newest: Mapping[str, Any] | None = None
    for candidate in probes:
        try:
            with urllib.request.urlopen(f"http://{candidate}/dgxm/telemetry", timeout=_TIMEOUT_S) as response:
                payload = json.load(response)
        except Exception:
            continue
        answered = True
        render = payload.get("render") if isinstance(payload, Mapping) else None
        record = render.get("vae_decode") if isinstance(render, Mapping) else None
        if isinstance(record, Mapping) and (newest is None or float(record.get("t", 0)) > float(newest.get("t", 0))):
            newest = record
    return answered, newest


def verdict(setting: str, answered: bool, record: Mapping[str, Any] | None) -> tuple[str, str, str]:
    """The row for a setting and the last decode a driver reported."""
    if setting != "auto":
        return OK, NAME, 'off ([worker_args] worker_vae_decode); every decode stays on the driver'
    if not answered:
        return OK, NAME, "auto; no local driver answered, so the last decode is unobserved"
    if record is None:
        return OK, NAME, "auto; the driver has made no stock VAE decode since it started"
    family = record.get("family", "unknown")
    if record.get("failures"):
        failures = list(record.get("failures") or [])
        return (WARN, NAME, f"auto; the last decode ({family}) fell back to the driver for "
                f"{record.get('fallback', 0)} of {record.get('units', 0)} units: {failures[0]} "
                f"({ENTRY})")
    if record.get("split"):
        remote = dict(record.get("remote") or {})
        shares = ", ".join(f"{count} on rank {rank}" for rank, count in sorted(remote.items()))
        return (OK, NAME, f"auto; the last decode ({family}) split over {record.get('ranks', 0)} ranks: "
                f"{record.get('units', 0)} units, {record.get('local', 0)} on the driver, {shares}, "
                f"{float(record.get('seconds', 0.0)):.1f} s")
    return OK, NAME, f"auto; the last decode ({family}) stayed on the driver: {record.get('reason', '')}"


def doctor_row(config: ClusterConfig | None) -> tuple[str, str, str]:
    """This box's row; never raises."""
    try:
        setting = _setting(config)
        answered, record = fetch_last() if setting == "auto" else (False, None)
        return verdict(setting, answered, record)
    except Exception as exc:
        return WARN, NAME, f"probe unavailable ({exc!r})"
