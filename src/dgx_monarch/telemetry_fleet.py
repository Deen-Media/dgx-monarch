"""Bounded per-job telemetry for the most recent Fleet wave.

Dispatch opens each row with the intended rank and driver timestamp. A reply
adds the reporting host, worker duration, and completion time; missing replies
leave rows open. Host comparison flags replies from a different configured
box. Prompt text is represented only by a digest.

This standard-library leaf is best effort: telemetry failures never fail a
render.
"""
from __future__ import annotations

import hashlib
import math
import threading
import time
from collections.abc import Mapping

from .log import get_logger

log = get_logger(__name__)

# One wave publishes at most this many rows. nodes/fleet.py admits up to 1024
# jobs and this block rides every telemetry poll, so the tail is dropped and
# the reader is told the listing is partial.
MAX_PUBLISHED_JOBS = 64
_DIGEST_CHARS = 12
_LABEL_CHARS = 32
# A worker names its own host, so the published name gets the bound
# telemetry_events puts on event text (_MAX_EVENT_TEXT, 192 characters). A
# Linux host name holds at most 64 characters, so the cut never changes the
# name the memory cards beside these rows are keyed on.
_HOST_CHARS = 192
# telemetry_events' bound, for the reason given there.
_MAX_SAFE_INTEGER = (1 << 53) - 1


def prompt_digest(text: object) -> str | None:
    """A short stable name for a prompt line, or None when a job carried none.

    A digest, not the line: this rides an HTTP route and the prompt is the
    operator's text, not telemetry.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:_DIGEST_CHARS]


def box_label(rank: object, gpus_per_host: object, host: object) -> str | None:
    """Name the machine a rank sits on, short enough for a sidebar row.

    Box ``rank // gpus_per_host``, the arithmetic capacity_agreement.collect
    builds its own coordinates from, so the number is the driver's and not one
    the reply chose. With no usable gpus_per_host the host's first label stands
    in. Neither path returns a full address; the host that answered rides the
    row in its own field.
    """
    index = box_index(rank, gpus_per_host)
    if index is not None:
        return f"box {index + 1}"
    if isinstance(host, str) and host.strip():
        return host.strip().split(".", 1)[0][:_LABEL_CHARS]
    return None


def box_index(rank: object, gpus_per_host: object) -> int | None:
    """The box a rank sits on from zero: ranks fill boxes in blocks of gpus."""
    gpus = _whole(gpus_per_host)
    index = _whole(rank)
    if index is None or index < 0 or gpus is None or gpus < 1:
        return None
    return index // gpus


def host_names(config: object) -> tuple[str, ...]:
    """This driver's name for each box, in host order; a local fleet has none."""
    hosts = getattr(config, "hosts", None)
    if not isinstance(hosts, (list, tuple)):
        return ()
    names = []
    for host in hosts:
        name = getattr(host, "name", None)
        names.append(name if isinstance(name, str) else "")
    return tuple(names)


def named_box(hosts: tuple[str, ...], host: object) -> int | None:
    """The box this driver's config gives an answering host, or None.

    None covers both "no host list" and "a name this config never uses": a
    config names its boxes by whatever reaches them, an address as readily as
    a hostname, while a worker names itself, and a ledger that cannot tell two
    spellings of one box apart stays quiet rather than mark every row again.
    """
    key = _short(host)
    if not key:
        return None
    for index, name in enumerate(hosts):
        if _short(name) == key:
            return index
    return None


def _short(name: object) -> str:
    """A host name cut to the part two spellings of one box agree on."""
    if not isinstance(name, str):
        return ""
    return name.strip().split(".", 1)[0].lower()


def _whole(value: object) -> int | None:
    """A non-bool int inside the JavaScript-safe range, else None.

    An integer outside that range would break the route it rides, so it reads as absent.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if not -_MAX_SAFE_INTEGER <= value <= _MAX_SAFE_INTEGER:
        return None
    return value


def _wall_s(value: object) -> float | None:
    """Return the worker-reported elapsed seconds, or None if invalid or missing."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        # An integer outside float range cannot be a measured duration.
        return None
    return round(number, 2) if math.isfinite(number) else None


def _elapsed(wave: dict) -> float:
    """Seconds since this wave opened, on a clock no time change can move."""
    return round(max(0.0, time.monotonic() - wave["clock"]), 2)


def _row_for(wave: dict, index: int) -> dict | None:
    """This job's row of the open wave, or None while nothing has opened one."""
    for row in wave["jobs"]:
        if row["job"] == index:
            return row
    return None


class FleetJobs:
    """The last Fleet wave, kept after it ends the way ``render.last`` is.

    One wave and one slot: ``open_wave`` replaces the record whole, so a
    finished wave stays readable until the next Fleet call starts.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wave: dict | None = None

    def open_wave(self, jobs: object, gpus_per_host: object,
                  config: object = None) -> None:
        """Start the record for a new wave and forget the previous one.

        ``config`` is this driver's own cluster config, read for its host
        names alone, so a row can say when a reply named a box the driver did
        not send that job to.
        """
        wave: dict | None
        try:
            rows = list(jobs) if isinstance(jobs, (list, tuple)) else []
            wave = {
                "t": time.time(),
                "clock": time.monotonic(),
                "count": len(rows),
                "gpus_per_host": gpus_per_host,
                "hosts": host_names(config),
                "digests": [
                    prompt_digest(job.get("text")) if isinstance(job, Mapping) else None
                    for job in rows
                ],
                "jobs": [],
                "truncated": False,
            }
        except Exception as exc:
            log.debug("fleet job ledger could not open a wave (%r)", exc)
            wave = None
        with self._lock:
            self._wave = wave

    def start_job(self, job_index: object, rank: object) -> None:
        """Open one job's row as the driver dispatches it: the box, and when."""
        with self._lock:
            wave = self._wave
            if wave is None:
                return
            try:
                self._open(wave, job_index, rank)
            except Exception as exc:
                log.debug("fleet job ledger dropped a dispatch row (%r)", exc)

    def record_job(self, job_index: object, rank: object, result: object) -> None:
        """Close one job's row with the box that answered it, and when."""
        with self._lock:
            wave = self._wave
            if wave is None:
                return
            try:
                self._append(wave, job_index, rank, result)
            except Exception as exc:
                log.debug("fleet job ledger dropped a job row (%r)", exc)

    @staticmethod
    def _open(wave: dict, job_index: object, rank: object) -> None:
        index = _whole(job_index)
        if index is None or not 0 <= index < wave["count"]:
            return
        if _row_for(wave, index) is not None:
            return
        if len(wave["jobs"]) >= MAX_PUBLISHED_JOBS:
            wave["truncated"] = True
            return
        submitted = _whole(rank)
        # The box the driver sent this job to, which is all it knows yet. The
        # answer fills the rest of the row in, and never arrives after a stop.
        wave["jobs"].append({
            "job": index,
            "prompt": wave["digests"][index],
            "host": None,
            "rank": submitted,
            "box": box_label(submitted, wave["gpus_per_host"], None),
            "wall_s": None,
            "started_s": _elapsed(wave),
            "ended_s": None,
        })

    @staticmethod
    def _append(wave: dict, job_index: object, rank: object, result: object) -> None:
        index = _whole(job_index)
        if index is None or not 0 <= index < wave["count"]:
            return
        row = _row_for(wave, index)
        if row is not None and row["ended_s"] is not None:
            return
        if row is None and len(wave["jobs"]) >= MAX_PUBLISHED_JOBS:
            wave["truncated"] = True
            return
        answer = result if isinstance(result, Mapping) else {}
        # Read the whole answer before touching the row: a reply that raises
        # halfway through costs the close, never half a row. With no
        # dispatch row it costs the row; with one, that row stays open.
        named = answer.get("host")
        host = named[:_HOST_CHARS] if isinstance(named, str) else None
        submitted = _whole(rank)
        job = {
            "job": index,
            "prompt": wave["digests"][index],
            "host": host,
            "rank": submitted,
            "box": box_label(submitted, wave["gpus_per_host"], host),
            "wall_s": _wall_s(answer.get("sample_s")),
            "started_s": row["started_s"] if row is not None else None,
            "ended_s": _elapsed(wave),
        }
        # Fleet workers report rank 0 in their own world-1 mesh. Keep the driver's
        # cluster-wide rank and compare only the reply's host with the intended
        # box; comparing ranks would flag healthy jobs on every other box.
        answered = named_box(wave["hosts"], host)
        expected = box_index(submitted, wave["gpus_per_host"])
        if answered is not None and expected is not None and answered != expected:
            job["host_reported"] = host
        if row is None:
            wave["jobs"].append(job)
        else:
            row.update(job)

    def snapshot(self) -> dict | None:
        """The public block, or None when this driver has run no Fleet wave."""
        with self._lock:
            wave = self._wave
            if wave is None:
                return None
            return {
                # When this wave opened, so a panel reading a retained block
                # after an unrelated render can say which wave it is looking at.
                "t": wave["t"],
                "count": wave["count"],
                "truncated": wave["truncated"],
                # Dispatch order, which is the order the jobs started, whatever
                # order their answers came back in.
                "jobs": [dict(row) for row in sorted(
                    wave["jobs"], key=lambda row: row["job"])],
            }


# Process lifetime, one slot: this driver's last Fleet wave. nodes/fleet.py
# replaces it per wave and nothing else writes it.
fleet_jobs = FleetJobs()


def open_wave_for(jobs: object, handle: object) -> None:
    """Start a wave's record: the ledger reads the handle, the node does not."""
    fleet_jobs.open_wave(jobs, getattr(handle, "gpus_per_host", 0),
                         getattr(handle, "config", None))


def with_fleet(render: object) -> object:
    """Attach the last wave's block to a render snapshot in place, and return it.

    A copy would split the one render block the route publishes into two.
    """
    block = fleet_jobs.snapshot()
    if isinstance(render, dict) and block is not None:
        render["fleet"] = block
    return render
