"""What one Worker-assisted VAE decode says about itself (stdlib leaf).

Every decode the route sees logs one INFO line on the driver: split over how
many ranks and how many units each took, or not split and why. A rank that
failed, declined or timed out adds one WARNING with the first cause; its units
were decoded here, so the image is the reference either way. The last record
is published in the telemetry route's render block (``vae_decode``), where
``dgxm doctor`` reads it, and as one ``vae_decode`` event in the event ring.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .log import get_logger

log = get_logger("dgx_monarch.vae_decode")


@dataclass
class DecodeRecord:
    """One decode call, as the log line and telemetry report it."""

    family: str
    entry: str
    split: bool = False
    reason: str = ""
    units: int = 0
    local: int = 0
    remote: dict[int, int] = field(default_factory=dict)
    fallback: int = 0
    failures: list[str] = field(default_factory=list)
    seconds: float = 0.0
    budget_waits: int = 0
    peak_reply_bytes: int = 0

    @property
    def ranks(self) -> int:
        """Ranks that decoded at least one unit, this driver included."""
        return (1 if self.local else 0) + sum(1 for count in self.remote.values() if count)

    def as_dict(self) -> dict[str, Any]:
        return {"t": time.time(), "family": self.family, "entry": self.entry, "split": self.split,
                "reason": self.reason, "ranks": self.ranks, "units": self.units, "local": self.local,
                "remote": {str(rank): count for rank, count in sorted(self.remote.items())},
                "fallback": self.fallback, "failures": list(self.failures[:4]),
                "seconds": round(self.seconds, 2), "budget_waits": self.budget_waits,
                "peak_reply_gib": round(self.peak_reply_bytes / (1 << 30), 2)}


_NODES = {"decode": "VAE Decode", "decode_tiled": "VAE Decode (Tiled)"}


def info_line(record: DecodeRecord) -> str:
    """The one INFO line a decode logs."""
    node = _NODES.get(record.entry, record.entry)
    if not record.split:
        return f"VAE decode not split ({record.family or 'unknown'}, {node}): {record.reason}"
    shares = ", ".join(f"{count} on rank {rank}" for rank, count in sorted(record.remote.items()) if count)
    waits = (f"; ranks paused {record.budget_waits} times on the reply budget, "
             f"{record.peak_reply_bytes / (1 << 30):.2f} GiB held at most" if record.budget_waits else "")
    return (f"VAE decode split over {record.ranks} ranks ({record.family}, {node}): "
            f"{record.units} units, {record.local} here, {shares}, {record.seconds:.1f} s{waits}")


def warning_line(record: DecodeRecord) -> str | None:
    """The WARNING a decode logs when any rank fell back, else None."""
    if not record.failures:
        return None
    return (f"Worker-assisted VAE decode fell back to this driver for {record.fallback} of "
            f"{record.units} units, so the image is the one-host decode: {record.failures[0]}")


_LOCK = threading.Lock()
_LAST: dict[str, Any] | None = None  # process lifetime, one slot: the last decode the route saw


def publish(record: DecodeRecord) -> None:
    """Log the decode, keep it as the last one and append one event."""
    global _LAST
    log.info("%s", info_line(record))
    warning = warning_line(record)
    if warning is not None:
        log.warning("%s", warning)
    row = record.as_dict()
    with _LOCK:
        _LAST = row
    try:
        from . import telemetry

        telemetry.emit("vae_decode", **{key: value for key, value in row.items() if key != "t"})
    except Exception:
        pass


def last() -> dict[str, Any] | None:
    with _LOCK:
        return None if _LAST is None else dict(_LAST)


def with_vae_decode(render: object) -> object:
    """Attach the last decode to a render snapshot in place, and return it."""
    row = last()
    if isinstance(render, dict) and row is not None:
        render["vae_decode"] = row
    return render
