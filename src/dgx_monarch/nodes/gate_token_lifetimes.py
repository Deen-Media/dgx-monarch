"""Denials have session lifetime; claims have ceremony lifetime; only PASS is capped."""
from __future__ import annotations

from typing import Any


def trim_session(runtime: Any) -> None:
    """Apply the session PASS cache's FIFO cap while its caller holds the lock."""
    while len(runtime._AUTO_GATE_SESSION) > runtime._AUTO_GATE_SESSION_LIMIT:
        runtime._AUTO_GATE_SESSION.pop(next(iter(runtime._AUTO_GATE_SESSION)))
