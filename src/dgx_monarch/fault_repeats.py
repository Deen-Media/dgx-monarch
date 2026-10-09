"""Report each distinct unhandled cluster fault, then count exact repeats.

The first report stays complete. Repeat counts include it and are logged at
2, 4, 8, and later powers of two, keeping persistent faults visible without
repeating full tracebacks. A different fault starts a new count.
"""
from __future__ import annotations

import threading
from typing import Any


def headline(text: str, limit: int = 200) -> str:
    """The first two lines of a fault: the actor that failed, then why."""
    return " ".join(part.strip() for part in text.split("\n")[:2])[:limit]


class RepeatGate:
    """One fault hook's memory of the last fault it reported."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._text: str | None = None
        self._count = 0

    def first_report(self, text: str, log: Any) -> bool:
        """True when the caller should report this fault whole."""
        with self._lock:
            if text != self._text:
                self._text, self._count = text, 1
                return True
            self._count += 1
            count = self._count
        if count & (count - 1) == 0:
            log.error("cluster fault repeated %d times: %s", count, headline(text))
        return False
