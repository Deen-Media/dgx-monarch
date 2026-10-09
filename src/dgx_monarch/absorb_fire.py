"""Log each teardown fault-suppression decision.

The fault-suppression logic was built against torchmonarch 0.5.0; two-host
operation still exercises it under 0.6.0 (docs/TROUBLESHOOTING.md #60). Keep per-marker logs
when evaluating whether a rule can be removed. This module retains no state.
``token_within_authority`` logs under ``handle.lock`` so the outcome and token
come from one consistent snapshot.
"""
from __future__ import annotations

from collections.abc import Iterable

from .log import get_logger

TAG = "dgxm-absorb-fire"
# Tests pin these names to the call sites and the docs/TROUBLESHOOTING.md #60 table.
SITES = ("reason-marked", "fault-marker", "in-flight", "grace-window",
         "token-authority", "string-fallback")
# Keep absorption evidence on the lifecycle logger operators already collect.
_log = get_logger("dgx_monarch.mesh")


def fire(site: str, detail: str = "") -> None:
    """Log one site and marker without duplicating the surrounding fault."""
    try:
        _log.info("%s site=%s detail=%r", TAG, site, detail)
    except BaseException:
        # A logging failure must never become a teardown failure.
        pass


def fire_if(site: str, taken: bool, detail: str = "") -> bool:
    """Record the site only when its branch was taken; return ``taken``."""
    if taken:
        fire(site, detail)
    return taken


def verdict(site: str, granted: bool) -> bool:
    """Log a time-window verdict without changing it."""
    fire(site, "granted" if granted else "expired")
    return granted


def hits(site: str, text: str, markers: Iterable[str]) -> bool:
    """Log each matching marker so pruning evidence remains marker-specific."""
    matched = [marker for marker in markers if marker in text]
    for marker in matched:
        fire(site, marker)
    return bool(matched)
