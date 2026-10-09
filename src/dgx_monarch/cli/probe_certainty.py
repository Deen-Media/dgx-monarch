"""Represent unobserved probe outcomes consistently across operator commands.

Doctor rows preserve their ok/warn/fail badges and carry uncertainty in a
reason token used by readiness and JSON consumers. Child commands carry it in
an exit code. Verified update uses this distinction to avoid rollback when a
probe lost transport rather than observed a failure.
"""
from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence

OK = "ok"
WARN = "WARN"
FAIL = "FAIL"

# EX_TEMPFAIL. The command reached no verdict rather than a negative one, and
# it stays non-zero so no shell script reads it as success.
UNKNOWN_EXIT = 75
UNOBSERVED = "unobserved"


def probe_row(
    status: str, name: str, detail: str, reason: str = "", critical: bool = False
) -> dict:
    """One doctor row, printed as it lands and tagged with what it observed.

    A detail opening with the ``unobserved:`` convention carries the token
    without restating it at the call site, so the prose an operator reads and
    the field a script reads cannot drift apart. ``critical`` marks the probes
    the run's own verdict rests on; only an unobserved critical row moves the
    exit code to ``UNKNOWN_EXIT``.
    """
    print(f"  [{status:^4}] {name}: {detail}")
    reason = reason or (UNOBSERVED if detail.startswith(f"{UNOBSERVED}:") else "")
    return {"status": status, "name": name, "detail": detail,
            **({"reason": reason} if reason else {}),
            **({"critical": True} if critical and reason else {})}


def unobserved(row: Mapping[str, object]) -> bool:
    """Whether this row's probe returned no observation at all."""
    return str(row.get("reason", "")) == UNOBSERVED


def exit_code(rows: Sequence[Mapping[str, object]], failures: int) -> int:
    """Return 1 on definite failure, UNKNOWN_EXIT on critical uncertainty, else 0.

    Unobserved advisory probes retain their row details but do not affect the exit
    code.
    """
    if failures:
        return 1
    blind = any(unobserved(row) and row.get("critical") is True for row in rows)
    return UNKNOWN_EXIT if blind else 0


def transport_unobserved(error: BaseException) -> bool:
    """Whether this failure means the probe never ran, not that it failed."""
    return isinstance(error, (OSError, subprocess.TimeoutExpired))


def blind_row(status: str, error: BaseException | None = None) -> tuple[str, str]:
    """``(status, reason)`` for a check whose probe may not have answered."""
    if error is not None and not transport_unobserved(error):
        return status, ""
    return WARN, UNOBSERVED
