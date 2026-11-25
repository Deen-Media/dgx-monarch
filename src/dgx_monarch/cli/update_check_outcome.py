"""Read how a verified update's doctor or smoke process ended.

A check that reported unknown stays unknown: the update makes its one
containment stop and nothing more. A check whose process ended with no verdict
at all, killed by a signal or stopped at its time limit, is a named failure, and
so is a smoke whose checks passed while its own ProcMesh stop went unconfirmed:
the update then restores the prior release, which leaves the pair working. That
restore runs only once the stop of every new Worker is confirmed; the Worker
unit stops its whole control group, so that stop ends every actor a check left on
a Worker host, and the check process itself has already exited or was reaped.
"""
from __future__ import annotations

import json

from .cluster_smoke import CLEANUP_CODES
from .probe_certainty import UNKNOWN_EXIT
from .update_types import Certainty, OperationResult
from .update_worker_lifecycle import operation_from_exit

TIMED_OUT = OperationResult(Certainty.FAILED, "timed_out")
KILLED = OperationResult(Certainty.FAILED, "killed")


def doctor_outcome(returncode: int) -> OperationResult:
    """Doctor's exit status: 0, a failed row (1, 2), unknown (75), or no verdict (a signal)."""
    return KILLED if returncode < 0 else operation_from_exit(returncode)


def smoke_outcome(returncode: int, stdout: str) -> OperationResult | None:
    """The smoke's result from its exit status, or None when it passed and its payload decides."""
    if returncode == UNKNOWN_EXIT:
        try:
            payload = json.loads(stdout.splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            return OperationResult(Certainty.UNKNOWN)
        if isinstance(payload, dict) and payload.get("error") in CLEANUP_CODES:
            return OperationResult(Certainty.FAILED, "cleanup_unconfirmed")
        return OperationResult(Certainty.UNKNOWN)
    if returncode < 0:
        return KILLED
    return OperationResult(Certainty.FAILED) if returncode != 0 else None
