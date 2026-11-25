"""Retire the release slots no host needs, once a verified update is finalized on every host.

This is housekeeping after the update's outcome is decided: nothing here
changes the outcome code. The receipt counts only what was removed.

Time: on each host the helper starts no new removal after
``RETIRE_BUDGET_S`` (120 s), nor after its caller stops reading, and holds the
lifecycle lock for that long at most, plus the one removal in progress. The
call allows twice the budget, ``RETIRE_CALL_S``, and that also covers the wait
for the lifecycle lock; a run whose lock wait and work outlast it is cut. A
cut run (a timeout, a lost connection, a killed helper) leaves at most one
slot renamed out of the slot pattern, which no update can take for a slot and
the next run removes first, and stops at its next output line.
"""
from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ..config import ClusterConfig, HostConfig
from .update_retire_remote import RETIRE_BUDGET_S
from .update_types import Certainty, OperationResult

if TYPE_CHECKING:  # the transaction imports this module; the host side loads only when it runs
    from .update_release_types import HostRunner

RETIRE_CALL_S = int(2 * RETIRE_BUDGET_S)
_REMOVED = re.compile(r"REMOVED \S+ ([0-9]{1,18})")
_RETIRED = re.compile(r"RETIRED ([0-9]{1,9}) ([0-9]{1,9}) ([0-9]{1,9})")


@dataclass(frozen=True)
class HostRetirement:
    confirmed: bool
    removed: int = 0
    bytes: int = 0
    not_removed: int = 0


def _read(stdout: object, returncode: int | None) -> HostRetirement:
    """Count the slots the helper printed as removed; confirmed only by its final line and exit 0."""
    text = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout if isinstance(stdout, str) else ""
    sizes = [int(match[1]) for match in map(_REMOVED.fullmatch, text.splitlines()) if match]
    final = next((match for match in map(_RETIRED.fullmatch, text.splitlines()) if match), None)
    confirmed = returncode == 0 and final is not None and int(final[1]) == len(sizes)
    not_removed = int(final[2]) + int(final[3]) if final is not None else 0
    return HostRetirement(confirmed, len(sizes), sum(sizes), not_removed)


def retire_host(config: ClusterConfig, host: HostConfig, token: str, snapshot: dict[str, Any], run: HostRunner) -> HostRetirement:
    from .update_installation import retire_script

    try:
        result = run(config, host, retire_script(config, host, token, snapshot), RETIRE_CALL_S)
    except subprocess.TimeoutExpired as exc:
        return _read(exc.stdout, None)
    except OSError:
        return HostRetirement(False)
    return _read(result.stdout, result.returncode)


def retire_slots(config: ClusterConfig, token: str, installations: Mapping[str, dict[str, Any]], run: HostRunner) -> OperationResult:
    """Retire every host's unneeded slots; one host's failure or a stop never touches the outcome."""
    counts = {"releases_retired": 0, "bytes_retired": 0, "releases_not_retired": 0, "hosts_retire_unconfirmed": 0}
    hosts: Sequence[HostConfig] = config.hosts
    for index, host in enumerate(hosts):
        try:
            outcome = retire_host(config, host, token, installations[host.name], run)
        except Exception:
            outcome = HostRetirement(False)
        except BaseException:
            counts["hosts_retire_unconfirmed"] += len(hosts) - index
            print("the retirement of old release slots was stopped; some may remain, and the next update tries again",
                  file=sys.stderr)
            break
        counts["releases_retired"] += outcome.removed
        counts["bytes_retired"] += outcome.bytes
        counts["releases_not_retired"] += outcome.not_removed
        if not outcome.confirmed:
            counts["hosts_retire_unconfirmed"] += 1
            print(f"the retirement of old release slots on {host.name} was not confirmed after {outcome.removed} "
                  "were removed; some may remain, and the next update tries again", file=sys.stderr)
        elif outcome.not_removed:
            print(f"{outcome.not_removed} old release slots on {host.name} were not removed this time; "
                  "the next update tries again", file=sys.stderr)
    unconfirmed = counts["hosts_retire_unconfirmed"] > 0
    return OperationResult(Certainty.UNKNOWN if unconfirmed else Certainty.SUCCEEDED, counts=counts)


def retire_after_finalize(ops: Any, staged: object) -> OperationResult:
    """Run the update's retirement; any failure or stop, a missing method included, reads as unconfirmed."""
    try:
        result = ops.retire_releases(staged)
    except BaseException:  # the update already finished: a stop here ends only the housekeeping
        return OperationResult(Certainty.UNKNOWN)
    return result if isinstance(result, OperationResult) else OperationResult(Certainty.UNKNOWN)


def retirement_status(result: OperationResult) -> Literal["succeeded", "partial", "unknown"]:
    """The receipt step status: succeeded, partial when some slots stayed, unknown when a host did not confirm."""
    counts = result.counts or {}
    if result.certainty != Certainty.SUCCEEDED:
        return "unknown"
    return "partial" if counts.get("releases_not_retired", 0) else "succeeded"
