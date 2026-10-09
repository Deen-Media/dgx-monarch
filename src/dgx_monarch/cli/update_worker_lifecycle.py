"""Generation-bound Worker lifecycle adapters for verified updates."""
from __future__ import annotations

import subprocess
from typing import Any

from ..config import ClusterConfig
from . import lifecycle
from .lifecycle_generation import validate_generation
from .probe_certainty import UNKNOWN_EXIT
from .update_types import Certainty, OperationResult, StagedRelease

# The unknown exit that doctor and status also use: the CLI has one unknown code.
LIFECYCLE_UNKNOWN_EXIT = UNKNOWN_EXIT


def operation_from_lifecycle(result: bool | None) -> OperationResult:
    certainty = (
        Certainty.SUCCEEDED
        if result is True
        else Certainty.UNKNOWN
        if result is None
        else Certainty.FAILED
    )
    return OperationResult(certainty)


def operation_from_exit(returncode: int) -> OperationResult:
    """Classify the child process's outcome from its exit status.

    A negative return code means a signal terminated the child. It cannot establish
    whether remote hosts were started, so the result remains unknown.
    """
    certainty = (
        Certainty.SUCCEEDED
        if returncode == 0
        else Certainty.UNKNOWN
        if returncode == LIFECYCLE_UNKNOWN_EXIT or returncode < 0
        else Certainty.FAILED
    )
    return OperationResult(certainty)


def start_target_main(config_path: str, generation: str | None) -> int:
    """Start target workers, returning 75 when the outcome is unobserved.

    Reserve exit 1 for a definite False from ``up``. Escaped exceptions must not
    share that code, which could authorize rollback over already-started hosts.
    """
    from ..config import load_cluster_config
    from .lifecycle import up

    try:
        result = up(
            load_cluster_config(config_path), sync=False, generation=generation)
    except BaseException:  # an escaped fault observed nothing at all
        return LIFECYCLE_UNKNOWN_EXIT
    if result is True:
        return 0
    return LIFECYCLE_UNKNOWN_EXIT if result is None else 1


def start_script() -> str:
    return (
        "import sys; "
        "from dgx_monarch.cli.update_worker_lifecycle import start_target_main; "
        "raise SystemExit(start_target_main(sys.argv[1], sys.argv[2] or None))"
    )


def staged_generation(releases: Any, staged: StagedRelease) -> str | None:
    """Return the transaction generation represented by the exact staged slot."""
    slot = getattr(releases, "slot", None)
    if slot is None:
        return None
    metadata = getattr(slot, "metadata", None)
    try:
        generation = validate_generation(slot.token)
    except (AttributeError, ValueError):
        generation = None
    if generation is None or metadata is None:
        return None
    actual = (
        getattr(metadata, "target_sha", None),
        getattr(metadata, "version", None),
        getattr(metadata, "torchmonarch_pin", None),
        getattr(metadata, "source_manifest", None),
        getattr(metadata, "dependency_manifest", None),
    )
    expected = (
        staged.target_sha,
        staged.version,
        staged.torchmonarch_pin,
        staged.source_manifest,
        staged.dependency_manifest,
    )
    return generation if actual == expected else None


def active_generation(releases: Any, staged: StagedRelease) -> str | None:
    """Require that the exact staged generation is also the active target."""
    generation = staged_generation(releases, staged)
    slot = getattr(releases, "slot", None)
    return generation if getattr(slot, "remote_activated", False) is True else None


def prior_generation(releases: Any, staged: StagedRelease) -> str | None:
    """Require confirmed restoration before starting the verified prior release."""
    generation = staged_generation(releases, staged)
    slot = getattr(releases, "slot", None)
    if (
        generation is None
        or getattr(slot, "remote_activated", True) is not False
        or getattr(slot, "activation_ambiguous", True) is not False
        or getattr(slot, "prior_metadata", None) is None
    ):
        return None
    return generation


def stop_active_generation(
    config: ClusterConfig, releases: Any, staged: StagedRelease
) -> OperationResult:
    """Stop Worker services only if they still run this transaction's generation."""
    generation = staged_generation(releases, staged)
    if generation is None:
        return OperationResult(Certainty.UNKNOWN)
    try:
        stopped = lifecycle.down_if_generation(config, generation)
    except (OSError, subprocess.TimeoutExpired):
        return OperationResult(Certainty.UNKNOWN)
    return operation_from_lifecycle(stopped)
