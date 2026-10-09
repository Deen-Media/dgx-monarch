"""Latch failed quarantine pushes only when worker mutation is unresolved.

``MeshHandle.apply_worker_args`` checks dirty state, sample leases, mutation
authority, and READY setup before dispatch. The three typed preflight failures
below have no remote effects and need no new DIRTY latch.
``force_stock_quarantine`` handles absent READY setup before the call. Other
failures remain fail-closed, preserving any outcome already published by the
dispatch.
"""
from __future__ import annotations

from typing import Any

from ..mesh_session import ConcurrentRenderSessionError
from ..mesh_setup_state import LifecycleBusyError, TopologyTransitionError
from ..transfer_utils import safe_note

# Driver-decided refusals, each raised from a preflight guard ahead of any
# dispatch. tests/test_refusal_classes.py files all three in
# NOT_REFUSAL_CLASSES as lifecycle faults: no worker saw the call.
PREFLIGHT_REFUSALS = (
    LifecycleBusyError,
    TopologyTransitionError,
    ConcurrentRenderSessionError,
)


def retire_failed_quarantine(
    handle: Any, exc: BaseException, timeout_s: float
) -> None:
    """Retire the policy marker, and latch DIRTY only for an unknown outcome."""
    from .. import mesh_setup
    from ..mesh import MeshHandle

    try:
        if not isinstance(handle, MeshHandle):
            return
        with handle.lock:
            handle.worker_args_key = None
            handle.active_worker_args = {}
            if isinstance(exc, PREFLIGHT_REFUSALS):
                # A verdict the driver reached with nothing in flight. Any
                # state a prior latch published stays exactly as it was.
                return
            if handle.setup_cleanup_state is None:
                # ``apply_worker_args`` publishes its own verdict for anything
                # it dispatched, naming the phase the operator has to act on.
                # Write only when none is published: a skip leaves a latch in
                # place, so it cannot drop one, and overwriting a published
                # verdict would blur its cause.
                handle.setup_cleanup_state = mesh_setup.cleanup_failure(
                    int(getattr(handle, "setup_generation", 0)),
                    "identity-gate stock quarantine", float(timeout_s), exc)
    except BaseException as latch_exc:
        safe_note(exc, "identity-gate quarantine DIRTY latch failed", latch_exc)
