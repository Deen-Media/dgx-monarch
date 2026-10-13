"""Driver-side native-return latch: after a failed native read, later setups use messaging.

A native latent read that fails on the driver keeps its destination rooted and
sends no ACK (docs/DESIGN.md section 5.6). Without this latch the next fleet
registers natively again and the same fault repeats on every render, rooting
one more latent each time. ``record`` latches native return off;
``native_enabled`` then answers False, so the next setup dispatch sends
``rdma_latent_return = False`` and the actors return latents by messaging.

The latch changes only what the next setup requests. It never frees, drops or
acknowledges anything, so the ownership rules stay as they are. It clears only
when the driver process exits: a recycle, including the abandoned-lease
self-heal, keeps it, because the rooted destination survives a recycle and a
recycle does not show that the fault went away (docs/TROUBLESHOOTING.md #115).
"""
from __future__ import annotations

import threading
import time
from typing import Any

from .error_utils import failure_text, safe_call
from .log import get_logger

log = get_logger(__name__)

TROUBLESHOOTING = "docs/TROUBLESHOOTING.md #115"
_LOCK = threading.Lock()


def clear_state() -> dict[str, Any]:
    """The state a fresh driver process starts with."""
    return {"latched": False, "reason": None, "latched_unix_s": None,
            "fault_count": 0, "last_fault_unix_s": None}


# Process lifetime: only a driver restart clears it (the module docstring says why).
_STATE: dict[str, Any] = clear_state()


class NativeReturnLatchedError(RuntimeError):
    """A render's native latent return failed on the driver; later setups use messaging."""


def _reason(error: BaseException) -> str:
    return type(error).__name__[:64]


def record(error: BaseException) -> None:
    """Latch native return off after a failed native read; never raises."""
    try:
        now = time.time()
        with _LOCK:
            first = not _STATE["latched"]
            if first:
                _STATE.update(latched=True, reason=_reason(error), latched_unix_s=now)
            _STATE["fault_count"] += 1
            _STATE["last_fault_unix_s"] = now
        if first:
            safe_call(log.warning,
                "native RDMA latent return failed on the driver (%s); native return is "
                "latched off for this ComfyUI process, so the next setup returns latents "
                "by actor messaging. Restart ComfyUI after fixing the fault to re-arm "
                "native return (%s)", _reason(error), TROUBLESHOOTING)
    except BaseException:
        pass


def latched() -> bool:
    with _LOCK:
        return bool(_STATE["latched"])


def native_enabled(configured: bool) -> bool:
    """What the next setup requests: the configured flag, unless the latch is set."""
    with _LOCK:
        return bool(configured) and not _STATE["latched"]


def snapshot() -> dict[str, Any]:
    """Telemetry view; the latch clears only with the driver process."""
    with _LOCK:
        return {"schema": 1, **_STATE, "clears_on": "driver_restart"}


def latched_error(error: BaseException) -> NativeReturnLatchedError:
    """The typed render error, chained by the caller from the native failure.

    The original text rides along, so the supervision classifier's text check
    still sees it. An ActorError's text is left out: it expands the remote
    traceback, which mesh_helpers also refuses to stringify.
    """
    hidden = any(cls.__name__ == "ActorError" for cls in type(error).__mro__)
    detail = "" if hidden else f": {failure_text(error)[:500]}"
    return NativeReturnLatchedError(
        f"native RDMA latent return failed on the driver ({_reason(error)}{detail}). "
        "The driver latched native return off, so the next setup returns latents by "
        "actor messaging. This render's lease stays abandoned, and a started read keeps its "
        f"destination until ComfyUI restarts, which also clears the latch ({TROUBLESHOOTING}).")
