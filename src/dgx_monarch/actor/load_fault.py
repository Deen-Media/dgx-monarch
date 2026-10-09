"""Inject a per-rank load failure for readiness acceptance tests.

The fault runs after cross-rank capacity agreement and local pricing, before
ComfyUI reads weights. This tests whether ``rank_readiness`` releases healthy
peers when one rank fails before the first collective. External file faults
usually fail during capacity agreement and cannot reliably target this window.

Both environment variables are required:

* ``DGXM_FAULT_LOAD_RANK=<rank>`` selects the failing rank.
* ``DGXM_ACCEPTANCE=1`` explicitly enables acceptance faults.

``arm`` reads them once per setup generation. Any nonempty fault setting logs
its outcome at WARNING, including ignored or unreadable settings. The default
is off. Changes after setup cannot affect an in-flight render.

The armed rank raises a typed class P refusal on every fresh load in that
generation. Reuse and swaps do not trigger it. Restart the worker between
acceptance runs.
"""
from __future__ import annotations

import os

from ..log import get_logger
from ..refusal import RefusalClass, refusal

log = get_logger(__name__)

RANK_ENV = "DGXM_FAULT_LOAD_RANK"
ACCEPTANCE_ENV = "DGXM_ACCEPTANCE"
TROUBLESHOOTING = 99

# Class P, no guard, never waivable, and it names what to do instead. It is
# neither a capacity answer nor a rule about evidence: this box was told to
# fail, so no consent would change the answer. The text says the fault was
# injected, because a refusal that hid that would send an operator hunting a
# load fault that does not exist.
FAULT_TEXT = (
    "acceptance fault injected on this rank. The debug knob "
    "DGXM_FAULT_LOAD_RANK names this rank, so this load refused on purpose, "
    "after every rank had agreed it could hold the model and before any weight "
    "was read. The model, the checkpoint and this machine are fine. "
    "Nothing was loaded, nothing was sampled, and the fleet is still up. The "
    "peer rank refuses too, with its own message about a peer that could not "
    "load. Unset DGXM_FAULT_LOAD_RANK and DGXM_ACCEPTANCE on this box's worker "
    "and restart it to render instead."
)


class InjectedLoadFaultError(RuntimeError):
    """The typed refusal the debug knob raises on the rank it names.

    Its own type and text, not the readiness exchange's
    ``PeerLoadFailureError``: that one is what a healthy rank raises about a
    peer, and in this box's journal it would name the wrong cause on the only
    box that knows the real one. As in ``rank_readiness``, the message is built
    at the raise and never at import, because the driver imports this module to
    unpickle a worker's exception.
    """


# Proc lifetime, and written only by ``arm`` at each setup generation. A
# render never writes it, so nothing in flight can arm or disarm the fault.
_ARMED: bool = False


def arm(rank: int) -> bool:
    """Read the knob for this setup generation and say whether this rank fails.

    Called once per worker setup, so the answer holds for the life of that
    generation. Every outcome but an unset or empty knob logs at WARNING,
    because a fault knob nobody sees in the journal gets left on.
    """
    global _ARMED
    _ARMED = False
    raw = (os.environ.get(RANK_ENV) or "").strip()
    if not raw:
        return False
    try:
        target = int(raw)
    except ValueError:
        log.warning(
            "%s=%r does not name a rank, so the debug load fault stays off",
            RANK_ENV, raw)
        return False
    if os.environ.get(ACCEPTANCE_ENV) != "1":
        log.warning(
            "%s=%d is set and %s is not 1, so the debug load fault stays off. "
            "This knob fails one rank on purpose for the acceptance leg that "
            "tests the post-load readiness exchange, and it is refused outside "
            "that leg. Set %s=1 on this worker as well if you are running that "
            "leg.",
            RANK_ENV, target, ACCEPTANCE_ENV, ACCEPTANCE_ENV)
        return False
    _ARMED = target == rank
    log.warning(
        "DEBUG LOAD FAULT ARMED: %s=%d with %s=1, so rank %d of this fleet "
        "refuses every fresh model load after the capacity agreement and "
        "before the weights are read. This worker is rank %d, so it %s. Unset "
        "both variables and restart this worker to render normally.",
        RANK_ENV, target, ACCEPTANCE_ENV, target, rank,
        "fails" if _ARMED else "loads as usual")
    return _ARMED


def check() -> None:
    """Refuse this rank's fresh load when the knob named it, else do nothing."""
    if not _ARMED:
        return
    log.warning("debug load fault: failing this load on purpose (%s)", RANK_ENV)
    raise InjectedLoadFaultError(refusal(
        RefusalClass.PHYSICS, FAULT_TEXT, troubleshooting=TROUBLESHOOTING))
