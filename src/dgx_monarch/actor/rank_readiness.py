"""Exchange post-load readiness before non-FSDP sampling collectives.

Capacity agreement precedes allocation, so memory changes, file errors, and
identity failures can still stop one rank during loading. Without another
exchange, healthy ranks would enter ``partial_load_guard.check`` and wait for
the failed peer until the process-group timeout.

Each rank joins ``partial_load_guard.all_ranks_true`` with its load outcome.
Failed ranks then re-raise their local cause; healthy ranks proceed only when
all are ready, otherwise raising ``PeerLoadFailureError``. This covers dm-cfg2
through ``dm_cfg2_residency`` and every other non-FSDP multi-rank topology.
Even ranks loading identical checkpoints can fail independently.

The topology predicate must be raise-free and rank-identical. No earlier
collective may run between loading and this exchange. FSDP is excluded because
its load can contain collectives; a readiness exchange could then meet a
different collective on a peer.

The exchange inherits the default group's timeout (``nccl_timeout_s``, 600 s
by default). It promptly releases peers only when every rank can join. A dead
rank is handled by driver supervision or the sample stall budget; a rank that
cannot allocate the readiness flag preserves its original error while peers
may wait for the group timeout. See docs/TROUBLESHOOTING.md #86.
"""
from __future__ import annotations

from typing import Any, NoReturn

from ..log import get_logger
from ..refusal import RefusalClass, refusal
from . import partial_load_guard

log = get_logger(__name__)


def exchange_ready(local_ready: bool) -> bool:
    """Cross this rank's post-load readiness, and answer whether every rank is.

    One primitive for every topology's exchange, so the two shapes cannot drift
    into two different collectives. Off-group (world 1, no process group) it
    answers the local flag and issues nothing.
    """
    return partial_load_guard.all_ranks_true(local_ready)


def whole_model_topology(worker: Any) -> bool:
    """Whether every rank of this fleet must hold the whole model.

    Multi-rank and not FSDP is the whole rule. uly2, ring2, dp2 and cfg2 all put
    the entire checkpoint on every rank, the same premise ``partial_load_guard``
    refuses on when one rank holds it only partially. The module docstring says
    why this gate must be raise-free and rank-identical, and why FSDP is out.
    """
    topo = getattr(worker, "topology", {}) or {}
    if int(getattr(worker, "world", 0) or 0) < 2:
        return False
    return not bool(topo.get("fsdp"))


class PeerLoadFailureError(RuntimeError):
    """Typed cross-rank refusal: a peer rank did not become ready after its load.

    Raised on the healthy rank. The rank that failed raises its own error,
    whose class and text name the real cause on that box. As with
    ``partial_load_guard.PartialLoadDivergenceError``, the driver imports this
    module to unpickle a worker's exception, so the message is built at the
    raise, never at import.
    """


# Same wording rule as partial_load_guard.REFUSAL_TEXT: no host and no rank
# inside, because MIN carries agreement and not identity, and each worker
# journal names its own box's cause. Class P on this rank: it cannot know the
# peer's cause, and no consent it could be offered would change the answer
# here; the failing box keeps its own class and its own card.
PEER_LOAD_TEXT = (
    "a peer rank could not load or verify the model for this render. Every rank "
    "of this topology holds the whole model, so a per-box memory shortfall, a "
    "filesystem error, or a capacity wall that fired on one box after the fleet "
    "had already agreed can stop one rank while this one loaded cleanly. That "
    "box's worker journal names the cause and prints its own refusal. Nothing "
    "was sampled and no rank went on. Fix the cause on that box and run again, "
    "or run the same graph on a single box instead."
)


def ready_or_raise() -> None:
    """Release the fleet when every rank loaded, else refuse on every rank.

    Called by a rank whose own load and identity check passed, after the load
    and before the render's first collective.
    """
    # Logged before the exchange on every rank, as partial_load_guard.check
    # does: MIN erases which box was short, and without this line a fleet that
    # crossed the exchange looks the same in the journal as one where it never ran.
    log.info("rank readiness: this rank loaded; crossing the readiness flag")
    if exchange_ready(True):
        return
    raise PeerLoadFailureError(refusal(RefusalClass.PHYSICS, PEER_LOAD_TEXT))


def not_ready(local_error: Exception) -> NoReturn:
    """Tell every peer this rank failed, then re-raise the cause.

    The failed rank still joins the readiness exchange with a not-ready flag, so
    a healthy peer refuses at once instead of waiting out the group timeout; the
    re-raised error names the real fault in this box's journal.
    """
    log.warning(
        "load failed on this rank; refusing on every rank: %r", local_error)
    # The flag is best effort; the raise after it is not. The fault this seam
    # was built for is a memory wall on this box, the state most likely to
    # break the exchange's own one-element tensor or its NCCL buffer. Letting
    # that error through would hand the operator a CUDA or NCCL error in place
    # of the typed cause, and the peer, which gets no flag either way, would
    # wait out the group timeout regardless. So the local cause always wins.
    try:
        exchange_ready(False)
    except Exception as exc:
        log.warning(
            "readiness flag not crossed (%r); the peer will wait on the group "
            "timeout", exc)
    raise local_error
