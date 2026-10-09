"""Exchange readiness after dm-cfg2's per-rank checkpoint loads.

Rank 0 loads the conditional checkpoint and rank 1 the unconditional one.
A corrupt file, identity mismatch, or load failure can affect only one rank.
Exchange readiness before the first block-loop collective so every rank
refuses instead of leaving a healthy peer waiting for the group timeout.

``rank_readiness.exchange_ready`` uses ``partial_load_guard.all_ranks_true``
(a MIN all-reduce over the default group). This module supplies the dm-cfg2
message and exception type; other non-FSDP multi-rank topologies use the
same exchange through ``rank_readiness``.
"""
from __future__ import annotations

from typing import NoReturn

from ..log import get_logger
from ..refusal import RefusalClass, refusal
from . import rank_readiness

log = get_logger(__name__)


class DualModelLoadDivergenceError(RuntimeError):
    """Typed cross-rank refusal: one rank could not load or verify its dm-cfg2 half.

    A truncated header, a per-box memory shortfall, a per-box filesystem error,
    or a rank-specific identity mismatch can stop one rank alone. As with
    ``partial_load_guard.PartialLoadDivergenceError``, the driver imports this on
    unpickle, so the message is built at the raise, never at import.
    """


# Same wording rule as partial_load_guard.REFUSAL_TEXT: no host or rank inside,
# each worker journal names its own box's cause. Names topology uly2 as the
# alternative: every rank there loads the same checkpoints, and the
# rank_readiness exchange refuses a one-rank failure on every rank instead of
# hanging the healthy one.
DIVERGENCE_TEXT = (
    "a peer rank could not load or verify its half of the dual-model cfg2 render. "
    "Each rank keeps a different checkpoint under this topology, so a truncated or "
    "mismatched file, a filesystem error, or a per-box memory shortfall can stop "
    "one box while this one loaded cleanly. That box's worker journal names the "
    "cause. Nothing was sampled. Fix the cause on that box and run again, or run "
    "the same graph on topology uly2 instead, where every rank loads the same "
    "checkpoints and a load failure surfaces on both ranks at once."
)


def ready_or_raise() -> None:
    """Release the fleet when every rank loaded its dm-cfg2 half, else refuse.

    Called by a rank whose own per-rank load and identity check passed. It puts a
    ready flag on the wire; when a peer rank failed, every rank raises here rather
    than this one blocking on the first collective until the NCCL timeout.
    """
    if rank_readiness.exchange_ready(True):
        return
    raise DualModelLoadDivergenceError(refusal(RefusalClass.PHYSICS, DIVERGENCE_TEXT))


def not_ready(local_error: Exception) -> NoReturn:
    """Tell every peer this rank failed its dm-cfg2 load, then re-raise the cause.

    The failed rank still joins the readiness exchange (with a not-ready flag) so
    a healthy peer is released to refuse instead of hanging, then re-raises its
    own error, which names the real fault in this box's journal.
    """
    log.warning(
        "dm-cfg2 residency load failed on this rank; refusing on every rank: %r",
        local_error)
    # Best effort, for the reason rank_readiness.not_ready spells out: the box
    # that just ran short is the one whose exchange can fail to allocate, and
    # the typed cause has to outlive the flag.
    try:
        rank_readiness.exchange_ready(False)
    except Exception as exc:
        log.warning(
            "readiness flag not crossed (%r); the peer will wait on the group "
            "timeout", exc)
    raise local_error
