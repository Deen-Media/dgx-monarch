"""The one retry a render makes after every Worker loop restarted under it.

``restart_reattach.heal`` retires the stale fleet and attaches by the new ids,
or raises the refusal that names its remedy; ``attempt_after_restart`` then
prices the render again on the new mesh and retries once. The render's finish
record reads ``reattach_facts`` to say whether a re-attach happened inside it
and how long it took (nodes/render_submit.py).
"""
from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING

from .. import restart_reattach
from ..delivery_faults import WorkerDeliveryError
from . import render_preflight

if TYPE_CHECKING:
    from .common import ModelSpec
    from .pending import PendingRender

# Render lifetime: what the finish record says about a re-attach inside this
# render (restart_reattach.py). Set only around the one retry after it.
_REATTACH_FACTS: ContextVar[dict | None] = ContextVar("dgxm_reattach_facts", default=None)
# read-only after import: the same two facts for every other render.
NO_REATTACH = {"reattach": None, "reattach_s": None}


def reattach_facts() -> dict:
    """The finish record's re-attach facts for this render (a ContextVar read)."""
    return _REATTACH_FACTS.get() or NO_REATTACH


def attempt_after_restart(
        attempt: Callable[[], PendingRender], refused: object,
        exc: WorkerDeliveryError, model: ModelSpec, request: dict,
        latent: dict) -> PendingRender:
    """Attach again inside this render when every Worker loop restarted, then retry once.

    ``restart_reattach.heal`` retires the stale fleet and attaches by the new
    ids, or raises the refusal that names its remedy. The retry's ``ensure_live``
    then finds the new mesh. Its finish record carries the re-attach and its wall.
    """
    outcome = restart_reattach.heal(refused, exc)
    # As on every process-stop path: the new fleet holds none of the old weights.
    from . import recycle_drain

    recycle_drain.drop_residency_memos()
    # The caller's memory check ran before the restart showed, so it may have
    # credited the old fleet's weights as resident. Price again on the new mesh
    # before the retry loads anything: weights and activations that each fit
    # can still overrun together, and a refusal here is the one a first
    # attempt without the credit raises. The check reads only the latent and
    # the request's positive conditioning, which the first attempt left alone.
    if getattr(model, "mesh", None) is not None and outcome.handle is not None:
        model = render_preflight._model_with_bound_handle(model, outcome.handle)
    render_preflight.activation_footprint_preflight_for_request(model, request, latent)
    token = _REATTACH_FACTS.set(
        {"reattach": restart_reattach.EVENT, "reattach_s": round(outcome.wall_s, 2)})
    try:
        return restart_reattach.retry_after_heal(attempt, exc)
    finally:
        _REATTACH_FACTS.reset(token)
