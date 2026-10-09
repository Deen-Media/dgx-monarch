"""Price the FSDP residency rung without loading model state.

Like ``capacity_quote``, this module imports no torch or ComfyUI and mutates
nothing. ``capacity_quote._fsdp_rung`` imports it locally to avoid a cycle:
this module imports the row type and rung vocabulary from ``capacity_quote``.
"""
from __future__ import annotations

from typing import Any

from .. import capacity_fit
from ..capacity_fit import preflight_wall_measured
from ..mesh_safety import StockLoadCapacityError
from ..refusal import untagged
from ..transfer_utils import failure_summary
from .capacity_quote import (
    RUNG_STOCK_FITS,
    VERDICT_FITS,
    VERDICT_NO_CLAIM,
    VERDICT_REFUSE,
    RankPrice,
    _shortfall,
)


def fsdp_rung(path: str, unet_name: str, model_options: dict, world: int | None,
              size: int, charged: int, context: dict[str, Any]) -> RankPrice:
    """The one rung whose check raises instead of returning.

    The adapter module has no top-level torch import and the check reads only
    /proc/meminfo, ``getsize`` and ``gpu_is_integrated``, so the purity rule
    holds. An import that fails anyway is ``no_claim``: a rank that cannot
    answer must not veto one that can. The adapter prices one checkpoint against
    the whole box, so the bytes an earlier slot took are subtracted here rather
    than passed into a guard both walls share.
    """
    common = dict(context, rung=RUNG_STOCK_FITS, checkpoint_bytes=size)
    try:
        from ..adapters import fsdp as adapter
    except Exception as exc:
        return RankPrice(VERDICT_NO_CLAIM, reason=(
            "the FSDP adapter could not be imported on this rank "
            f"({failure_summary(exc)})"), **common)
    # The adapter's names, not this module's, so two meminfo reads in one quote
    # agree. Its check runs first, on every device.
    try:
        from ..adapters.detect import sniff_fsdp_launch_quant_proof

        proof = sniff_fsdp_launch_quant_proof(path)
        checkpoint_kind = proof.quant_kind
    except Exception:
        checkpoint_kind = None
    avail = adapter.mem_available_bytes()
    required = capacity_fit.fsdp_required_bytes(size, world, checkpoint_kind)
    try:
        adapter.fsdp_load_capacity_check(
            path, unet_name, model_options, world=world,
            checkpoint_kind=checkpoint_kind)
    except StockLoadCapacityError as exc:
        # The adapter owns the sentence; the tag comes off (a mid-card tag names no boundary).
        return RankPrice(VERDICT_REFUSE, priced=True, reason=untagged(str(exc)), **common,
                         measured=(preflight_wall_measured(size, avail - charged, required, False)
                                   if avail is not None else None))
    if model_options.get("dtype") is not None or not adapter.gpu_is_integrated():
        return RankPrice(VERDICT_NO_CLAIM, reason=(
            "this load casts its dtype or does not run on an integrated GPU, so file size "
            "does not bound the FSDP shard build"), **common)
    if avail is None:
        return RankPrice(VERDICT_NO_CLAIM,
                         reason="kernel MemAvailable is unreadable here", **common)
    avail -= charged
    if required > avail:
        # The adapter priced this file against the whole box; an earlier slot
        # has already spent part of it and the wall it will meet knows that.
        return RankPrice(
            VERDICT_REFUSE, priced=True, **common,
            measured=preflight_wall_measured(size, avail, required, False),
            reason="FSDP launch: the shard build does not fit" + _shortfall(required, avail))
    return RankPrice(
        VERDICT_FITS, priced=True, **common,
        measured=preflight_wall_measured(size, avail, required, True),
        reason="FSDP launch: the shard-build transient check admits this load")
