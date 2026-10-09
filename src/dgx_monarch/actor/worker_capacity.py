"""Return per-rank capacity rows for the driver's load agreement.

The ``worker.capacity_quote`` endpoint delegates here without taking the GPU
lock, touching CUDA, or mutating the store. Read ComfyUI's pinned-staging flag
once and pass it to the ComfyUI-free ``capacity_quote.price``.

Every path returns a row, including unpriced and refused loads. The driver
raises fleet refusals from these rows; a worker exception would cross the actor
boundary as a traceback without the structured rank information it needs.
"""
from __future__ import annotations

import os
import socket
from dataclasses import replace
from typing import Any

from ..log import get_logger
from ..transfer_utils import failure_summary
from . import capacity_quote, store_fsdp
from .capacity_quote import RankPrice

log = get_logger(__name__)

_NOT_SET_UP = "this rank has not completed setup"


def _host() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "this host"


def _no_claim(slot: str, reason: str) -> RankPrice:
    return RankPrice(
        verdict=capacity_quote.VERDICT_NO_CLAIM,
        rung="", reason=reason, slot=slot or capacity_quote.SLOT_COND)


def _slot_of(spec: Any) -> str:
    slot = spec.get("slot") if isinstance(spec, dict) else None
    return slot if slot in capacity_quote.SLOTS else capacity_quote.SLOT_COND


def _charged_bytes(row: RankPrice, spec: Any = None) -> int:
    """What this slot takes out of the rank's budget for the next one.

    A reuse places nothing. Otherwise the slot spends the ``required_bytes`` of
    the block its own verdict was taken on, or the file's bytes when that block
    carries no price. The row picks the block: a slab row charged the stock
    block would also spend the stock placement peak above the file
    (``capacity_fit.STOCK_PLACEMENT_RATIO`` times the file), which a slab
    load never reaches, and refuse the next slot for it.

    A row that priced nothing carries no bytes either, so the spec's own file
    is read: a slot this rank could not price still fills the box, and charging
    it nothing admits a second checkpoint against the first one's memory.
    """
    if row.would_load == capacity_quote.WOULD_LOAD_REUSE:
        return 0
    block = row.slab_measured if row.use_slab else row.measured
    required = (block or {}).get("required_bytes")
    if isinstance(required, int):
        return required
    return int(row.checkpoint_bytes) or _spec_bytes(spec)


def _spec_bytes(spec: Any) -> int:
    """The size on disk of the checkpoint a spec names, or nothing."""
    from .model_store import resolve_model_path

    try:
        path = str(resolve_model_path("diffusion_models", str(spec["unet_name"])))
        return os.path.getsize(path) if os.path.exists(path) else 0
    except Exception:
        return 0


def _request_key(snapshot: dict, slot: str, spec: dict, unet_name: str) -> str:
    """The key ``ModelStore.ensure`` would build for this request.

    ``ensure`` reuses on the whole key and takes the quant kind from the
    resident whenever the name and the options match, so the slot supplies that
    term and no header is read here. A slot holding nothing needs no key, and a
    stack this helper cannot sign states none, which reads as a swap.
    """
    held = snapshot.get(slot)
    if not isinstance(held, dict):
        return ""
    from .model_store import lora_signature, normalize_options

    try:
        return repr((unet_name, normalize_options(spec.get("options")),
                     lora_signature(spec.get("loras")), held.get("quant")))
    except Exception:
        return ""


def _reserve_bytes(worker: Any) -> int:
    """The operator's own co-residency headroom, in bytes.

    ``driver_footprint`` already charges it, so a worker wall that ignores it
    admits on the rank what the driver refuses on the head. Unreadable reads
    as nothing rather than raising: this rank still owes an answer.
    """
    try:
        reserve = float(getattr(worker, "uma_reserve_gb", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0
    return max(int(reserve * (1 << 30)), 0)


def _my_slots(worker: Any, specs: list[dict]) -> list[dict]:
    """Under dm-cfg2 a rank holds one checkpoint, so it prices one.

    Charging both slots on both ranks would spend a budget no rank spends and
    refuse a shape that runs today. Outside that topology every rank holds
    every slot and the list is unchanged.
    """
    from .sample_protocol import _dual_model_cfg2_topology

    if len(specs) < 2 or not _dual_model_cfg2_topology(worker):
        return specs
    rank: Any
    try:
        from ..adapters.base import cfg_rank

        rank = cfg_rank()
    except Exception:
        rank = getattr(worker, "rank", None)
    if not isinstance(rank, int):
        return specs
    mine = capacity_quote.SLOT_COND if rank == 0 else capacity_quote.SLOT_UNCOND
    return [spec for spec in specs if _slot_of(spec) == mine] or specs


def _price(worker: Any, store: Any, spec: dict, *, snapshot: dict,
           pinned_staging: bool | None, rescue_consent: dict | None,
           charged: int) -> RankPrice:
    from .model_store import (
        SLAB_VOUCHED_FAMILIES,
        _slab_capable_path,
        _to_comfy_model_options,
        memoized_family,
        resolve_model_path,
    )

    slot = _slot_of(spec)
    unet_name = str(spec.get("unet_name") or "")
    options = spec.get("options")
    path = str(resolve_model_path("diffusion_models", unet_name))
    stored = store.uncond if slot == capacity_quote.SLOT_UNCOND else store.current
    # The same flag ``ensure`` reads before it drops the resident: a first stock
    # load that discovered a vouched family reloads into slab whatever the memo
    # says. It applies only to the checkpoint that set it.
    base_key = getattr(stored, "base_key", ()) or ()
    retry = bool(getattr(stored, "slab_auto_retry", False)
                 and base_key and base_key[0] == unet_name)
    return capacity_quote.price(
        path=path,
        unet_name=unet_name,
        model_options=_to_comfy_model_options(options),
        slab_weights=store.slab_weights,
        slab_capable_path=_slab_capable_path(path),
        lora_low_rss=store.lora_low_rss,
        reserve_bytes=_reserve_bytes(worker),
        fsdp_launch=store_fsdp.active(worker),
        blocked_reason=store.slab_blocked_reason,
        world=getattr(worker, "world", None),
        authoritative_slab_retry=retry,
        memoized_family=memoized_family,
        vouched_families=SLAB_VOUCHED_FAMILIES,
        rescue_consent=rescue_consent,
        family_override=getattr(store, "family_override", None),
        node_options=options,
        lora_stack=spec.get("loras"),
        slot=slot,
        pinned_staging=pinned_staging,
        store_snapshot=snapshot,
        budget_bytes_already_charged=charged,
        request_key=_request_key(snapshot, slot, spec, unet_name),
    )


def _pinned_staging() -> bool | None:
    """ComfyUI's pinned-staging flag, or None when it cannot be read."""
    try:
        from . import comfy_dynamic

        return bool(comfy_dynamic.pinned_staging_active())
    except Exception as exc:
        log.warning("capacity quote could not read ComfyUI's pinned-staging flag (%s)",
                    failure_summary(exc))
        return None


def quote(worker: Any, request: Any) -> dict:
    """Price this rank's share of one request and answer with rows.

    Every slot of a request is priced against one snapshot and one running
    budget, in load order. Priced apart, two checkpoints that each fit alone
    would both be admitted when the pair does not fit.
    """
    specs = request.get("specs") if isinstance(request, dict) else None
    specs = [spec for spec in specs if isinstance(spec, dict)] \
        if isinstance(specs, list) else []
    rows: list[dict[str, Any]] = []
    envelope: dict[str, Any] = {
        "host": _host(),
        "rank": getattr(worker, "rank", None),
        "world": getattr(worker, "world", None),
        "setup": getattr(worker, "_setup_key", None) is not None,
        "quotes": rows,
    }
    store = getattr(worker, "store", None)
    if not envelope["setup"] or store is None:
        rows.extend(_no_claim(_slot_of(spec), _NOT_SET_UP).to_row()
                    for spec in specs or [{}])
        return envelope
    consent = request.get("rescue_consent") if isinstance(request, dict) else None
    snapshot = store.snapshot()
    pinned = _pinned_staging()
    charged = 0
    for spec in _my_slots(worker, specs):
        try:
            row = _price(worker, store, spec, snapshot=snapshot,
                         pinned_staging=pinned,
                         rescue_consent=consent if isinstance(consent, dict) else None,
                         charged=charged)
        except Exception as exc:
            # No claim either way, which admits: a probe this rank could not run
            # is not evidence that the load does not fit here.
            log.warning("capacity quote made no claim for slot %s (%s)",
                        _slot_of(spec), failure_summary(exc))
            row = _no_claim(_slot_of(spec), failure_summary(exc))
        charged += _charged_bytes(row, spec)
        # ``shmem_bytes`` is the ladder's own read and is not touched here: one
        # /proc/meminfo parser, and one read per row.
        rows.append(replace(
            row,
            retained_failed_load_slabs=int(
                snapshot.get("retained_failed_load_slabs", 0) or 0),
            failed_load_cleanup_pending=bool(
                snapshot.get("failed_load_cleanup_pending", False)),
        ).to_row())
    return envelope
