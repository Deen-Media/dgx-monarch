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
from . import actor_footprint, capacity_quote, store_fsdp
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


def _cast_bytes(spec: Any) -> int:
    """The least a load cast to another weight dtype holds, or nothing.

    A credit and a clear's gain must not exceed what the drop returns, so this
    takes the cast estimate's lower bound (``cast_resident_bytes(lower=True)``):
    every float tensor at the smaller of its stored and cast size, which never
    exceeds the file or the copy, and nothing where the header cannot be read.
    The price's own figure is the upper bound, which ComfyUI can undercut: it
    builds fp32 auxiliaries at the cast's width on a copying load.
    """
    from .. import capacity_fit
    from .model_store import _to_comfy_model_options, resolve_model_path

    try:
        dtype = _to_comfy_model_options(spec.get("options")).get("dtype")
        if dtype is None:
            return _spec_bytes(spec)
        path = str(resolve_model_path("diffusion_models", str(spec["unet_name"])))
        return capacity_fit.cast_resident_bytes(path, dtype, lower=True)
    except Exception:
        return 0


def _resident_bytes(worker: Any, stored: Any, name: str) -> int:
    """What a clear returns for this resident's GPU copy.

    An FSDP shard holds one rank's share, so it is credited nothing and the
    card counts the arena alone, which only ever withholds a card. A load cast
    to another weight dtype holds the copy its cast built (``_cast_bytes``);
    any other stock resident holds about its file.
    """
    if getattr(stored, "fsdp_checkpoint_pin", None) is not None:
        return 0
    try:
        if store_fsdp.active(worker):
            return 0
        base_key: tuple[Any, ...] = tuple(getattr(stored, "base_key", ()) or ())
        options = dict(base_key[1]) if len(base_key) > 1 else {}
    except Exception:
        return 0
    spec = {"unet_name": name, "options": options}
    return _cast_bytes(spec) if options.get("weight_dtype") else _spec_bytes(spec)


def _backup_bytes(worker: Any, store: Any, stored: Any, slot: str) -> int:
    """The host bytes of ComfyUI's backup of this slot's held stack, or 0.

    ComfyUI keeps each weight a stack patches, as it was, on the host
    (``ModelPatcher.patch_weight_to_device``); every clone of the base shares that
    dict, and an in-place change puts those weights back from it and places again
    only what it lacks (2026-11-05: the actor grew 0.8 GiB on a strength change).
    The sum of the tensors' bytes is a lower bound of the memory they hold. It is
    reported only for the state that was measured: classic mode, a stock, uncast,
    full-precision resident placed whole on the GPU, the other slot empty (the
    gates ``store_bake.copies_off_the_gpu`` reads), and a held stack whose every
    patched weight the backup holds. Anything else, or a read that races the
    store, is 0, which credits nothing.
    """
    from . import store_bake, store_residency
    from .store_detect import _is_quantized_kind

    try:
        base, active = stored.base_patcher, stored.active_patcher
        backup, patches, model = base.backup, active.patches, base.model
        other = store.uncond if slot == capacity_quote.SLOT_COND else store.current
        base_key = tuple(stored.base_key or ())
        if (store.lora_low_rss or other is not None or not stored.lora_sig
                or not isinstance(backup, dict) or not isinstance(patches, dict) or not patches
                or store_residency.describe(stored)["residency"] != "stock"
                or getattr(base, "_dgxm_fsdp", False) or store_fsdp.active(worker)
                or (len(base_key) > 1 and dict(base_key[1]).get("weight_dtype"))
                or _is_quantized_kind(stored.quant_kind) or getattr(model, "model_lowvram", False)
                or store_bake._placed_bytes(model) <= 0 or not set(patches) <= set(backup)):
            return 0
        total = 0
        for entry in list(backup.values()):
            weight: Any = getattr(entry, "weight", None)
            if getattr(getattr(weight, "device", None), "type", None) == "cpu":
                total += int(weight.numel()) * int(weight.element_size())
        return total
    except Exception:
        return 0


def _resident_facts(worker: Any, store: Any, keys: dict[str, str | None] | None = None,
                    backups: dict[str, int] | None = None) -> dict[str, Any]:
    """What a clear of this actor would return, for the old-model card.

    The name comes from the store itself, never parsed back out of the
    snapshot's repr; the size is the GPU copy the resident holds
    (``_resident_bytes``). The driver decides which residency's bytes a clear
    returns (``old_model.rank_gain``), so every held slot is reported. ``keys``
    receives the request key of each resident read, in the snapshot's form, so
    the caller can bind these figures to the snapshot it prices against;
    ``backups`` receives each slot's ``_backup_bytes`` in the same pass.
    """
    names: dict[str, str | None] = {}
    sizes: dict[str, int] = {}
    for slot, stored in ((capacity_quote.SLOT_COND, getattr(store, "current", None)),
                         (capacity_quote.SLOT_UNCOND, getattr(store, "uncond", None))):
        if keys is not None:
            key = getattr(stored, "request_key", None)
            keys[slot] = None if key is None else repr(key)
        base_key = getattr(stored, "base_key", ()) or ()
        name = str(base_key[0]) if base_key else None
        names[slot] = name
        sizes[slot] = _resident_bytes(worker, stored, name) if name else 0
        if backups is not None:
            backups[slot] = _backup_bytes(worker, store, stored, slot) if name else 0
    return {"resident_names": names, "resident_bytes": sizes,
            "arena_bytes": actor_footprint.arena_bytes()}


def _with_resident_bytes(snapshot: Any, facts: dict[str, Any], keys: dict[str, str | None],
                         backups: dict[str, int] | None = None) -> dict:
    """The store snapshot with each held slot's GPU copy and name beside its residency.

    The swap credit reads them (``capacity_lora_bake.credit_stock_swap``): a load
    that replaces a stock resident is priced after the store drops that copy,
    and its refusal names the model dropped. The LoRA credit also reads the
    slot's backup (``capacity_lora_bake.held_backup_credit``). This endpoint runs
    outside the GPU lock, so a slot can change between the facts and the
    snapshot. A slot whose snapshot entry does not carry the request key the
    facts were read from gets none of them, and its facts are cleared too: no
    credit and no old-model figures for a model the snapshot does not describe.
    """
    out = dict(snapshot) if isinstance(snapshot, dict) else {}
    for slot in capacity_quote.SLOTS:
        entry = out.get(slot)
        if not isinstance(entry, dict) or keys.get(slot) is None or keys[slot] != entry.get("request_key"):
            facts["resident_bytes"][slot], facts["resident_names"][slot] = 0, None
            continue
        out[slot] = {**entry, "resident_bytes": int(facts["resident_bytes"].get(slot) or 0),
                     "resident_name": facts["resident_names"].get(slot) or "",
                     "backup_bytes": int((backups or {}).get(slot) or 0)}
    return out


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
    keys: dict[str, str | None] = {}
    backups: dict[str, int] = {}
    facts = _resident_facts(worker, store, keys, backups)
    snapshot = _with_resident_bytes(store.snapshot(), facts, keys, backups)
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
            **facts,
        ).to_row())
    return envelope
