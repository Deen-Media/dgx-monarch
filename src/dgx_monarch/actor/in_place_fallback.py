"""Bake a LoRA stack in place where its classic bake does not fit, when the identity gate allows it.

With ``lora_low_rss`` off the store bakes a stack ComfyUI's classic way: the
slot is dropped, base and stack load fresh from the host, and ComfyUI keeps a
host backup. On unified memory that peak is the stock price plus the classic
bake's own term (``capacity_classic_bake``), and some loads the box can run do
not fit it. The default's own path costs far less: bake on the base the slot
already holds (``store_bake.lazy_swap_transition``), with nothing copied to
the host. ``take`` runs that path for one stack change when all of this holds:

- ``lora_low_rss`` is off and the request carries a stack;
- the driver sent this render's permission: the identity gate holds an exact
  PASS for this combination's in-place bake
  (``nodes/in_place_bake_permission``). A memory rule never moves a
  combination the gate has not passed onto another bake path;
- the classic price, credited what dropping the slot's resident returns, does
  not fit, and the in-place price does.

Where the slot holds this base bare on stock residency, the bake runs on it.
Where it holds anything else (nothing, another model, this base with a stack
of either kind) the slot is dropped, the base loads bare at the ordinary stock
price, and the bake runs on that: a stack is never baked on top of another.

``ModelStore.ensure`` calls ``take`` before the classic hot swap and before
any drop, so the quote (``worker_capacity``), the store and the fresh load's
walls price one stack change by ``route``'s arguments and pick the same bake.

The slot it leaves is baked, with no backup and no un-bake record, so the
next stack change reloads the base. It is marked ``baked_in_place``, as the
default's in-place bakes are (``store_bake.lazy_swap_transition``), and
``must_reload`` keeps a request with the switch off and no permission from
reusing such a slot.

The permission is bound to the sample dispatch by ``permission`` and read by
``permitted``; every other load path (a loader node, the scheduler node, a
gate ceremony, a requeue built without the field) reads no permission and
keeps the classic bake. ComfyUI and the store are imported only inside
functions.
"""
from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Callable, Iterator
from typing import Any

from .. import capacity_classic_bake
from ..log import get_logger
from ..transfer_utils import safe_call

log = get_logger(__name__)

_PERMITTED: contextvars.ContextVar[bool] = contextvars.ContextVar("dgxm_in_place_bake_proven", default=False)

# The fresh load's walls, after ``take`` held the permission and did not bake.
TRIED = ("baking it in place was tried first and did not fit, or its bake failed (the Worker's log has "
         "the figures)")


@contextlib.contextmanager
def permission(request: Any) -> Iterator[None]:
    """Bind this sample request's permission for the loads made inside the block."""
    granted = isinstance(request, dict) and request.get(capacity_classic_bake.IN_PLACE_BAKE_PROVEN) is True
    token = _PERMITTED.set(bool(granted))
    try:
        yield
    finally:
        _PERMITTED.reset(token)


def permitted() -> bool:
    return _PERMITTED.get() is True


def must_reload(store: Any, stored: Any, allowed: bool | None = None) -> bool:
    """Whether a request may not reuse this slot: it was baked in place, the switch is off, and no permission came."""
    allowed = permitted() if allowed is None else allowed
    return bool(getattr(stored, "baked_in_place", False)) and not store.lora_low_rss and not allowed


def _is_stock(stored: Any) -> bool:
    from . import store_residency

    return (not getattr(stored.base_patcher, "_dgxm_fsdp", False)
            and store_residency.describe(stored)["residency"] == "stock")


def is_bare_stock(stored: Any) -> bool:
    """Whether this slot's base carries no stack by the store's record and by ComfyUI's.

    ComfyUI's backup must be empty too: a classic slot swapped to the bare
    base keeps the old stack's originals there, and the old stack on the
    device, until a sampler load runs.
    """
    return (not stored.lora_sig and not stored.base_baked and stored.unbake is None
            and not getattr(stored.base_patcher, "backup", None) and _is_stock(stored))


def held_bytes(stored: Any) -> int:
    """What dropping this slot's stock resident returns, by ComfyUI's ledger; 0 when unknown or not stock."""
    if stored is None or not _is_stock(stored):
        return 0
    value = getattr(getattr(stored.base_patcher, "model", None), "model_loaded_weight_memory", 0)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def route(store: Any, stored: Any, held: bool, allowed: bool) -> dict[str, Any] | None:
    """``capacity_classic_bake.priced``'s arguments for one stack change; ``None`` when no classic bake is in play.

    One function for the quote, the store and the walls, so they cannot price
    one change two ways. ``held`` says the slot holds this request's base.
    """
    if store.lora_low_rss or getattr(store, "slab_weights", False) is True:
        return None
    return {"held_bytes": held_bytes(stored), "bare_load": not (held and is_bare_stock(stored)),
            "in_place": True if allowed else capacity_classic_bake.GATE_HAS_NOT_PASSED}


def quote_snapshot(store: Any, stored: Any, snapshot: dict, slot: str, allowed: bool) -> dict:
    """The snapshot the quote prices against: a slot ``ensure`` will not reuse is not read as a reuse."""
    entry = snapshot.get(slot)
    if stored is None or not isinstance(entry, dict) or not must_reload(store, stored, allowed):
        return snapshot
    return {**snapshot, slot: {**entry, "request_key": None, "swaps_in_place": False}}


def fresh_load_probe(store: Any, slot: str, lora_stack: list | None) -> Callable[[str, dict], Any] | None:
    """The stock probe the Worker's own walls price a fresh load with.

    ``None`` on the default and with no stack: nothing is added to the stock
    price. A fresh load never bakes in place and has already been handed the
    slot to drop, so nothing is credited and the reason the in-place bake was
    not taken rides along for the refusal.
    """
    if store.lora_low_rss or not lora_stack:
        return None
    from .capacity_quote import default_resolve_lora_path

    return capacity_classic_bake.probe(
        lora_stack, default_resolve_lora_path,
        in_place=TRIED if permitted() else capacity_classic_bake.GATE_HAS_NOT_PASSED)


def take(store: Any, slot: str, unet_name: str, options: dict | None, lora_stack: list | None,
         base_key: tuple, requested_identity: dict, logger: Any, request_artifact_identity: Callable[..., Any],
         load_bare: Callable[[], Any]) -> tuple[Any, str] | None:
    """Bake this stack in place if the classic price misses, the in-place one fits and the gate allows it.

    ``None`` sends the caller on to the classic hot swap or the fresh load,
    whose own walls price the classic bake and refuse or admit it.
    ``load_bare`` loads the base with no stack into an empty slot. A bake
    that starts and fails has dropped the slot (``lazy_swap_transition``
    does), and the fresh load then prices the classic bake against what that
    returned.
    """
    if store.lora_low_rss or not lora_stack or not permitted():
        return None
    from .. import capacity_fit
    from . import store_bake
    from .capacity_quote import default_resolve_lora_path
    from .comfy_bridge import resolve_model_path
    from .store_detect import _to_comfy_model_options

    stored = store._stored(slot)
    args = route(store, stored, stored is not None and stored.base_key == base_key, True)
    if args is None:
        return None
    occupied = stored is not None
    stored = None  # no pin across a drop, a load or a bake
    path = str(resolve_model_path("diffusion_models", unet_name))
    fit = capacity_classic_bake.priced(
        capacity_fit.stock_load_fit(path, _to_comfy_model_options(options)), path, lora_stack,
        default_resolve_lora_path, **args)
    if not getattr(fit, "in_place_bytes", 0):
        note = str(getattr(fit, "in_place_note", "") or "")
        if note:  # the classic price missed and so did the in-place one: the walls refuse, this says why
            safe_call(log.info, "LoRA bake: the classic bake of this stack does not fit here%s", note)
        return None
    classic_gib = capacity_fit.gib(capacity_fit.stock_required_bytes(fit.size_bytes, fit.cast_bytes)
                                   + fit.bake_bytes - args["held_bytes"])
    if args["bare_load"]:
        if occupied:
            store._drop(slot)  # never bake on top of a stack; its copy comes back first
        load_bare()
        bare = store._stored(slot)
        usable = bare is not None and bare.base_key[:1] == (unet_name,) and is_bare_stock(bare)
        # The loaded base's own key: the kind detected at load can differ from the one the
        # request's options implied, and the baked slot must be keyed as its base is.
        base_key = bare.base_key if usable else base_key
        bare = None
        if not usable:
            return None
    result = store_bake.lazy_swap_transition(
        store, slot, unet_name, lora_stack, base_key, requested_identity,
        request_artifact_identity=request_artifact_identity, logger=logger)
    if result is None:
        return None
    baked = store._stored(slot)
    if baked is not None:
        baked.unbake = None  # the next stack change reloads the base; no later change restores
    baked = None
    safe_call(
        log.info,
        "LoRA bake: the classic bake of this stack needs %.1f GiB here and %s GiB is available, so the "
        "Worker baked it in place on the loaded model and kept no weight backup (lora_low_rss is off; "
        "the identity gate has passed this combination's in-place bake). A stack change on this model "
        "reloads it from disk", classic_gib, fit.avail_gib)
    return result


__all__ = ["TRIED", "fresh_load_probe", "held_bytes", "is_bare_stock", "must_reload", "permission",
           "permitted", "quote_snapshot", "route", "take"]
