"""Bake a LoRA stack in place where its classic bake does not fit, when the identity gate allows it.

With ``lora_low_rss`` off the store bakes a stack ComfyUI's classic way: the
slot is dropped, base and stack load fresh from the host, and ComfyUI keeps a
host backup. On unified memory that peak is the stock price plus the classic
bake's own term (``capacity_classic_bake``), and some loads the box can run do
not fit it. The default's own path costs far less: bake on the base the slot
already holds (``store_bake.lazy_swap_transition``), with nothing copied to
the host. ``take`` runs that path for one stack change when all of this holds:

- ``lora_low_rss`` is off and the request carries a stack;
- the classic price, credited the held copy a drop would return, does not fit,
  and the in-place price does;
- the slot holds this base bare on stock residency, or holds nothing, in which
  case the base loads bare first at the ordinary stock price;
- the driver sent this render's permission: the identity gate holds an exact
  PASS for this combination's in-place bake
  (``nodes/in_place_bake_permission``). A memory rule never moves a
  combination the gate has not passed onto another bake path.

The slot it leaves is baked, with no backup and no un-bake record, so the
next stack change reloads from disk and no later change takes the restore
path. It is marked ``baked_in_place``, as the default's in-place bakes are
(``store_bake.lazy_swap_transition``), and ``must_reload`` keeps a request
with the switch off and no permission from reusing such a slot.

The permission is bound to the sample dispatch by ``permission`` and read by
``permitted``; every other load path (a loader node, a gate ceremony, a
requeue built without the field) reads no permission and keeps the classic
bake. ComfyUI and the store are imported only inside functions.
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


def must_reload(store: Any, stored: Any) -> bool:
    """Whether a request may not reuse this slot: it was baked in place, the switch is off, and no permission came."""
    return bool(getattr(stored, "baked_in_place", False)) and not store.lora_low_rss and not permitted()


def _is_bare_stock(stored: Any) -> bool:
    from . import store_residency

    return (not stored.lora_sig and not stored.base_baked and stored.unbake is None
            and not getattr(stored.base_patcher, "_dgxm_fsdp", False)
            and store_residency.describe(stored)["residency"] == "stock")


def eligibility(store: Any, stored: Any, held: bool, allowed: bool) -> str | bool:
    """``True`` when the store may bake in place here, else why not; "" when the classic bake is not in play."""
    if store.lora_low_rss or getattr(store, "slab_weights", False) is True:
        return ""
    if held and not _is_bare_stock(stored):
        return capacity_classic_bake.HOLDS_A_STACK
    return True if allowed else capacity_classic_bake.NO_GATE_PASS


def held_bytes(stored: Any) -> int:
    """What ComfyUI's ledger says this slot's base holds on the device; 0 when unknown."""
    value = getattr(getattr(getattr(stored, "base_patcher", None), "model", None), "model_loaded_weight_memory", 0)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def fresh_load_probe(store: Any, slot: str, base_key: tuple,
                     lora_stack: list | None) -> Callable[[str, dict], Any] | None:
    """The stock probe the Worker's own walls price a fresh load with, built before the slot is dropped.

    ``None`` on the default and with no stack: nothing is added to the stock
    price. A stack change on a held classic base has already read the stack
    (``store_bake.hot_swap_transition`` builds the patched clone first), so its
    bytes are in the reading. A fresh load never bakes in place, so the reason
    the fallback was not taken rides along for the refusal, and never ``True``.
    """
    if store.lora_low_rss or not lora_stack:
        return None
    from .capacity_quote import default_resolve_lora_path

    stored = store._stored(slot)
    held = stored is not None and stored.base_key == base_key
    why = eligibility(store, stored, held, permitted())
    return capacity_classic_bake.probe(
        lora_stack, default_resolve_lora_path, stack_read=bool(held and not stored.base_baked),
        in_place="that is not available either" if why is True else why)


def take(store: Any, slot: str, unet_name: str, options: dict | None, lora_stack: list | None,
         base_key: tuple, requested_identity: dict, logger: Any, request_artifact_identity: Callable[..., Any],
         load_bare: Callable[[], Any]) -> tuple[Any, str] | None:
    """Bake this stack in place if the classic price misses, the in-place one fits and the gate allows it.

    ``None`` sends the caller to the fresh load, whose own walls price the
    classic bake and refuse or admit it. ``load_bare`` loads the base with no
    stack into an empty slot. A bake that starts and fails has dropped the
    slot (``lazy_swap_transition`` does), and the fresh load then prices the
    classic bake against what that returned.
    """
    if store.lora_low_rss or not lora_stack:
        return None
    from .. import capacity_fit
    from . import store_bake
    from .capacity_quote import default_resolve_lora_path
    from .comfy_bridge import resolve_model_path
    from .store_detect import _to_comfy_model_options

    stored = store._stored(slot)
    held = stored is not None and stored.base_key == base_key
    why = eligibility(store, stored, held, permitted())
    if why is not True:
        return None
    path = str(resolve_model_path("diffusion_models", unet_name))
    returned = held_bytes(stored) if held else 0
    fit = capacity_classic_bake.priced(
        capacity_fit.stock_load_fit(path, _to_comfy_model_options(options)), path, lora_stack,
        default_resolve_lora_path, stack_read=held, held_base_bytes=returned, in_place=True,
        bare_load=not held)
    stored = None  # no pin across a load or a bake that may drop the slot
    if not getattr(fit, "in_place_bytes", 0):
        return None
    # The classic price here: the fresh load with its bake, less the copy a drop would return.
    classic_gib = capacity_fit.gib(capacity_fit.stock_required_bytes(fit.size_bytes, fit.cast_bytes)
                                   + fit.bake_bytes - returned)
    if not held:
        load_bare()
        bare = store._stored(slot)
        usable = bare is not None and bare.base_key[:1] == (unet_name,) and _is_bare_stock(bare)
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
        baked.unbake = None  # the next stack change reloads from disk; no later change restores
    baked = None
    safe_call(
        log.info,
        "LoRA bake: the classic bake of this stack needs %.1f GiB here and %s GiB is available, so the "
        "Worker baked it in place on the loaded model and kept no weight backup (lora_low_rss is off; "
        "the identity gate has passed this combination's in-place bake). A stack change on this model "
        "reloads it from disk", classic_gib, fit.avail_gib)
    return result


__all__ = ["eligibility", "fresh_load_probe", "held_bytes", "must_reload", "permission", "permitted", "take"]
