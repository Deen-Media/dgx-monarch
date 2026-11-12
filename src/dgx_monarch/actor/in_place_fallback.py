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

``ModelStore.ensure`` calls ``take`` where a fresh load would follow: after
the classic hot swap has declined or was not tried, and before any drop. A
change the hot swap makes in place (the held stack's backup covers it, or
the other slot holds a model) is not a fresh load and takes neither the term
nor the fallback. The quote (``worker_capacity``) reads the same state
through ``loads_fresh``, and prices by ``route``'s arguments, so the quote,
the store and the fresh load's walls pick the same bake.

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
from typing import Any, TypeGuard

from .. import capacity_classic_bake
from ..log import get_logger
from ..transfer_utils import safe_call

log = get_logger(__name__)

_PERMITTED: contextvars.ContextVar[bool] = contextvars.ContextVar("dgxm_in_place_bake_proven", default=False)

# The fresh load's walls, when the permission came and no in-place bake was made. ``take``
# logs its own figures when it priced one; a slot ``ensure`` set aside is never offered.
TRIED = ("baking it in place was not done: it did not fit, its bake failed, or this slot was set "
         "aside (the Worker's log says which)")
# The quote's clause for a slot ``ensure`` will set aside (``checkpoint_moved``): no in-place bake is offered.
SET_ASIDE = ("baking it in place is not offered: the checkpoint file has changed on disk since this slot "
             "loaded it, so base and stack load fresh")


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


def holds_base(stored: Any, base_key: Any) -> TypeGuard[Any]:
    """Whether this slot holds the request's base: the one comparison ``ModelStore.ensure`` makes.

    ``ensure`` calls this at each of its branches, ``take`` calls it, and the
    quote calls it with a key from the store's own builder. It compares the
    whole base key, the file name and the loader options together. A request
    that keeps the file and changes an option (the weight dtype, for one)
    asks for another base, and the store drops the slot and loads fresh.
    """
    return stored is not None and bool(base_key) and getattr(stored, "base_key", None) == base_key


def checkpoint_moved(stored: Any, checkpoint_path: str) -> bool:
    """Whether the checkpoint file differs on disk from the fingerprint this slot recorded at its load.

    ``ensure`` sets such a slot aside before any swap: it does not hot-swap
    it, never offers it to ``take``, and loads base and stack fresh. The quote
    asks here, so it prices that load and no in-place bake (``SET_ASIDE``). A
    slot that recorded no fingerprint gives nothing to compare; a file that
    cannot be fingerprinted reads as moved, which prices the dearer load.
    """
    identity = getattr(stored, "artifact_identity", None)
    artifacts = identity.get("artifacts") if isinstance(identity, dict) else None
    first = artifacts[0] if isinstance(artifacts, list) and artifacts else None
    recorded = first.get("signature") if isinstance(first, dict) else None
    if recorded is None:
        return False
    from ..gate_ledger import artifact_signature

    try:
        return bool(artifact_signature(checkpoint_path) != recorded)
    except Exception:
        return True


def _held_files_unchanged(stored: Any, names: set, resolve_lora_path: Callable[[str], str]) -> bool:
    """No LoRA file of these names differs on disk from the fingerprint the slot recorded at its load.

    A file rewritten under its own name may patch other weights than the held
    stack did, and the store builds the stack from the file as it is now. A
    slot that recorded no fingerprint for a name gives nothing to compare and
    reads as unchanged; a file that cannot be fingerprinted reads as changed.
    """
    identity = getattr(stored, "artifact_identity", None)
    artifacts = identity.get("artifacts") if isinstance(identity, dict) else None
    recorded = {item.get("name"): item.get("signature") for item in artifacts or ()
                if isinstance(item, dict) and item.get("kind") == "loras"}
    if not recorded:
        return True
    from ..gate_ledger import artifact_signature

    try:
        return all(name not in recorded or artifact_signature(resolve_lora_path(str(name))) == recorded[name]
                   for name in names)
    except Exception:
        return False


def backup_covers(stored: Any, checkpoint_path: str, lora_stack: list | None,
                  resolve_lora_path: Callable[[str], str]) -> bool:
    """Whether ComfyUI's backup on this slot holds every weight ``lora_stack`` patches.

    This is the fact the store's classic hot swap acts on
    (``store_bake.copies_off_the_gpu``: no patched weight outside the backup
    means the change is made in place), read from the slot itself and so true
    for every format, without building the stack. The backup's own keys are
    the source. A stack of the held stack's own files (a strength change, a
    reorder, a LoRA removed) patches what the held stack patched, as long as
    those files are the ones the slot loaded (``_held_files_unchanged``). Any
    other stack is mapped from the checkpoint's and the LoRAs' headers and each
    weight looked for among the backup's keys; where a header cannot be read
    the answer is no, which prices the dearer fresh load.
    """
    backup = getattr(getattr(stored, "base_patcher", None), "backup", None)
    if not isinstance(backup, dict) or not lora_stack:
        return True  # the store reads a backup it cannot inspect, or an empty stack, as nothing to copy
    if not backup:
        return False
    held_patches = getattr(getattr(stored, "active_patcher", None), "patches", None)
    held_names = {entry[0] for entry in getattr(stored, "lora_sig", ()) or () if entry}
    new_names = {entry.get("name") for entry in lora_stack if isinstance(entry, dict)}
    if (new_names <= held_names and isinstance(held_patches, dict) and set(held_patches) <= set(backup)
            and _held_files_unchanged(stored, new_names, resolve_lora_path)):
        return True
    from ..capacity_lora_bake import _prefix_variants
    from ..safetensors_header import read_safetensors_header

    try:
        tensors = read_safetensors_header(checkpoint_path).tensors
    except Exception:
        return False
    backed = {(variant, "." + key.rpartition(".")[2])
              for key in backup for variant in _prefix_variants(key.rpartition(".")[0])}
    patched = capacity_classic_bake.patched_weights(tensors, lora_stack, resolve_lora_path)
    return all(any((variant, "." + key.rpartition(".")[2]) in backed
                   for variant in _prefix_variants(key.rpartition(".")[0])) for key in patched)


def loads_fresh(stored: Any, held: bool, other_held: bool, covered: bool) -> bool:
    """Whether ``ModelStore.ensure`` ends this classic stack change in a fresh load.

    The classic bake's term, and the in-place bake in its place, belong to a
    change the store makes by dropping the slot and loading base and stack
    fresh. A change the classic hot swap makes in place has neither: ComfyUI
    re-applies the stack from its own backup, which grew the actor by 0.8 GiB
    on the 2026-11-05 hardware leg. ``ensure`` gets this from its own order
    (the hot swap first, then ``take``); the quote asks here, with what its
    snapshot knows: the hot swap is tried on a held base that is not baked,
    and it stays in place when the other slot holds a model, when the base is
    not a stock resident ComfyUI has placed, or when ComfyUI's backup covers
    the new stack (``covered``, from ``backup_covers``).
    """
    if not held or stored.base_baked:
        return True
    if other_held or held_bytes(stored) <= 0:
        return False
    return not covered


def route(store: Any, stored: Any, held: bool, allowed: bool | str) -> dict[str, Any] | None:
    """``capacity_classic_bake.priced``'s arguments for one stack change; ``None`` when no classic bake is in play.

    One function for the quote, the store and the walls, so they cannot price
    one change two ways. ``held`` says the slot holds this request's base
    (``holds_base``: the whole base key, never the file name alone).
    ``allowed`` is ``True`` when the store may bake in place, ``False`` when
    the gate's permission did not come, or the clause that says why the store
    will not offer it (``SET_ASIDE``).
    """
    if store.lora_low_rss or getattr(store, "slab_weights", False) is True:
        return None
    return {"held_bytes": held_bytes(stored), "bare_load": not (held and is_bare_stock(stored)),
            "in_place": True if allowed is True else (allowed or capacity_classic_bake.GATE_HAS_NOT_PASSED)}


def stack_files_moved(stored: Any, lora_stack: list | None, resolve_lora_path: Callable[[str], str]) -> bool:
    """Whether a LoRA file of this stack differs on disk from the fingerprint the slot recorded for it."""
    names = {entry.get("name") for entry in lora_stack or () if isinstance(entry, dict)}
    return not _held_files_unchanged(stored, names, resolve_lora_path)


def quote_snapshot(store: Any, stored: Any, snapshot: dict, slot: str, allowed: bool,
                   request_key: str = "", moved: bool = False) -> dict:
    """The snapshot the quote prices against: a slot ``ensure`` will not reuse is not read as a reuse.

    Two cases. A slot baked in place that this request may not reuse
    (``must_reload``). And a request whose key is the slot's own while a file
    it names has changed on disk since the slot loaded it (``moved``: the
    checkpoint, or a LoRA of the stack): ``ensure`` reuses only when the
    recorded fingerprints match too, so it reloads, with the classic bake
    where the stack loads fresh. Read as a reuse, that request was quoted
    nothing.
    """
    entry = snapshot.get(slot)
    if stored is None or not isinstance(entry, dict):
        return snapshot
    stale = moved and bool(request_key) and entry.get("request_key") == request_key
    if not stale and not must_reload(store, stored, allowed):
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

    ``None`` sends the caller on to the fresh load, whose own walls price the
    classic bake and refuse or admit it.
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
    args = route(store, stored, holds_base(stored, base_key), True)
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


__all__ = ["SET_ASIDE", "TRIED", "backup_covers", "checkpoint_moved", "fresh_load_probe", "held_bytes", "holds_base",
           "is_bare_stock", "loads_fresh", "must_reload", "permission", "permitted", "quote_snapshot", "route",
           "stack_files_moved", "take"]
