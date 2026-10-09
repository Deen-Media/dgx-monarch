"""Apply the shared slab load and LoRA-bake capacity checks.

All three slab residency choices and the rescue offer use the same load price.
``admit_bake`` then checks the exact live patch set before a slab-resident LoRA
bake, covering keys the header estimate could not map.

A slab capacity refusal is class C without a consent option: slab is the final
residency choice, so consent cannot make the load fit. The message still names
changes that would reduce the requirement (DESIGN.md §5.9). Only
``admit_bake`` reads live tensors; the other helpers use pricing data.
"""
from __future__ import annotations

from .. import capacity_fit
from ..capacity_fit import SlabFit
from ..gate_audit_evidence import measured_tag
from ..log import get_logger
from ..mesh_safety import StockLoadCapacityError
from ..refusal import RefusalClass, refusal
from ..transfer_utils import failure_summary, safe_call

log = get_logger(__name__)

GUARD = "slab_load_preflight"
_TROUBLESHOOTING = 53

_FREE_MEMORY = (
    "free unified memory on this host (drop caches, stop co-resident LLM "
    "servers, dgxm reap)"
)
_PRUNE = "use a pruned or quantized artifact"
# Suggest FSDP only when the caller can prove the family supports it. Some
# families refuse FSDP structurally, so a capacity remedy must not assume it.
# No current caller provides that proof or passes ``fsdp_lever``.
_FSDP_LEVER = (
    "or, where the family supports it, an fsdp preset on world 2, which shards "
    "the weights instead of placing them whole"
)


def _floor_phrase(fit: SlabFit, reserve_bytes: int | None) -> str:
    """Name the figure the wall charged, and where it came from.

    The default arm also names what the floor buys, so the figure does not read
    as a round number: ``capacity_floor.ABSOLUTE_HOST_FLOOR_BYTES`` is derived
    from ComfyUI's own rule for keeping a model whole.
    """
    reserve = int(reserve_bytes or 0)
    if fit.floor_bytes <= capacity_fit.ABSOLUTE_HOST_FLOOR_BYTES:
        # A reserve the absolute floor charged over is still a figure the
        # operator set, and a card that never mentions it reads as a host that
        # never read the setting. Either the arena cap clipped a large reserve
        # under the floor or the floor outranked a small one; both name it.
        clipped = ""
        if reserve and reserve != fit.floor_bytes:
            became = ("which the arena cap clipped under it"
                      if reserve > fit.floor_bytes else "which it outranks")
            clipped = (f", above the {capacity_fit.gib(reserve)} GiB this host "
                       f"reserves (uma_reserve_gb), {became}")
        return (f"plus a {fit.floor_gib} GiB floor for the rest of the box{clipped}, "
                "below which ComfyUI stops keeping a model whole and offloads "
                "part of the weights mid-render")
    if fit.floor_bytes == reserve:
        return (f"plus the {fit.floor_gib} GiB this host reserves (uma_reserve_gb), "
                "which is above the default floor")
    # The arena cap clipped the reserve: an operator who set the figure and
    # reads a smaller one back needs both numbers.
    return (f"plus a {fit.floor_gib} GiB floor, the "
            f"{capacity_fit.gib(reserve)} GiB this host reserves (uma_reserve_gb) "
            "capped by the arena a stock load would have retained")


def _stock_required_gib(fit: SlabFit) -> float:
    """What a stock load of the same file would need, from the same size.

    Arithmetic on the size this probe already read, never a second probe: the
    slab rungs return above the stock fit probe and that ordering does not move.
    It goes through ``capacity_fit.stock_required_bytes`` rather than the
    multiplier alone, so the card cannot quote a stock price the stock wall
    would not charge.
    """
    return capacity_fit.gib(capacity_fit.stock_required_bytes(fit.size_bytes))


def _shortfall_gib(fit: SlabFit) -> float:
    avail = fit.avail_bytes or 0
    return round((fit.required_bytes - avail) / (1 << 30), 1)


def _lora_phrase(fit: SlabFit) -> str:
    """The bake's own term, which follows the floor in the sentence, or nothing."""
    if not fit.lora_bytes:
        return ""
    return (f" and {fit.lora_gib} GiB for the LoRA stack's bake (the device-side "
            "duplicate comfy's merge-and-free leaves until the slab reabsorbs it)")


def _stock_comparison(fit: SlabFit) -> str:
    """A whole-file LoRA fallback (2x) plus its floor can price slab near or above stock's 2.1x."""
    stock_gib = _stock_required_gib(fit)
    if stock_gib >= fit.required_gib:
        return (
            f"Stock residency is not the smaller option here: it needs {stock_gib} "
            f"GiB for the same file, because a stock load also holds the host copy "
            f"it is placed from, about {capacity_fit.STOCK_PLACEMENT_RATIO} times the "
            "file above the same floor. ")
    return (f"Stock residency needs {stock_gib} GiB for the same file; the LoRA "
            "stack's bake is what pushes this slab price above it. ")


def card(fit: SlabFit, unet_name: str, host: str, *,
         reserve_bytes: int | None = None, fsdp_lever: bool = False) -> str:
    """Explain why this host cannot fit the slab load and what would fit.

    Compare the stock requirement too: switching slab off must not look like a
    remedy when it would require more memory.
    """
    options = [_FREE_MEMORY, _PRUNE]
    tail = (", " + _FSDP_LEVER) if fsdp_lever else ""
    return (
        f"slab residency cannot load {unet_name} on {host}: the zero-copy slab "
        "holds the checkpoint's own bytes in shared memory, so this load needs "
        f"{fit.required_gib} GiB (a {fit.size_gib} GiB file "
        f"{_floor_phrase(fit, reserve_bytes)}{_lora_phrase(fit)}) and only "
        f"{fit.avail_gib} GiB of unified memory is available, "
        f"{_shortfall_gib(fit)} GiB short. {_stock_comparison(fit)}"
        f"What would fit: {', '.join(options)}{tail}. This "
        "refusal is capacity-protective (class C, StockLoadCapacityError "
        "family); nothing was loaded and nothing was quarantined."
    )


def rescue_blocker(fit: SlabFit, *, fsdp_lever: bool = False) -> tuple[str, str]:
    """The why-and-what-would-fit pair the rescue seam prints when slab is short.

    The numbers are rendered into the ``why`` string so the blocker tuple keeps
    the shape the sweep matches, and the seam raises through the rescue guard:
    the primary refusal is the stock price.
    """
    short = _shortfall_gib(fit)
    tail = (", " + _FSDP_LEVER) if fsdp_lever else ""
    return (
        f"zero-copy slab residency does not fit either: it needs "
        f"{fit.required_gib} GiB (the file plus a {fit.floor_gib} GiB floor"
        f"{_lora_phrase(fit)}) against the same {fit.avail_gib} GiB, {short} GiB short",
        f"{short} GiB more free memory, or a pruned or quantized artifact{tail}",
    )


def admit(fit: SlabFit, unet_name: str, host: str, *,
          reserve_bytes: int | None = None, fsdp_lever: bool = False) -> None:
    """Admit a fitting or skipped slab probe; otherwise raise its capacity refusal.

    A skipped probe has no evidence of insufficient capacity and follows the
    same admission rule as ``store_residency.preload_capacity_check``. Both slab
    load checks share this function, guard, and error text.
    """
    if not fit.applies or fit.fits:
        return
    raise StockLoadCapacityError(
        refusal(RefusalClass.CAPACITY,
                card(fit, unet_name, host, reserve_bytes=reserve_bytes,
                     fsdp_lever=fsdp_lever),
                guard=GUARD, waivable=False, troubleshooting=_TROUBLESHOOTING)
        + _measured_tail(fit))


def _measured_tail(fit: SlabFit) -> str:
    """The machine tail, which must never suppress the refusal it decorates."""
    try:
        return "\n" + measured_tag(fit.measured())
    except (TypeError, ValueError) as exc:
        safe_call(log.warning,
                  "slab capacity refusal could not carry its measured numbers (%s)",
                  failure_summary(exc))
        return ""


BAKE_GUARD = "slab_lora_bake_preflight"
_BAKE_TROUBLESHOOTING = 103


def _patched_live_bytes(active) -> int:
    """Each patched key's own live byte count: exact, no mapping needed."""
    from .unbake import live_tensor

    total = 0
    for key in active.patches:
        tensor = live_tensor(active.model, key)
        total += tensor.numel() * tensor.element_size()
    return total


def admit_bake(active, unet_name: str, host: str, floor_bytes: int | None = None) -> None:
    """The exact backstop before a slab-resident bake, ``_build_active``'s own
    patch set against the floor the slab price charged (``capacity_fit.slab_floor_bytes``,
    so a configured ``uma_reserve_gb`` holds here too), one moment before
    ``_merge_and_free`` allocates anything."""
    from .. import mesh_safety

    avail = mesh_safety.mem_available_bytes()
    if avail is None:
        return
    try:
        patched = _patched_live_bytes(active)
    except (AttributeError, TypeError) as exc:
        # No claim either way: the header estimate already priced the ladder.
        safe_call(log.debug, "bake backstop could not read the patch set (%s)",
                  failure_summary(exc))
        return
    floor = max(int(floor_bytes or 0), capacity_fit.ABSOLUTE_HOST_FLOOR_BYTES)
    required = patched + floor
    if required <= avail:
        return
    raise StockLoadCapacityError(refusal(
        RefusalClass.CAPACITY,
        f"the LoRA bake for {unet_name} cannot proceed on {host}: comfy's "
        "merge-and-free leaves a device-side duplicate of every patched key "
        f"until the slab reabsorbs it, and the exact patched set needs "
        f"{capacity_fit.gib(required)} GiB (a {capacity_fit.gib(patched)} GiB "
        f"duplicate plus the {capacity_fit.gib(floor)} "
        f"GiB floor the slab price charged) against {capacity_fit.gib(avail)} GiB of unified memory, "
        f"{capacity_fit.gib(required - avail)} GiB short. What would fit: free "
        "unified memory on this host (drop caches, stop co-resident LLM servers, "
        "dgxm reap), a smaller LoRA stack, or slab_weights=off, which bakes through "
        "the stock arena instead. This refusal is capacity-protective (class C, "
        "StockLoadCapacityError family); the base weights were loaded, the bake "
        "did not run, and nothing was quarantined.",
        guard=BAKE_GUARD, waivable=False, troubleshooting=_BAKE_TROUBLESHOOTING))
