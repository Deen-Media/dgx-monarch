"""Slab residency through a lazy LoRA swap: one copy of every weight.

``store_bake.lazy_swap`` restores and re-bakes the keys a stack change
touches. ComfyUI bakes a quantized weight into a new tensor, so on a slab
resident each such key is returned to its own slab region as the swap finishes
with it (``SwapSettle``), and a stack whose newly patched key could not go
back is sent to a fresh load before any write (``slab_misfits``).
"""
from __future__ import annotations

from ..log import get_logger
from ..transfer_utils import safe_call

log = get_logger(__name__)

DIFFUSION_PREFIX = "diffusion_model."


class SwapSettle:
    """Return each key a lazy swap touched to where the resident keeps its weights.

    A restore rebuilds a quantized wrapper from a fresh copy and ComfyUI's bake
    replaces it out of place, so on a slab resident every patched quantized
    key would otherwise sit twice in unified memory: once in its idle slab
    region and once in cudaMalloc, while every capacity figure counts the slab
    once. Settling the key the moment the swap is done with it (restored, then
    baked when the new stack patches it) puts it back in its own region, so at
    most the key in hand has a second copy. Plain keys restore and bake in
    place and stay where they are.
    """

    def __init__(self, stored):
        self.slab = getattr(stored, "slab", None)
        self.model = stored.base_patcher.model
        self.quantized = False
        self.moved = 0
        self.keys = 0

    def __call__(self, key: str) -> None:
        from .unbake import _is_quantized, live_tensor

        try:
            self.quantized = self.quantized or _is_quantized(live_tensor(self.model, key))
        except (AttributeError, TypeError):
            pass
        if self.slab is None or not key.startswith(DIFFUSION_PREFIX):
            return
        self.keys += 1
        self.moved += self.slab.reabsorb_key(self.model.diffusion_model, key[len(DIFFUSION_PREFIX):])

    def finish(self) -> None:
        """Give the freed copies back: the allocator otherwise keeps them as its own cache."""
        if self.slab is None and not self.quantized:
            return
        import comfy.model_management as mm

        mm.soft_empty_cache()
        if self.slab is not None:
            safe_call(
                log.info,
                "slab residency after the swap: %d key(s) checked, %.2f GiB returned to the slab%s",
                self.keys, self.moved / (1 << 30),
                f", quant strays {self.slab.stray_bytes / (1 << 30):.2f} GiB stay in cudaMalloc"
                if self.slab.stray_bytes else "")


def slab_misfits(stored, active) -> list[str]:
    """Keys this stack newly patches whose baked payload no slab region can take back.

    Such a key would stay in cudaMalloc beside its idle region, a second copy no
    price saw made, so the swap does not run: the store loads fresh, and the
    load's own walls price the bake whole (``store_slab_admit.admit_bake``). A
    key that already holds bytes outside the slab is not new: the fresh load
    priced that copy, and each swap replaces it with one of the same size.
    """
    slab = getattr(stored, "slab", None)
    patches = getattr(active, "patches", None)
    dm = getattr(getattr(getattr(stored, "base_patcher", None), "model", None), "diffusion_model", None)
    if (slab is None or dm is None or not isinstance(patches, dict)
            or not callable(getattr(slab, "baked_payload_fits", None))):
        return []
    misfits = []
    for key in patches:
        if not key.startswith(DIFFUSION_PREFIX):
            continue
        name = key[len(DIFFUSION_PREFIX):]
        if not slab.strays(name) and not slab.baked_payload_fits(dm, name):
            misfits.append(key)
    return misfits
