"""One image a VAE batch: a batch rule that does not read memory.

NOT INSTALLED. Nothing calls ``install`` yet; it waits on a decision (the
design note of 2026-11-11, D2) and on a hardware timing of its cost.

ComfyUI's ``VAE.encode`` and ``VAE.decode`` take ``free memory / estimate``
items in a batch. A VAE's result for an item depends on the batch it shared:
on the GPU, six views encoded 6, 5+1, 4+2, 3+3, 2+2+2 or one at a time gave
six different latents, 56 to 64 percent of the values apart by one BF16 step
(the V1 probe of 2026-11-11), and even on CPU the split moves float32 bits.
So the same graph gives different conditioning as memory moves. A corrected
reading (``driver_uma_memory``) makes the full batch the rule at sizes the
box prices under what it has; at 2048 a view the estimate for six views is
most of a Spark and the split still follows the session.

Only one split never depends on memory: one item. Here an image VAE's encode
or decode of N items runs as N calls of ComfyUI's own function on one item
each, joined in order. Each call's batch count is 1 whatever memory reads, so
the result is one fixed tensor for a given input. A video VAE is left alone:
its frames are one item along time, not a batch.

What it costs is speed where a batch of N beats N batches of one. What it
changes once: today's full-batch bits.

To settle before anyone installs it: ``vae_decode_route`` also wraps
``VAE.decode`` (Worker-assisted decode, armed by the Init node, so after node
import). Installed first, this wrapper sits inside that one and each item goes
through the route's own decode; ``uninstall`` here restores ``__wrapped__``
and would drop a wrapper installed over it, so it is for tests, where nothing
else is installed.
"""
from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from typing import Any

_KEY = "_dgxm_item_batches"
_INSTALL_LOCK = threading.Lock()


def takes_items(vae: Any, tensor: Any) -> bool:
    """Whether ``tensor`` is a batch of more than one image (or image latent) for this VAE."""
    if getattr(vae, "latent_dim", 2) != 2 or getattr(vae, "extra_1d_channel", None) is not None:
        return False
    shape = getattr(tensor, "shape", None)
    return shape is not None and len(shape) >= 4 and int(shape[0]) > 1


def by_item(original: Callable[..., Any]) -> Callable[..., Any]:
    """``original`` (``VAE.encode`` or ``VAE.decode``) run on one item at a time for an image VAE."""

    @functools.wraps(original)
    def one_item_a_batch(self: Any, tensor: Any, *args: Any, **kwargs: Any) -> Any:
        if not takes_items(self, tensor):
            return original(self, tensor, *args, **kwargs)
        import torch

        return torch.cat([original(self, tensor[index:index + 1], *args, **kwargs)
                          for index in range(int(tensor.shape[0]))], dim=0)

    setattr(one_item_a_batch, _KEY, True)
    return one_item_a_batch


def install() -> None:
    """Wrap ``comfy.sd.VAE.encode`` and ``decode`` once. Not called anywhere yet (see above)."""
    import comfy.sd

    with _INSTALL_LOCK:
        for entry in ("encode", "decode"):
            current = getattr(comfy.sd.VAE, entry)
            if not getattr(current, _KEY, False):
                setattr(comfy.sd.VAE, entry, by_item(current))


def uninstall() -> None:
    """Undo ``install`` (tests)."""
    import comfy.sd

    with _INSTALL_LOCK:
        for entry in ("encode", "decode"):
            current = getattr(comfy.sd.VAE, entry)
            if getattr(current, _KEY, False):
                setattr(comfy.sd.VAE, entry, current.__wrapped__)


__all__ = ["by_item", "install", "takes_items", "uninstall"]
