"""One image a VAE batch: a batch rule that does not read memory.

OFF BY DEFAULT (``ON_BY_DEFAULT`` below). It goes on by default only if
hardware shows three things: it costs under 5 percent of the VAE's own time,
one graph gives one conditioning hash at every size with it on, and its
result for N images equals N separate stock encodes of one image bit for bit.
Until then ``DGXM_VAE_ITEM_BATCHES=1`` on the driver turns it on for a trial,
and ``DGXM_DISABLE_VAE_ITEM_BATCHES=1`` turns it off whatever the default.

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
changes: a ComfyUI with the rule on gives each image the result of encoding
it alone. A stock ComfyUI with room for the whole batch gives each image the
result of encoding it beside the others, which differs (by one BF16 step on
most values in the probe above). So with the rule on, a batch of images does
not match the same batch on a box without this pack, in either direction;
both sides match for a single image.

``vae_decode_route`` also wraps ``VAE.decode`` (Worker-assisted decode, armed
by the Init node, so after node import). This wrapper is installed first and
sits inside that one, so each item goes through the route's own decode.
``uninstall`` here restores ``__wrapped__`` and would drop a wrapper installed
over it, so it is for tests, where nothing else is installed.
"""
from __future__ import annotations

import functools
import os
import threading
from collections.abc import Callable, Mapping
from typing import Any

from .log import get_logger

log = get_logger(__name__)

# The decision's one line: hardware decides whether this becomes True.
ON_BY_DEFAULT = False
ENABLE_ENV = "DGXM_VAE_ITEM_BATCHES"
DISABLE_ENV = "DGXM_DISABLE_VAE_ITEM_BATCHES"

_KEY = "_dgxm_item_batches"
_INSTALL_LOCK = threading.Lock()


def wanted(environ: Mapping[str, str] | None = None) -> bool:
    """Whether this process should run the rule: the off switch wins, then the trial switch, then the default."""
    env = os.environ if environ is None else environ
    if env.get(DISABLE_ENV) == "1":
        return False
    return ON_BY_DEFAULT or env.get(ENABLE_ENV) == "1"


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


def install_driver() -> str:
    """The ComfyUI process's hook, at node import. Never raises.

    Returns ``off`` (the default today, or the off switch), ``installed`` or
    ``no_comfy``.
    """
    if not wanted():
        return "off"
    try:
        install()
    except Exception as exc:
        log.warning("one image a VAE batch could not be installed, so ComfyUI's own batches stay "
                    "(%s: %s)", type(exc).__name__, exc)
        return "no_comfy"
    log.info("VAE encodes and decodes of image batches run one image a batch in this process, so a "
             "result does not depend on free memory (%s=1 turns this off)", DISABLE_ENV)
    return "installed"


def install() -> None:
    """Wrap ``comfy.sd.VAE.encode`` and ``decode`` once."""
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


__all__ = ["DISABLE_ENV", "ENABLE_ENV", "ON_BY_DEFAULT", "by_item", "install", "install_driver",
           "takes_items", "uninstall", "wanted"]
