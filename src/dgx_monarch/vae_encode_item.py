"""The driver encodes a batch of images one image a call, so the result does not read memory.

ComfyUI's ``VAE.encode`` takes ``free memory / estimate`` images in a batch,
and a VAE's result for an image depends on the batch it shared. On the GPU,
six views encoded 6, 5+1, 4+2, 3+3, 2+2+2 or one at a time gave six different
latents, 56 to 64 percent of the values apart by one BF16 step (2026-11-11),
and even on CPU the split moves float32 bits. So the same graph gave other
conditioning as memory moved. The corrected reading (``driver_uma_memory``)
steadies the count where a whole batch fits; a batch priced at most of the box
still split by what the session had.

Only one split never depends on memory: one image. Here an image VAE's encode
of N images runs as N calls of ComfyUI's own ``VAE.encode`` on one image each,
joined in order. Each call's batch count is 1 whatever memory reads, so the
result is one fixed tensor for a given input, and it is stock's own: image i's
latent is what stock gives for image i alone.

On a DGX Spark with the one VAE measured (FLUX.2's, 2026-11-11, 4 and 6 images
at 1024 and 2048) this was never slower than the whole batch (0.86 to 1.0 of
its time) and peaked at 2.28 GiB where six 2048 images in one batch took
12.41. Other image VAEs are exact by construction and were not timed.

What it changes against a box without this pack: a stock ComfyUI with room for
the whole batch encodes each image beside the others and gets other bits (one
BF16 step on most values). A single image matches either way. A stock render
and a DGX Monarch render in this driver share the driver's conditioning, so
they are compared on the same bits.

Scope. On by default at node import where the reading is: Linux, with the
driver's CUDA device integrated (it shares the host's memory).
``DGXM_DISABLE_VAE_ENCODE_ITEM=1`` on the driver turns it off; the reading's
own switch does not. Encode only: ``VAE.decode`` is not wrapped. With the VAE
measured above a decode gave the same result in any batch, and one image a
call read up to 10 percent slower; no other VAE's decode was measured.
``VAE.encode_tiled`` is not wrapped either. A video VAE is left whole (its frames are one item along
time), and so is anything that is not an image VAE.

The image test is stock's own. ``VAE.encode`` treats the first dimension as
separate items exactly when ``latent_dim`` is 2, or 3 with ``not_video`` set
(``comfy/sd.py``); the audio VAEs with a 2-D latent carry
``extra_1d_channel`` and take a waveform, not a 4-D image batch. If ComfyUI
stops setting any of the three attributes, the call is left whole.

It steps aside, once and for the life of the process, when ComfyUI does not
behave as this version expects: a call with arguments it does not know, a
batch it cannot slice, or results it cannot join. Stock's whole-batch encode
answers then and from then on, one WARNING says why, and ``dgxm doctor`` shows
it. An error ComfyUI itself raises while encoding an image (out of memory, a
cancelled prompt) is passed on as it is; nothing is run again after it.
"""
from __future__ import annotations

import functools
import os
import sys
import threading
from collections.abc import Callable
from typing import Any

from . import driver_uma_memory
from .log import get_logger

log = get_logger(__name__)

DISABLE_ENV = "DGXM_DISABLE_VAE_ENCODE_ITEM"

# What this process did, for ``/dgxm/telemetry`` (``driver_rules``) and doctor's row.
INSTALLED = "installed"
OFF = "off"
STEPPED_ASIDE = "stepped_aside"
NOT_LINUX = "not_linux"
NOT_INTEGRATED = "not_integrated"
NO_COMFY = "no_comfy"
PROBE_FAILED = "probe_failed"
UNSET = "unset"

_KEY = "_dgxm_encode_item"
_ABSENT = object()
_INSTALL_LOCK = threading.Lock()
_STATE = {"state": UNSET}


def state() -> str:
    """What the rule is in this process: one of the names above."""
    return _STATE["state"]


def _set(value: str) -> str:
    _STATE["state"] = value
    return value


def takes_items(vae: Any, tensor: Any) -> bool:
    """Whether ``tensor`` is a batch of more than one image for an image VAE.

    True only for a 4-D batch whose VAE says, by the attributes ``VAE.encode``
    itself branches on, that its first dimension holds separate images. A VAE
    that lacks one of them is not one this rule knows, and is left whole.
    """
    shape = getattr(tensor, "shape", None)
    if shape is None or len(shape) != 4 or int(shape[0]) <= 1:
        return False
    if getattr(vae, "extra_1d_channel", _ABSENT) is not None:
        return False  # an audio VAE, or a VAE without the attribute
    latent_dim = getattr(vae, "latent_dim", None)
    if latent_dim == 2:
        return True
    return latent_dim == 3 and getattr(vae, "not_video", None) is True


def _join(parts: list[Any]) -> Any:
    """The per-image results as one batch, or an error if they are not what one image's encode returns.

    ``torch.cat`` widens mixed dtypes without a word, so results that differ in dtype are an error here.
    """
    import torch

    first = parts[0]
    for part in parts:
        if not isinstance(part, torch.Tensor):
            raise TypeError(f"an encode of one image returned {type(part).__name__}, not a tensor")
        if part.shape[0] != 1 or part.shape[1:] != first.shape[1:] or part.dtype != first.dtype:
            raise ValueError(f"an encode of one image returned shape {tuple(part.shape)} {part.dtype}")
    return torch.cat(parts, dim=0)


def by_item(original: Callable[..., Any]) -> Callable[..., Any]:
    """``original`` (``VAE.encode``) run on one image at a time for an image VAE."""
    aside: list[str] = []

    def step_aside(why: str) -> None:
        aside.append(why)
        _set(STEPPED_ASIDE)
        log.warning("one image a VAE encode stepped aside (%s); ComfyUI's own batches answer from here "
                    "on in this process, so a batch of images can encode to other bits as memory moves", why)

    @functools.wraps(original)
    def one_image_a_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        if aside:
            return original(self, *args, **kwargs)
        if len(args) == 1 and not kwargs:
            pixels = args[0]
        elif not args and set(kwargs) == {"pixel_samples"}:
            pixels = kwargs["pixel_samples"]
        else:
            step_aside("VAE.encode was called with arguments this pack version does not know")
            return original(self, *args, **kwargs)
        try:
            count = int(pixels.shape[0]) if takes_items(self, pixels) else 0
            items = [pixels[index:index + 1] for index in range(count)]
        except Exception as exc:
            step_aside(f"the batch could not be read: {type(exc).__name__}: {exc}")
            return original(self, *args, **kwargs)
        if not items:
            return original(self, *args, **kwargs)
        # Stock's own call on one image. What it raises is stock's error and is passed on.
        parts = [original(self, item) for item in items]
        try:
            return _join(parts)
        except Exception as exc:
            step_aside(f"the per-image results could not be joined: {type(exc).__name__}: {exc}")
            return original(self, *args, **kwargs)

    setattr(one_image_a_call, _KEY, True)
    return one_image_a_call


def install_driver() -> str:
    """The ComfyUI process's hook, at node import. Never raises.

    Returns, and keeps for ``state``, what it did: ``installed``, ``off`` (the
    environment switch), ``not_linux``, ``not_integrated`` (a discrete GPU or
    the CPU), ``no_comfy`` or ``probe_failed``. The gate is the reading's
    (``driver_uma_memory``), probed here so that the reading's own switch does
    not turn this rule off.
    """
    if os.environ.get(DISABLE_ENV) == "1":
        log.info("%s=1: ComfyUI's own VAE encode batches stay in this process", DISABLE_ENV)
        return _set(OFF)
    if not sys.platform.startswith("linux"):
        return _set(NOT_LINUX)
    try:
        model_management, torch = driver_uma_memory.comfy_modules()
    except Exception:
        return _set(NO_COMFY)
    try:
        total = driver_uma_memory.integrated_total_bytes(torch, model_management.get_torch_device())
    except Exception as exc:
        log.warning("could not tell whether this GPU shares the host's memory (%s: %s); ComfyUI's own "
                    "VAE encode batches stay", type(exc).__name__, exc)
        return _set(PROBE_FAILED)
    if total is None:
        return _set(NOT_INTEGRATED)
    try:
        install()
    except Exception as exc:
        log.warning("one image a VAE encode could not be installed, so ComfyUI's own batches stay "
                    "(%s: %s)", type(exc).__name__, exc)
        return _set(NO_COMFY)
    log.info("unified memory: a VAE encode of several images runs one image a call in this process, so "
             "its result does not depend on free memory (%s=1 turns this off)", DISABLE_ENV)
    return _set(INSTALLED)


def _vae_class() -> Any:
    """ComfyUI's ``VAE`` class, only if this process already imported ``comfy.sd``.

    It imports nothing itself: importing ComfyUI here would pick its device
    before the process that owns it does. The ComfyUI process has imported
    ``comfy.sd`` long before it loads a node pack.
    """
    module = sys.modules.get("comfy.sd")
    if module is None:
        raise ImportError("comfy.sd is not imported in this process")
    return module.VAE


def install() -> None:
    """Wrap ``comfy.sd.VAE.encode`` once. ``VAE.decode`` is left as it is."""
    vae = _vae_class()
    with _INSTALL_LOCK:
        current = vae.encode
        if not getattr(current, _KEY, False):
            vae.encode = by_item(current)


def uninstall() -> None:
    """Undo ``install`` (tests)."""
    vae = _vae_class()
    with _INSTALL_LOCK:
        current = vae.encode
        if getattr(current, _KEY, False):
            vae.encode = current.__wrapped__
    _set(UNSET)


__all__ = ["DISABLE_ENV", "INSTALLED", "NOT_INTEGRATED", "NOT_LINUX", "NO_COMFY", "OFF", "PROBE_FAILED",
           "STEPPED_ASIDE", "UNSET", "by_item", "install", "install_driver", "state", "takes_items",
           "uninstall"]
