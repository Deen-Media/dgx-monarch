"""What ComfyUI hands a model-level attention op the pack installs.

ComfyUI c194dd00c lets a model file name the attention kernel for each block,
so models pass ``preferred_attention=self.comfy_attention`` to their attention
call. ComfyUI 8d534945e and later also wrap q, k and v in
``AttentionTensorContainer``, a single-owner box the backend empties with
``take()``. Comfy's ``wrap_attn`` (comfy/ldm/modules/attention.py) pops the
keyword and takes the containers before it calls
``transformer_options["optimized_attention_override"]``, so the pack's
overrides never see either. An op the pack installs on the model itself, such
as Anima's ``attn_op``, is called before ``wrap_attn`` and receives both.

Under sequence parallelism the Init node's ``attention`` widget and the
sequence-parallel dispatcher choose the kernel. The per-block preference is
logged at debug level once per call site and otherwise ignored, so a
checkpoint that pins a kernel per block renders with the widget's kernel
where stock ComfyUI on one GPU would use the pinned one (docs/ADAPTERS.md).

Such an op accepts a closed set of keywords. Nothing upstream filters what
comfy hands it, and the dispatcher drops an unknown keyword for an ordinary
kernel, so any other keyword refuses here (``refuse_unadopted_keywords``).
"""
from __future__ import annotations

import threading
from collections.abc import Iterable

import torch

from ..log import get_logger
from ..refusal import RefusalClass, refusal
from .base import UnsupportedModelError

log = get_logger(__name__)

_noted_sites: set[str] = set()
_noted_lock = threading.Lock()


def _is_container(value) -> bool:
    """Comfy's container, matched by its ``take`` method.

    A tensor has a ``take`` method of its own, so tensors are ruled out first.
    Duck typing keeps this module free of a comfy import.
    """
    return not isinstance(value, torch.Tensor) and callable(getattr(value, "take", None))


def refuse_unadopted_keywords(site: str, keywords: Iterable[str]) -> None:
    """Refuse a keyword ComfyUI passes that the op at ``site`` does not apply.

    Dropping it could change attention without a word, so the render stops.
    Comfy hands every rank the same call, so every rank refuses before the
    op's first all-to-all, and the class tag lets the driver consume the
    sample lease rather than abandon it (nodes/pending.typed_worker_refusal).
    """
    names = sorted(keywords)
    if not names:
        return
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"{site}: ComfyUI passed {', '.join(map(repr, names))} to the attention "
        "op dgx-monarch installs, and this release does not apply that argument "
        "under sequence parallelism. It is a ComfyUI change the pack has not "
        "adopted yet, and dropping it could change the output. Run this "
        "workflow on topology 'single' (mode=local with gpus_per_host=1) for a "
        "stock one-GPU run, or on a ComfyUI build this release supports.",
        troubleshooting=116,
    ))


def take_qkv(q, k, v):
    """Return plain q, k and v, taking comfy's containers once as wrap_attn does.

    Plain tensors pass through untouched. A mix of containers and tensors is
    refused, as comfy refuses it.
    """
    boxed = (_is_container(q), _is_container(k), _is_container(v))
    if not any(boxed):
        return q, k, v
    if not all(boxed):
        raise TypeError("q, k, and v must all be attention tensor containers")
    return q.take(), k.take(), v.take()


def note_preferred_attention(site: str, preferred) -> None:
    """Log once per call site that a per-block kernel preference was ignored."""
    if preferred is None:
        return
    with _noted_lock:
        if site in _noted_sites:
            return
        _noted_sites.add(site)
    pinned = getattr(preferred, "function", None) is not None
    log.debug(
        "%s: ComfyUI passed a per-block attention preference (config=%r, "
        "kernel pinned=%s); the Init attention widget and the sequence-parallel "
        "dispatcher choose the kernel, so it is ignored",
        site, getattr(preferred, "config", None), pinned)
