"""Run the stock attention the pack leaves in place without a block's kernel pin.

ComfyUI lets a checkpoint pin an attention kernel per block (the comfy-kitchen
int8 kernel, or comfy-kitchen Sol with its tau), and models hand it to
``optimized_attention`` as ``preferred_attention``. Under sequence parallelism
the pack runs the sharded attention itself and leaves some stock attention in
place: cross-attention against the replicated text, and the refiners every
rank runs whole before the sharded block loop. Stock ``wrap_attn``
(comfy/ldm/modules/attention.py) would run a pinned kernel there under the
widget's stamp, and the pack would ask for no class-K Sol-Attn waiver.

``wrap_attn`` checks ``transformer_options["optimized_attention_override"]``
before it reads the preference, so the copy ``stock_attention_options`` builds
drops every pin at the calls that receive it. Its override hands the call to
``func``, the backend body ``wrap_attn`` would run itself with no pin, with the
same arguments, so an unpinned checkpoint renders exactly as before and a
pinned one renders as stock does unpinned. One default backend would differ:
with ``--use-ck-attention`` ComfyUI runs the comfy-kitchen int8 kernel, which
takes boxed q, k and v through a path of its own, and the override would run
its unboxed path instead. A worker never passes that flag. ``wrap_attn``
removes the preference before it calls an override, so this override cannot
log the pin. ``bind_stock_attention`` serves a module whose callers hand it no
options, such as the text refiners comfy runs from ``extra_conds``, and it can
log the pin, because it sees the module.

Only ``inject_usp`` uses these. Topologies without sequence parallelism keep
every stock call, so a pin runs there as it does in stock ComfyUI
(docs/ADAPTERS.md).
"""
from __future__ import annotations

import types

from .attention_inputs import STOCK_BACKEND_RUNS, note_preferred_attention
from .base import USP_ATTENTION_OVERRIDE_ATTR

_OVERRIDE = "optimized_attention_override"


def default_attention_override(func, *args, **kwargs):
    """Run the call as ``wrap_attn`` runs it when no block pins a kernel."""
    return func(*args, **kwargs)


setattr(default_attention_override, USP_ATTENTION_OVERRIDE_ATTR, True)


def stock_attention_options(transformer_options: dict | None) -> dict:
    """Copy options so the stock attention calls that receive them ignore a pin.

    The copy keeps an override the options already carry: ``wrap_attn`` runs
    it ahead of the pin, as stock ComfyUI would. The copy is shallow, as ``usp_options``'
    is, so nested patch dicts stay shared.
    """
    options = dict(transformer_options or {})
    if options.get(_OVERRIDE) is None:
        options[_OVERRIDE] = default_attention_override
    return options


def bind_stock_attention(module, site: str) -> None:
    """Bind ``module.forward`` so its stock attention call ignores a pin.

    For an attention module whose forward accepts ``transformer_options`` and
    whose callers pass none. The bound forward hands every argument to the
    class's own forward, so it replaces no stock code; a ComfyUI that dropped
    the ``transformer_options`` parameter fails the call loudly. It logs the
    pin once per ``site``.
    """
    stock_forward = type(module).forward

    def forward(self, *args, transformer_options=None, **kwargs):
        note_preferred_attention(site, getattr(self, "comfy_attention", None),
                                 chooser=STOCK_BACKEND_RUNS)
        return stock_forward(self, *args,
                             transformer_options=stock_attention_options(transformer_options),
                             **kwargs)

    module.forward = types.MethodType(forward, module)
