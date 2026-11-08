"""Fused-norm parity for the Flux-family and Qwen Image 2.1 sharded blocks (issue #54).

comfy-kitchen's fused AdaLN picks its CUDA kernel by a call's kernel rows, so a
rank's window, normalized in a call sized to the window, can take the other
kernel than stock's call over the whole stream (DESIGN.md, the parity rule;
adapters/flux3_adaln_parity.py). On 2026-11-07 uly2 at 512 square read
one-step NRMS 0.006 to 0.062 against one GPU on Flux 1, Flux 2, Chroma and
Qwen Image 2.1 for that reason, while 1024 square read bit-identical: there a
rank's windows hold more than 1024 rows, as stock's calls do.

The Flux-family forward (adapters/flux_family.py) runs comfy's own
``DoubleStreamBlock`` and ``SingleStreamBlock`` on a rank's windows and puts
stock's row count of each stream in the block's transformer options under
``NORM_ROWS`` (``with_norm_rows``). ``bind_flux_norm_parity`` binds a block
so each ``modulated_norm`` call runs through ``at_stock_rows`` at its stream's
count: a double block's image and text norms at the image and text rows, a
single block's at the joint rows. Qwen Image 2.1's sharded forward
(adapters/qwen_image21.py) gives its target blocks the prefix plus the whole
target as the ``joint`` count, and ``bind_qwen21_norm_parity`` fills
``_modulated_norm``'s first call to it; its second call redoes the prefix
rows, which every rank already holds whole. Options without the count, and
every topology without sequence parallelism, run the class forward unchanged.
The op is row-local, so the filled call returns stock's bits for the window's
rows.
"""
from __future__ import annotations

import inspect
import types
from typing import Any

from ..log import get_logger
from .flux3_adaln_parity import NORM_GLOBAL, at_stock_rows, loads_global, route_global

log = get_logger(__name__)

NORM_ROWS = "dgxm_stock_norm_rows"
QWEN21_NORM_GLOBAL = "_modulated_norm"
# The stock Qwen Image 2.1 block forward's parameters, which the bound forward
# repeats; a ComfyUI that changes them leaves the block unbound and logged.
_QWEN21_FORWARD_PARAMETERS = ("self", "x", "mod", "pe", "attn_fn", "prefix_len",
                              "transformer_options")
# Each Flux-family norm module and the stream whose rows its call covers.
_FLUX_NORMS = (("img_norm1", "img"), ("img_norm2", "img"), ("txt_norm1", "txt"),
               ("txt_norm2", "txt"), ("pre_norm", "joint"))


def _routable_forward(block: Any, global_name: str) -> types.FunctionType | None:
    """The block's class ``forward`` when its own code reads ``global_name``."""
    forward = getattr(type(block), "forward", None)
    if isinstance(forward, types.FunctionType) and loads_global(forward, global_name):
        return forward
    return None


def with_norm_rows(options: dict, **rows: int) -> dict:
    """``options`` with stock's row count of each stream the blocks normalize."""
    options[NORM_ROWS] = dict(rows)
    return options


def _flux_plan(block: Any, rows: dict) -> list[tuple[Any, int]]:
    """``(norm module, stock rows)`` for every norm this block's streams carry."""
    return [(getattr(block, name), rows[stream]) for name, stream in _FLUX_NORMS
            if stream in rows and getattr(block, name, None) is not None]


def _flux_parity_norm(stock_norm, plan: list[tuple[Any, int]]):
    """Stock's ``modulated_norm``, each call launched at its stream's stock rows."""

    def modulated_norm(x, norm, scale, shift, modulation_dims=None):
        rows = next((count for known, count in plan if known is norm), None)
        # The blocks call it once per norm module with no modulation_dims on
        # the sharded path, which refuses the span-modulated reference method.
        # Any other call means ComfyUI changed the block.
        assert rows is not None and modulation_dims is None, (
            "flux-family norm parity: a stock block made a fused-norm call that is "
            "not one of its streams' calls, so ComfyUI changed the block; "
            "tests/test_adaln_parity_blocks.py names the change against an installed ComfyUI")
        return at_stock_rows(lambda call: stock_norm(call, norm, scale, shift), x, rows)

    return modulated_norm


def bind_flux_norm_parity(block: Any) -> bool:
    """Bind ``block.forward`` so each ``modulated_norm`` call runs at stock's rows.

    The rows come from the call's ``transformer_options`` (``with_norm_rows``);
    options without them run the class forward. A class forward that reads no
    ``modulated_norm`` global is left unbound. A repeat bind replaces the last
    one, because it wraps the class function. Returns whether it bound.
    """
    found = _routable_forward(block, NORM_GLOBAL)
    if found is None:
        return False
    stock_forward: types.FunctionType = found

    def forward(self, *args, transformer_options={}, **kwargs):
        rows = (transformer_options or {}).get(NORM_ROWS)
        if not rows:
            return stock_forward(self, *args, transformer_options=transformer_options, **kwargs)
        plan = _flux_plan(self, rows)
        routed = route_global(stock_forward, NORM_GLOBAL,
                              lambda stock_norm: _flux_parity_norm(stock_norm, plan))
        return routed(self, *args, transformer_options=transformer_options, **kwargs)

    block.forward = types.MethodType(forward, block)
    return True


def bind_qwen21_norm_parity(block: Any) -> bool:
    """Bind a Qwen Image 2.1 block so ``_modulated_norm`` runs at stock's rows.

    The count is the ``joint`` entry of the call's options (``with_norm_rows``),
    the prefix plus the whole target; options without it run the class forward.
    A class forward that reads no ``_modulated_norm`` global, or whose
    parameters are not the ones the bound forward repeats, is left unbound.
    Returns whether the block was bound.
    """
    found = _routable_forward(block, QWEN21_NORM_GLOBAL)
    if found is None or tuple(inspect.signature(found).parameters) != _QWEN21_FORWARD_PARAMETERS:
        return False
    stock_forward: types.FunctionType = found

    def forward(self, x, mod, pe, attn_fn, prefix_len, transformer_options={}):
        rows = ((transformer_options or {}).get(NORM_ROWS) or {}).get("joint")
        if rows is None:
            return stock_forward(self, x, mod, pe, attn_fn, prefix_len, transformer_options)

        def wrap(stock_norm):
            def modulated_norm(norm, y, scale, prefix, zero):
                return at_stock_rows(lambda call: stock_norm(norm, call, scale, prefix, zero),
                                     y, rows)
            return modulated_norm

        routed = route_global(stock_forward, QWEN21_NORM_GLOBAL, wrap)
        return routed(self, x, mod, pe, attn_fn, prefix_len, transformer_options)

    block.forward = types.MethodType(forward, block)
    return True


def bind_blocks(blocks, binder, family: str) -> None:
    """Bind every block a sharded forward runs; log the classes left unbound.

    A block whose class forward stopped reading its norm global, or a Qwen
    Image 2.1 block whose forward takes other parameters, runs unrouted: the
    path every render took before the route, bit-identical at the shipped
    sizes and off one GPU at 512 square. The Worker log names it, and the
    real-ComfyUI check in tests/test_adaln_parity_blocks.py names it in CI.
    """
    left = sorted({type(block).__name__ for block in blocks if not binder(block)})
    if left:
        log.warning("%s fused-norm parity: %s.forward reads no norm global this release "
                    "routes, or takes other parameters, so its calls run at the window's "
                    "rows (issue #54)", family, ", ".join(left))
