"""FLUX.3 Klein fused-norm parity: each span piece runs at stock's kernel row count.

Stock ``Block.forward`` (comfy/ldm/flux3/model.py) normalizes every row of its
stream in one ``modulated_norm`` call with the first span's modulation, then
redoes each later span in a call of its own. ``modulated_norm`` runs
comfy-kitchen's fused AdaLN, whose CUDA launcher picks a kernel by the call's
kernel rows, batch times rows. Measured on 2026-10-16, comfy-kitchen 0.2.37
took one kernel at 1024 kernel rows or fewer and another above, and the two
rounded a few outputs one bf16 step apart. A rank holds only a window of the stream, so its calls are
smaller than stock's and can fall on the other side of that branch. The op is
row-local, so a piece gets stock's bits back from a call of stock's size.

``at_stock_rows`` fills a piece's call with zero rows up to stock's row count,
runs the op once and keeps the piece's rows. It reads no branch point, so it
holds for whatever row counts a later comfy-kitchen keys its kernels on.
``bind_adaln_parity`` binds one sharded block's ``forward``: a call that
passes ``stock_norm_rows`` runs stock's own ``Block.forward`` code with its one
global ``modulated_norm`` resolved to a wrapper that sends each span's call
through ``at_stock_rows``. Stock's class, its module and every block the
sharded forward does not call stay as they are, and topologies without
sequence parallelism never bind. MOSS-Image's binder
(adapters/moss_image_seams.py) reuses ``at_stock_rows``, ``loads_global`` and
``route_global``. This module imports no ComfyUI module, so the CPU suite
reaches it without one.
"""
from __future__ import annotations

import dis
import types
from collections.abc import Callable, Sequence
from typing import Any

import torch

NORM_GLOBAL = "modulated_norm"


def pad_rows(piece_rows: int, stock_rows: int) -> int:
    """The zero rows that fill a ``piece_rows`` call to stock's ``stock_rows``.

    Both counts are rows a sample; the batch is the same in both calls, so the
    kernel rows match once the rows do.
    """
    if not 0 < piece_rows <= stock_rows:
        raise ValueError(
            f"fused-norm parity: a {piece_rows}-row piece cannot sit in a "
            f"{stock_rows}-row stock call")
    return stock_rows - piece_rows


def at_stock_rows(op: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor,
                  stock_rows: int) -> torch.Tensor:
    """``op(x)`` for a row-local ``op``, launched over ``stock_rows`` rows a sample.

    ``x`` is ``[B, n, D]``. Zero rows after the piece fill the call to stock's
    size, so the launch holds stock's ``B * stock_rows`` kernel rows, and the
    result keeps the piece's ``n`` rows. The result is contiguous, as stock's
    own output is, so the GEMMs after it see stock's layout at batch 2 too. A
    piece of stock's size runs as it is, with no copy.
    """
    batch, rows, width = x.shape
    if not pad_rows(rows, stock_rows):
        return op(x)
    call = x.new_empty(batch, stock_rows, width)
    call[:, :rows].copy_(x)
    call[:, rows:].zero_()
    return op(call)[:, :rows].contiguous()


def loads_global(forward: Callable[..., Any], name: str) -> bool:
    """Whether ``forward``'s own code reads ``name`` as a module global."""
    function = getattr(forward, "__func__", forward)
    if not isinstance(function, types.FunctionType):
        return False
    return any(ins.opname == "LOAD_GLOBAL" and ins.argval == name
               for ins in dis.get_instructions(function))


def loads_norm_global(forward: Callable[..., Any]) -> bool:
    """Whether ``forward``'s own code reads ``modulated_norm`` as a global."""
    return loads_global(forward, NORM_GLOBAL)


def _plan(spans: Sequence, stock_norm_rows: Sequence[int]) -> list[tuple[Any, int]]:
    """``(scale, stock rows)`` for each clipped span, the key its norm call carries."""
    if len(spans) != len(stock_norm_rows):
        raise ValueError(
            f"flux3 norm parity: {len(spans)} spans and {len(stock_norm_rows)} "
            "stock row counts")
    plan: list[tuple[Any, int]] = []
    for (_start, _end, mod), rows in zip(spans, stock_norm_rows, strict=True):
        for scale, known in plan:
            assert scale is not mod.scale or known == rows, (
                "flux3 norm parity: two spans share one modulation with different "
                "stock row counts")
        plan.append((mod.scale, rows))
    return plan


def _parity_norm(stock_norm: Callable[..., torch.Tensor],
                 plan: list[tuple[Any, int]]) -> Callable[..., torch.Tensor]:
    """Stock's ``modulated_norm``, each call launched at its span's stock row count."""

    def modulated_norm(x, norm, scale, shift, modulation_dims=None):
        rows = next((rows for known, rows in plan if known is scale), None)
        # Block.forward calls the norm once per span with that span's
        # modulation and no modulation_dims. Any other call means ComfyUI
        # changed the block, and passing it through would hide a divergence.
        assert rows is not None and modulation_dims is None, (
            "flux3 norm parity: stock Block.forward made a fused-norm call that is "
            "not one of its spans' calls, so ComfyUI changed the block; the "
            "canary in tests/test_flux3_comfy_exactness.py names the change")
        return at_stock_rows(lambda call: stock_norm(call, norm, scale, shift), x, rows)

    return modulated_norm


def route_global(stock_forward: types.FunctionType, name: str,
                 wrap: Callable[[Callable[..., Any]], Callable[..., Any]]) -> types.FunctionType:
    """Stock's forward code, reading the live module globals with ``name`` wrapped.

    The namespace is copied per call, so a module global ComfyUI or a test
    rebinds later still reaches the code, and the module itself never changes.
    ``wrap`` receives stock's global and returns what the code reads instead.
    """
    namespace = dict(stock_forward.__globals__)
    stock_global = namespace.get(name)
    if not callable(stock_global):
        raise TypeError(
            f"fused-norm parity: {stock_forward.__module__}.{name} is "
            f"{type(stock_global).__name__}, not a callable stock global")
    namespace[name] = wrap(stock_global)
    routed = types.FunctionType(stock_forward.__code__, namespace, stock_forward.__name__,
                                stock_forward.__defaults__, stock_forward.__closure__)
    routed.__kwdefaults__ = stock_forward.__kwdefaults__
    routed.__qualname__ = stock_forward.__qualname__
    return routed


def _routed(stock_forward: types.FunctionType, plan: list[tuple[Any, int]]) -> types.FunctionType:
    """Stock's forward code with its ``modulated_norm`` global launched at stock's rows."""
    return route_global(stock_forward, NORM_GLOBAL, lambda stock_norm: _parity_norm(stock_norm, plan))


def bind_adaln_parity(block: Any) -> bool:
    """Bind ``block.forward`` so each fused-norm call runs at stock's row count.

    The bound forward takes stock's arguments plus ``stock_norm_rows``, one
    count a span in the order of ``spans``: the stream's rows for the first
    span, whose call covers every row, and the stock span's rows for each
    later one. Without it, or when the class forward reads no
    ``modulated_norm`` global, the call is the class forward's own. A repeat
    bind replaces the last one, because it wraps the class function. Returns
    whether the class forward reads the global, which the caller logs.
    """
    stock_forward = type(block).forward
    routable = loads_norm_global(stock_forward)

    def forward(self, x, spans, *args, stock_norm_rows=None, **kwargs):
        if stock_norm_rows is None or not routable:
            return stock_forward(self, x, spans, *args, **kwargs)
        routed = _routed(stock_forward, _plan(spans, stock_norm_rows))
        return routed(self, x, spans, *args, **kwargs)

    block.forward = types.MethodType(forward, block)
    return routable
