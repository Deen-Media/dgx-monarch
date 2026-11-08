"""LTX stock-row parity: the row-count-keyed kernels in a sharded block run at stock's row count.

Under sequence parallelism a rank runs LTX's blocks on its share of each stream's rows, and a
kernel that picks its implementation from the row count can round a row differently from the
one stock ComfyUI launches over the whole stream. On 2026-11-07 LTX 2.3 on uly2 read one-step
latent NRMS 0.012 to 0.014 against one GPU, where Wan 2.1 on the same build was bit-identical.
An op probe on GB10 the same day, under torch 2.15.0 (cuBLAS 13.4.1.3) and torch 2.12.0 (cuBLAS
13.4.0.1) alike, named the two kernels in LTXAV's blocks that key on the call's rows:

* the per-head gate projection ``to_gate_logits``, a GEMM 32 columns wide (one column a head):
  at a rank's 7040, 2048 and 1024 rows it rounded 21,000 to 101,000 outputs differently from
  the call at stock's 14080, 4096 and 2048 rows, max abs 0.03 to 0.06, the class Krea2's
  narrow output Linear met. The wide q, k, v, out and FFN GEMMs kept a row's bits;
* comfy-kitchen's fused ``rms_adaln``, which takes a warp kernel at 1024 kernel rows or fewer
  and a block kernel above (``launch_adaln_kernel``, the switch FLUX.3 and MOSS-Image already
  route): it differed at 1024 rows a rank and nowhere above.

Both are row-local, so a rank gets stock's bits back from a call of stock's size. Each call
fills the rank's rows with zero rows up to the stream's stock count, the rank's rows at the
front, through FLUX.3's ``at_stock_rows``; the probe read that fill exact for both calls, as it
read a fill with the rows at their stock offset.

``StockRows.sharded`` is entered by the bound ``_process_transformer_blocks`` around stock's
block loop and names each stream's geometry; outside it every bound call is stock's own. The
fused-norm route reaches stock's block code through its ``comfy`` global, with an overlay
whose ``quant_ops.ck.rms_adaln`` is the filled call and every other name stock's own, and
through the module global ``apply_cross_attention_adaln``, which the block's text cross
attention calls for both streams. Topologies without sequence parallelism never bind. Canary
seam 51 (ltx stock-row parity) holds the globals this reads.
"""
from __future__ import annotations

import threading
import types
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, NoReturn

import torch

from ..refusal import RefusalClass, refusal
from .base import UnsupportedModelError
from .flux3_adaln_parity import at_stock_rows, loads_global

VIDEO, AUDIO = "video", "audio"
COMFY_GLOBAL = "comfy"
CROSS_ADALN_GLOBAL = "apply_cross_attention_adaln"
TEXT_CROSS_METHOD = "_apply_text_cross_attention"
# The stream whose rows each gated attention's queries (and so its gate input) carry.
# comfy names a cross attention for the stream it carries into the other, so
# audio_to_video_attn takes video queries (BasicAVTransformerBlock.forward).
GATE_STREAMS = (("attn1", VIDEO), ("attn2", VIDEO), ("audio_to_video_attn", VIDEO),
                ("audio_attn1", AUDIO), ("audio_attn2", AUDIO), ("video_to_audio_attn", AUDIO))


@dataclass(frozen=True)
class ShardRows:
    """One stream's rows a sample: stock's count, and this rank's, pad rows included."""

    stock: int
    local: int


def refuse_unplanned_call(detail: str) -> NoReturn:
    """A row-count-keyed call the sharded block was not planned for, on every rank alike."""
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"ltx: {detail} inside the sharded block loop, so ComfyUI changed the LTX block. This "
        "release launches each row-count-keyed call at stock's row count and cannot show that "
        "a call it does not know computes what one GPU computes. Use an updated dgx-monarch, "
        "or topology 'single' (mode=local with gpus_per_host=1); canary seam 51 (ltx "
        "stock-row parity) names the change.",
    ))


def _rows_detail(what: str, rows: int, planned: int) -> str:
    return f"a {what} call carried {rows} rows where this rank holds {planned}"


def refuse_unroutable(changed: list[str]) -> NoReturn:
    """Refuse at inject, before any bind, when stock's block no longer reads a routed global."""
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"ltx: this ComfyUI's {', '.join(changed)} no longer reads the globals the stock-row "
        "parity route binds, so the sharded forward cannot launch the fused AdaLN at stock's "
        "row count and cannot show that it computes what one GPU computes. Use topology "
        "'single' (mode=local with gpus_per_host=1), or a dgx-monarch release that supports "
        "this ComfyUI; canary seam 51 (ltx stock-row parity) names the change.",
    ))


class StockRows(threading.local):
    """Each stream's geometry while the sharded block loop runs, on this thread."""

    def __init__(self) -> None:
        self.streams: dict[str, ShardRows] | None = None

    @contextmanager
    def sharded(self, streams: Mapping[str, ShardRows]) -> Iterator[None]:
        self.streams = dict(streams)
        try:
            yield
        finally:
            self.streams = None

    def rows(self, stream: str) -> ShardRows | None:
        return None if self.streams is None else self.streams.get(stream)


def at_stream_rows(op: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor,
                   rows: ShardRows, what: str) -> torch.Tensor:
    """``op(x)`` launched over the stream's stock rows, x's rows at the front (``at_stock_rows``)."""
    if x.ndim != 3 or x.shape[1] != rows.local:
        refuse_unplanned_call(_rows_detail(what, int(x.shape[-2]), rows.local))
    if rows.stock == rows.local:
        return op(x)
    return at_stock_rows(op, x, rows.stock)


def bind_gate_parity(attn: Any, stream: str, state: StockRows) -> bool:
    """Bind ``attn.to_gate_logits.forward`` to run over the stream's stock rows.

    Returns False for an attention with no gate (LTXV and ungated checkpoints)."""
    gate = getattr(attn, "to_gate_logits", None)
    if gate is None:
        return False
    stock_forward = type(gate).forward

    def forward(self, x, *args, **kwargs):
        rows = state.rows(stream)
        if rows is None or args or kwargs:
            return stock_forward(self, x, *args, **kwargs)
        return at_stream_rows(lambda call: stock_forward(self, call), x, rows, "gate projection")

    gate.forward = types.MethodType(forward, gate)
    return True


def _fill_rows(t: torch.Tensor, local: int, stock: int) -> torch.Tensor:
    """A per-token modulation filled with zero rows to stock's count; a broadcast one as it is."""
    if t.ndim != 3 or t.shape[1] != local or local <= 1:
        return t
    out = t.new_zeros(t.shape[0], stock, t.shape[2])
    out[:, :local] = t
    return out


def parity_rms_adaln(state: StockRows, stream: str,
                     stock: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """``ck.rms_adaln`` launched over the stream's stock rows, prefix placement."""

    def rms_adaln(x, scale, shift, eps=1e-6):
        rows = state.rows(stream)
        if rows is None:
            return stock(x, scale, shift, eps)
        call_scale = _fill_rows(scale, rows.local, rows.stock)
        call_shift = _fill_rows(shift, rows.local, rows.stock)
        return at_stream_rows(lambda call: stock(call, call_scale, call_shift, eps), x, rows,
                              "fused AdaLN")

    return rms_adaln


class _Overlay:
    """Reads every attribute from ``target`` except the ones in ``overrides``."""

    __slots__ = ("_overrides", "_target")

    def __init__(self, target: Any, overrides: Mapping[str, Any]) -> None:
        self._target, self._overrides = target, dict(overrides)

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._target, name)


def comfy_overlay(comfy: Any, state: StockRows, stream: str) -> _Overlay:
    """Stock's ``comfy`` package with ``quant_ops.ck.rms_adaln`` launched at stock rows."""
    quant_ops = comfy.quant_ops
    ck = quant_ops.ck
    routed = parity_rms_adaln(state, stream, ck.rms_adaln)
    return _Overlay(comfy, {"quant_ops": _Overlay(quant_ops, {"ck": _Overlay(ck, {"rms_adaln": routed})})})


def with_globals(function: types.FunctionType,
                 wraps: Mapping[str, Callable[[Any], Any]]) -> types.FunctionType:
    """Stock's code reading a per-call copy of its module globals, the named ones wrapped.

    Unlike FLUX.3's ``route_global`` this takes a module global too (``comfy``). The copy
    is made per call, so a global ComfyUI or a test rebinds later still reaches the code,
    and the module itself never changes.
    """
    namespace = dict(function.__globals__)
    for name, wrap in wraps.items():
        namespace[name] = wrap(namespace[name])
    routed = types.FunctionType(function.__code__, namespace, function.__name__,
                                function.__defaults__, function.__closure__)
    routed.__kwdefaults__ = function.__kwdefaults__
    routed.__qualname__ = function.__qualname__
    return routed


def _comfy_at(state: StockRows, stream: str) -> Callable[[Any], _Overlay]:
    return lambda comfy: comfy_overlay(comfy, state, stream)


def _cross_adaln(state: StockRows, stream_of: Callable[[Any], str]):
    """Wrap stock's ``apply_cross_attention_adaln`` so each call routes its stream's norm."""

    def wrap(stock_fn):
        def apply_cross_attention_adaln(x, context, attn, *args, **kwargs):
            routed = with_globals(stock_fn, {COMFY_GLOBAL: _comfy_at(state, stream_of(attn))})
            return routed(x, context, attn, *args, **kwargs)
        return apply_cross_attention_adaln
    return wrap


def unroutable(blocks, cross_adaln: Callable) -> list[str]:
    """Stock functions these blocks run that no longer read a global the route binds."""
    changed = set()
    if not loads_global(cross_adaln, COMFY_GLOBAL):
        changed.add(CROSS_ADALN_GLOBAL)
    for block in blocks:
        cls = type(block)
        if not loads_global(cls.forward, COMFY_GLOBAL):
            changed.add(f"{cls.__name__}.forward")
        if not getattr(block, "cross_attention_adaln", False):
            continue
        text_cross = getattr(cls, TEXT_CROSS_METHOD, None)
        holder, label = ((text_cross, TEXT_CROSS_METHOD) if text_cross is not None
                         else (cls.forward, "forward"))
        if not loads_global(holder, CROSS_ADALN_GLOBAL):
            changed.add(f"{cls.__name__}.{label}")
    return sorted(changed)


def bind_norm_parity(block: Any, state: StockRows) -> None:
    """Bind one block so each of its fused-AdaLN calls launches at its stream's stock rows.

    LTXAV's ``BasicAVTransformerBlock.forward`` normalizes only video rows itself and hands
    both streams' text cross attention to ``_apply_text_cross_attention``, which calls
    ``apply_cross_attention_adaln``; LTXV's ``BasicTransformerBlock.forward`` does both on
    video rows. ``unroutable`` has proven each read before this binds. Outside the sharded
    loop both bound methods are stock's own.
    """
    cls = type(block)
    stock_forward = cls.forward
    text_cross = getattr(cls, TEXT_CROSS_METHOD, None)

    def by_attention(attn):
        if attn is block.attn2:
            return VIDEO
        if attn is getattr(block, "audio_attn2", None):
            return AUDIO
        refuse_unplanned_call("a text cross-attention AdaLN ran for an attention that is "
                              "neither the video nor the audio text attention")

    wraps: dict[str, Callable[[Any], Any]] = {COMFY_GLOBAL: _comfy_at(state, VIDEO)}
    if text_cross is None:
        wraps[CROSS_ADALN_GLOBAL] = _cross_adaln(state, lambda _attn: VIDEO)

    def forward(self, *args, **kwargs):
        if state.streams is None:
            return stock_forward(self, *args, **kwargs)
        return with_globals(stock_forward, wraps)(self, *args, **kwargs)

    block.forward = types.MethodType(forward, block)
    if text_cross is not None:
        cross_wraps = {CROSS_ADALN_GLOBAL: _cross_adaln(state, by_attention)}

        def apply_text_cross_attention(self, *args, **kwargs):
            if state.streams is None:
                return text_cross(self, *args, **kwargs)
            return with_globals(text_cross, cross_wraps)(self, *args, **kwargs)

        setattr(block, TEXT_CROSS_METHOD, types.MethodType(apply_text_cross_attention, block))


def bind_stock_row_parity(diffusion_model: Any, state: StockRows) -> int:
    """Bind every block's fused AdaLN and every gate; return the gates bound.

    Refuses before binding anything when stock's code no longer reads a routed global.
    """
    from comfy.ldm.lightricks.model import apply_cross_attention_adaln

    blocks = list(diffusion_model.transformer_blocks)
    changed = unroutable(blocks, apply_cross_attention_adaln)
    if changed:
        refuse_unroutable(changed)
    gates = 0
    for block in blocks:
        bind_norm_parity(block, state)
        for name, stream in GATE_STREAMS:
            attn = getattr(block, name, None)
            if attn is not None:
                gates += bind_gate_parity(attn, stream, state)
    return gates
