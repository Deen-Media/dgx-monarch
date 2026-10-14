"""Stand-ins for comfy's attention dispatch in the per-block pin tests (issue #30).

``install`` puts stub comfy leaves and a one-rank ``xfuser.core.distributed``
in ``sys.modules``. The stub ``optimized_attention`` runs through the
reproduction of comfy's ``wrap_attn`` in tests/test_attention_container_seam.py,
so an override in ``transformer_options`` is checked before a block's
preference, as in comfy. ``Pin`` stands in for a ``ComfyAttention`` that a
checkpoint pins to comfy-kitchen Sol: its function counts its runs and returns
zeros, which the default attention never returns. ``StockAttention`` stands in
for a stock comfy attention module that the pack leaves in place.

``assert_pins_dropped`` runs a family's injected forward three ways: pinned
with the pack's override, unpinned with no override (stock dispatch for a
checkpoint with no pin), and pinned with the override removed, which must run
every pin, so each test proves its fixture reaches every site it names.
"""
from __future__ import annotations

import sys
import types
from collections.abc import Callable

import torch

from test_attention_container_seam import _AttentionTensorContainer, _wrap_attn


class Pin:
    """A ComfyAttention stand-in, pinned to comfy-kitchen Sol or unpinned."""

    def __init__(self, pinned: bool = True):
        self.config = {"attention": "comfy_kitchen_sol", "tau": 1.0} if pinned else None
        self.function = self._run if pinned else None
        self.runs = 0

    def _run(self, q, k, v, heads, **_kwargs):
        self.runs += 1
        if isinstance(q, _AttentionTensorContainer):
            q, k, v = q.take(), k.take(), v.take()
        return torch.zeros_like(q)


def _default_attention(q, k, v, heads, mask=None, **_kwargs):
    """Exact attention on ``(B, L, heads * D)`` rows, comfy's unpinned path here."""
    b, rows, inner = q.shape
    d = inner // heads

    def split(t):
        return t.reshape(b, t.shape[1], heads, d).transpose(1, 2)

    out = torch.nn.functional.scaled_dot_product_attention(
        split(q), split(k), split(v), attn_mask=mask)
    return out.transpose(1, 2).reshape(b, rows, inner)


class StockAttention(torch.nn.Module):
    """A stock comfy attention module: q from x, k and v from the context or x.

    It hands comfy its preference and whatever options its caller passed, as
    comfy's modules do, and adds the attention to its input.
    """

    def __init__(self, pins: list, pinned: bool, heads: int = 2):
        super().__init__()
        self.heads = heads
        self.comfy_attention = Pin(pinned)
        pins.append(self.comfy_attention)

    def forward(self, x, context=None, *_args, transformer_options=None, **_kwargs):
        attention = sys.modules["comfy.ldm.modules.attention"]
        box = attention.AttentionTensorContainer
        kv = x if context is None else context
        out = attention.optimized_attention(
            box(x), box(kv), box(kv), self.heads,
            preferred_attention=self.comfy_attention,
            transformer_options=transformer_options)
        return x + out


def _modules(monkeypatch, tree: dict[str, dict]) -> None:
    made: dict[str, types.ModuleType] = {}
    for name in sorted(tree, key=lambda n: n.count(".")):
        parts = name.split(".")
        for depth in range(1, len(parts) + 1):
            prefix = ".".join(parts[:depth])
            if prefix in made:
                continue
            module = types.ModuleType(prefix)
            made[prefix] = module
            if depth > 1:
                setattr(made[".".join(parts[:depth - 1])], parts[depth - 1], module)
            monkeypatch.setitem(sys.modules, prefix, module)
        for attr, value in tree[name].items():
            setattr(made[name], attr, value)


def install(monkeypatch) -> None:
    """Stub the comfy leaves the injected forwards import, and one SP rank."""
    optimized_attention = _wrap_attn(_default_attention)
    _modules(monkeypatch, {
        "comfy.ldm.modules.attention": {
            "AttentionTensorContainer": _AttentionTensorContainer,
            "optimized_attention": optimized_attention,
            "optimized_attention_masked": optimized_attention,
        },
        "comfy.ldm.common_dit": {"pad_to_patch_size": lambda x, _patch: x},
        "comfy.ldm.flux.math": {"apply_rope1": lambda t, _freqs: t},
        "comfy.ldm.flux.layers": {
            "timestep_embedding": lambda t, dim, **_kw: torch.zeros(t.shape[0], dim)},
        "comfy.ldm.wan.model": {
            "sinusoidal_embedding_1d": lambda dim, t: torch.zeros(t.shape[0], dim)},
        "comfy.model_management": {"cast_to": lambda t, dtype=None, device=None: t},
        "xfuser.core.distributed": {
            "get_sequence_parallel_world_size": lambda: 1,
            "get_sequence_parallel_rank": lambda: 0,
            "get_ulysses_parallel_world_size": lambda: 1,
            "get_ring_parallel_world_size": lambda: 1,
            "get_sp_group": lambda: types.SimpleNamespace(all_gather=lambda t, dim: t),
        },
    })


def one_rank_attention(q, k, v, heads, **_kwargs):
    """A one-rank stand-in for the sharded dispatcher: plain exact attention."""
    return _default_attention(q, k, v, heads)


def without_override(monkeypatch, *modules) -> None:
    """Undo the pack's change in ``modules``: options pass through, no binding."""
    for module in modules:
        if hasattr(module, "stock_attention_options"):
            monkeypatch.setattr(module, "stock_attention_options",
                                lambda options: dict(options or {}))
        if hasattr(module, "bind_stock_attention"):
            monkeypatch.setattr(module, "bind_stock_attention", lambda _module, _site: None)


def assert_pins_dropped(monkeypatch, run: Callable[[bool], tuple], *modules,
                        undo: Callable | None = None) -> None:
    """``run(pinned)`` builds, injects and runs one forward: ``(output, pins)``.

    ``undo(patch)`` removes any other pin drop the family has, such as a bound
    copy of a stock forward.
    """
    out, pins = run(True)
    assert pins, "the fixture builds no pinned site"
    assert [pin.runs for pin in pins] == [0] * len(pins), (
        "a block's pinned kernel ran under sequence parallelism")
    with monkeypatch.context() as patch:
        without_override(patch, *modules)
        if undo is not None:
            undo(patch)
        reference, _ = run(False)
        stock_out, stock_pins = run(True)
    assert torch.equal(out, reference), (
        "dropping the pin changed the output: it must equal stock with no pin")
    assert all(pin.runs > 0 for pin in stock_pins), (
        "without the override some site never ran its pin, so the fixture misses it")
    assert not torch.equal(stock_out, reference)
