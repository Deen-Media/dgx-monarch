"""MiniMax H3's output head and block loop through the sharded forward, against stock (real ComfyUI, CPU).

ComfyUI ede4e5f (v0.45.0) changed the last argument of the H3 output head
from the two flow shifts to one decreasing sigma grid per stream, and moved
the PDD head bank's block choice onto that grid. This build's floor is
v0.45.0, so the adapter reads the installed head's form once at inject
(adapters/minimax_h3_packing.head_form): ``grids`` goes on, 0.44.x's
``shifts`` goes on only with a one-block head, which never reads the slot,
and anything else refuses class P before any bind.

Each test runs the form its tree has and skips with a reason on the other:
on v0.45.0 a stacked head (two and four blocks, the sampler's table on the
head grid and off it, at H3's shifts and at 1.0 and 1.0) must give stock's
bits at world 1, and the adapter's grid must equal stock's ``pdd_grid``; on
0.44.0 a one-block head must give stock's bits, what v1.4.0 rendered, and a
stacked one must refuse at inject. On both trees the sharded block loop must
make stock's prefetch calls, block scope included, because stock's forward
runs it inside ComfyUI's malloc graph. Identity shard and gather stand in for
world 1, and the attention override is dropped, so the blocks run stock
attention in both legs.
"""
from __future__ import annotations

import copy
import inspect
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.adapters import minimax_h3 as adapter_module
from dgx_monarch.adapters import minimax_h3_packing as packing
from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.minimax_h3 import MiniMaxH3Adapter
from dgx_monarch.refusal import RefusalClass, parse_refusal_tag
from module_location_helpers import from_checkout

COMFY_DIR = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
TEXT_DIM = 40


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy", "nodes", "node_helpers") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def comfy():
    """Import real ComfyUI, then leave sys.modules as found."""
    preserved = {n: m for n, m in sys.modules.items() if _is_comfy_module(n)}
    for name in preserved:
        sys.modules.pop(name, None)
    path_before = list(sys.path)
    argv_before = sys.argv
    if os.path.isdir(COMFY_DIR) and COMFY_DIR not in sys.path:
        sys.path.insert(0, COMFY_DIR)
    try:
        sys.argv = ["pytest-h3-head-form", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            pytest.importorskip("comfy.ldm.minimax.model")
            import comfy.ldm.minimax.model as h3
            import comfy.model_prefetch
            import comfy.ops
        finally:
            sys.argv = argv_before
        yield SimpleNamespace(h3=h3, ops=comfy.ops, prefetch=comfy.model_prefetch)
    finally:
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or from_checkout(module, COMFY_DIR)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = path_before


def _installed_form(comfy) -> str:
    names = list(inspect.signature(comfy.h3.FinalLayer.forward).parameters)
    return names[7] if len(names) > 7 else ""


def _needs(comfy, form):
    if _installed_form(comfy) != form:
        pytest.skip(f"this ComfyUI's H3 head takes {_installed_form(comfy)!r}, not {form!r}")


def _tiny(comfy, pdd_steps):
    """A two-block MiniMax H3 DiT in the curve form with a `pdd_steps`-block head bank."""
    torch.manual_seed(0)
    model = comfy.h3.MiniMaxH3Model(
        hidden_size=48, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2,
        attention_head_dim=24, ffn_hidden_size=64, text_dim=TEXT_DIM, time_embed_dim=16,
        rope_inv_freq_len=4, adaln_curve_grid=33, pdd_steps=pdd_steps, dtype=torch.float32,
        operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            if name.endswith("inv_freq"):
                tensor.copy_(1.0 / (10000 ** (torch.arange(tensor.numel()) / tensor.numel())))
            elif "norm" in name and name.endswith("weight"):
                tensor.copy_(1.0 + 0.1 * torch.randn_like(tensor))
            else:
                tensor.copy_(0.2 * torch.randn_like(tensor))
    return model.requires_grad_(False)


def _bound(model):
    clone = copy.deepcopy(model)
    MiniMaxH3Adapter().inject_usp(clone, InjectionContext(topology_sp=1, usp_attention=object()))
    return clone


@pytest.fixture
def world_one(monkeypatch):
    monkeypatch.setattr(adapter_module, "sp_world", lambda: 1)
    monkeypatch.setattr(adapter_module, "shard_seq", lambda tensor, dim=1: (tensor, tensor.shape[dim]))
    monkeypatch.setattr(adapter_module, "sp_gather", lambda tensor, _orig, dim=1: tensor)
    monkeypatch.setattr(adapter_module, "padded_row_indices", lambda _segments: [])
    monkeypatch.setattr(adapter_module, "usp_options", lambda options, _attn, drop_rows=None: dict(options))


def _run(comfy, network, sigma, table, shifts):
    generator = torch.Generator().manual_seed(3)
    video = torch.randn(1, 24, 2, 4, 4, generator=generator)
    audio = torch.randn(1, 32, 2, 3, generator=generator)
    context = torch.randn(1, 5, TEXT_DIM, generator=generator)
    timestep = torch.tensor([sigma * 1000.0])
    payload = {"seed": 7, "audio_scale": shifts[0] / shifts[1],
               "layout": comfy.h3.PackedLayout(5, 2, 4, 4, 3)}
    options = {"minimax_h3_sigma_shift_video": shifts[0], "minimax_h3_sigma_shift_audio": shifts[1],
               "sample_sigmas": table, "sigmas": timestep / 1000.0}
    with torch.inference_mode():
        return network.forward([video, audio], timestep, context, options, minimax_payload=payload)


def _tables(steps, shift):
    """The sampler's table on the head grid and off it."""
    grid = [packing.pdd_grid(steps, shift, _shift_sigma)[k] for k in range(steps + 1)]
    return {"on-grid": torch.tensor(grid, dtype=torch.float32),
            "off-grid": torch.tensor([1.0, 0.97, 0.9, 0.87, 0.7, 0.45, 0.2, 0.0])}


def _shift_sigma(sigma, from_shift, to_shift):
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


@pytest.mark.parametrize("shifts", [(12.0, 3.0), (1.0, 1.0)], ids=["h3-shift", "unit-shift"])
@pytest.mark.parametrize("table_kind", ["on-grid", "off-grid"])
@pytest.mark.parametrize("pdd_steps", [1, 2, 4])
def test_a_stacked_head_on_the_v045_form_gives_stocks_bits(comfy, world_one, pdd_steps, table_kind, shifts):
    _needs(comfy, packing.HEAD_GRIDS)
    model = _tiny(comfy, pdd_steps)
    bound = _bound(model)
    assert "_forward" in vars(bound) and "_forward" not in vars(model)
    table = _tables(pdd_steps, shifts[0])[table_kind]
    for sigma in [float(value) for value in table[:-1]]:
        stock, ours = _run(comfy, model, sigma, table, shifts), _run(comfy, bound, sigma, table, shifts)
        assert torch.equal(stock[0], ours[0]), f"video velocity differs from stock at sigma {sigma}"
        assert torch.equal(stock[1], ours[1]), f"audio velocity differs from stock at sigma {sigma}"
        assert torch.isfinite(stock[0]).all() and stock[0].abs().sum() > 0


@pytest.mark.parametrize("steps", [1, 2, 4, 6])
@pytest.mark.parametrize("shift", [12.0, 3.0, 1.0, 8.0])
def test_the_adapters_grid_is_stocks_pdd_grid(comfy, steps, shift):
    if not hasattr(comfy.h3, "pdd_grid"):
        pytest.skip("this ComfyUI predates pdd_grid (ComfyUI 0.45.0 added it)")
    ours = packing.pdd_grid(steps, shift, comfy.h3.time_shift_sigma)
    assert ours == comfy.h3.pdd_grid(steps, shift)


@pytest.mark.parametrize("sigma", [1.0, 0.9, 0.5, 0.1])
def test_a_one_block_head_on_the_v044_form_gives_stocks_bits(comfy, world_one, sigma):
    """The bits v1.4.0 rendered: the slot carries the shifts, which one block never reads."""
    _needs(comfy, packing.HEAD_SHIFTS)
    model = _tiny(comfy, 1)
    bound = _bound(model)
    table = torch.tensor([1.0, 0.9, 0.5, 0.1, 0.0])
    for shifts in ((12.0, 3.0), (1.0, 1.0)):
        stock, ours = _run(comfy, model, sigma, table, shifts), _run(comfy, bound, sigma, table, shifts)
        assert torch.equal(stock[0], ours[0]) and torch.equal(stock[1], ours[1])


def test_a_stacked_head_on_the_v044_form_refuses_at_inject(comfy):
    _needs(comfy, packing.HEAD_SHIFTS)
    clone = copy.deepcopy(_tiny(comfy, 2))
    with pytest.raises(UnsupportedModelError) as caught:
        MiniMaxH3Adapter().inject_usp(clone, InjectionContext(topology_sp=2, usp_attention=object()))
    assert "ComfyUI 0.45.0 or newer" in str(caught.value)
    assert parse_refusal_tag(str(caught.value)).refusal_class is RefusalClass.PHYSICS
    assert "_forward" not in vars(clone)


def test_the_block_loop_pops_as_stock_does(comfy, world_one, monkeypatch):
    """The queue over the blocks, every pop with the block scope, and the closing pop."""
    events: list = []
    real_queue, real_pop = comfy.prefetch.make_prefetch_queue, comfy.prefetch.prefetch_queue_pop

    def make_prefetch_queue(queue, device, options):
        events.append(("queue", [type(module).__name__ for module in queue]))
        return real_queue(queue, device, options)

    def prefetch_queue_pop(queue, device, module, *args, **kwargs):
        events.append(("pop", type(module).__name__ if module is not None else None,
                       args, tuple(sorted(kwargs.items()))))
        return real_pop(queue, device, module, *args, **kwargs)

    monkeypatch.setattr(comfy.prefetch, "make_prefetch_queue", make_prefetch_queue)
    monkeypatch.setattr(comfy.prefetch, "prefetch_queue_pop", prefetch_queue_pop)
    model = _tiny(comfy, 1)
    bound = _bound(model)
    table = torch.tensor([1.0, 0.5, 0.0])
    _run(comfy, model, 0.5, table, (12.0, 3.0))
    stock = list(events)
    events.clear()
    _run(comfy, bound, 0.5, table, (12.0, 3.0))
    assert events == stock
    assert stock[0] == ("queue", ["DiTBlock", "DiTBlock"])
    assert [event for event in stock if event[0] == "pop"] == (
        [("pop", "DiTBlock", (), (("malloc_scope", "block"),))] * 2
        + [("pop", None, (), (("malloc_scope", "block"),))])


def test_every_h3_block_sees_stocks_block_index(comfy, world_one, monkeypatch):
    """Stock's block loop sets block_index on the options each block takes; so does ours.

    No reader on the H3 video path changes a bit with it (no H3 module, attention
    override or patch this build ships reads it), so the H3 output is unchanged;
    a patch that reads it sees stock's value.
    """
    seen: list = []
    real = comfy.h3.DiTBlock.forward

    def forward(block, x, t_emb, mod_segments, rope_freqs, transformer_options=None, attention=None):
        transformer_options = {} if transformer_options is None else transformer_options
        seen.append(transformer_options.get("block_index"))
        return real(block, x, t_emb, mod_segments, rope_freqs,
                    transformer_options=transformer_options, attention=attention)

    monkeypatch.setattr(comfy.h3.DiTBlock, "forward", forward)
    model = _tiny(comfy, 1)
    bound = _bound(model)
    generator = torch.Generator().manual_seed(3)
    video = torch.randn(1, 24, 2, 4, 4, generator=generator)
    audio = torch.randn(1, 32, 2, 3, generator=generator)
    context = torch.randn(1, 5, TEXT_DIM, generator=generator)
    timestep = torch.tensor([500.0])
    finals, outputs = [], []
    for network in (model, bound):
        seen.clear()
        payload = {"seed": 7, "audio_scale": 4.0, "layout": comfy.h3.PackedLayout(5, 2, 4, 4, 3)}
        options = {"minimax_h3_sigma_shift_video": 12.0, "minimax_h3_sigma_shift_audio": 3.0,
                   "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]), "sigmas": timestep / 1000.0}
        with torch.inference_mode():
            outputs.append(network.forward([video, audio], timestep, context, options,
                                           minimax_payload=payload))
        finals.append((list(seen), options.get("block_index")))
    assert finals[0] == finals[1] == ([0, 1], 1)
    assert all(torch.equal(a, b) for a, b in zip(outputs[0], outputs[1], strict=True))
