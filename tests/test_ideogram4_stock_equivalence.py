"""Ideogram4's sharded forward against comfy's own forward with exact tensor equality (CPU).

Builds comfy's real Ideogram4Transformer2DModel at toy size in float32 and runs
its stock `_forward` once. Then it runs the adapter's bound forward for two
ranks, one thread each, with only the all-to-all, the sequence gather and the
kernel stubbed (tests/test_flux_pad_exclusion.py `_install_stubs`). One CPU
attention serves both sides, so the comparison is of operands: every kernel
call must receive stock's q, k and v, the same rows in the same order, and the
latent must equal stock's with a largest difference of exactly zero.

It covers odd and even text and image row counts, batch 1 and 2, and the
unconditional model's image-only call. Ideogram4 takes no reference latents
and no guidance, and the shipped template feeds no attention mask, so none is
covered here; a padded multi-prompt mask stays an open departure (pixeldit.py
drops it with a warning).

The zero bound holds only where this CPU's GEMM gives a row the same bits at
the shard's row count as at stock's. The per-block Linears are not treated for
row count, so a CPU that breaks that would fail this test on the untreated
departure rather than on the one under test; a precondition checks it and
skips the equality with that reason, and the negative control still runs.
It skips where ComfyUI is not importable; the comfy-canary job runs it against
comfy master. comfy.cli_args parses argv on import, so the fixture forces
``--cpu``.
"""
from __future__ import annotations

import copy
import os
import sys
import threading

import pytest
import torch

from dgx_monarch.adapters import base
from dgx_monarch.adapters.base import InjectionContext
from dgx_monarch.adapters.pixeldit import Ideogram4Adapter
from module_location_helpers import from_checkout
from test_flux_pad_exclusion import _LOCAL, _Fabric, _install_stubs, _rank

WORLD = 2
# (text rows, image grid); None is the unconditional model's image-only call.
# The packed total decides the pad: 4 + 8 and 5 + 9 divide two, 5 + 8 and
# 4 + 9 pad one row, and an image-only call pads when the grid is odd.
CASES = [
    (4, (2, 4)), (5, (2, 4)), (4, (3, 3)), (5, (3, 3)),
    (None, (2, 4)), (None, (3, 3)),
]


def _comfy_dir() -> str:
    return os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def comfy_ideogram4():
    """Import real `comfy.ops` and the Ideogram4 model, then put sys.modules back.

    The restore also drops modules the import pulled in from the checkout
    under other names, so later stub-based files find the namespace as it was.
    One intra-op thread keeps the CPU reductions on one schedule and the run light.
    """
    comfy_dir = _comfy_dir()
    preserved = {name: module for name, module in sys.modules.items()
                 if _is_comfy_module(name)}
    for name in preserved:
        sys.modules.pop(name, None)
    path_before, argv_before = list(sys.path), list(sys.argv)
    threads_before = torch.get_num_threads()
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    try:
        sys.argv = ["pytest-ideogram4-stock-equivalence", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            ops = pytest.importorskip("comfy.ops")
            ideogram = pytest.importorskip("comfy.ldm.ideogram4.model")
        finally:
            sys.argv[:] = argv_before
        torch.set_num_threads(1)
        yield ops, ideogram
    finally:
        torch.set_num_threads(threads_before)
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or from_checkout(module, comfy_dir)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = path_before


def _model(ops, ideogram):
    """Comfy's model at toy size; every dimension a multiple of 16."""
    torch.manual_seed(4)
    model = ideogram.Ideogram4Transformer2DModel(
        in_channels=16, num_layers=2, num_attention_heads=2, attention_head_dim=16,
        intermediate_size=64, adaln_dim=16, llm_features_dim=32, rope_theta=10000.0,
        mrope_section=(4, 2, 2), dtype=torch.float32, device="cpu",
        operations=ops.disable_weight_init)
    with torch.no_grad():
        # disable_weight_init leaves every parameter uninitialized.
        for name, parameter in model.named_parameters():
            centre = 1.0 if "norm" in name else 0.0
            parameter.copy_(torch.randn_like(parameter) * 0.2 + centre)
    return model.eval()


def _inputs(text_rows, grid, batch):
    generator = torch.Generator().manual_seed(1000 * batch + 10 * (text_rows or 0) + grid[1])
    x = torch.randn(batch, 16, *grid, generator=generator)
    timesteps = torch.tensor([0.3, 0.65])[:batch]
    context = None
    if text_rows is not None:
        context = torch.randn(batch, text_rows, 32, generator=generator)
    return x, timesteps, context


def _attend(q, k, v):
    """One attention for both sides over (B, L, H, D), the same op sequence.

    Each score and each output row reduces over a fixed axis in float64, so the
    number of query rows in a call cannot change a row's bits.
    """
    q_h, k_h, v_h = (t.transpose(1, 2).to(torch.float64) for t in (q, k, v))
    scores = (q_h.unsqueeze(-2) * k_h.unsqueeze(-3)).sum(-1) / q_h.shape[-1] ** 0.5
    weights = torch.softmax(scores, dim=-1)
    out = (weights.unsqueeze(-1) * v_h.unsqueeze(-3)).sum(-2)
    return out.transpose(1, 2).to(q.dtype)


class _Kernel:
    """The fabric the stubs exchange through, and the operands each call saw."""

    def __init__(self, world: int) -> None:
        self.world = world
        self.fabric = _Fabric(world)
        self.calls: list[tuple] = []

    def attend(self, entry, q, k, v):
        self.calls.append((_rank(), entry, q.clone(), k.clone(), v.clone()))
        return _attend(q, k, v)


def _stock(model, x, timesteps, context):
    """Comfy's own `_forward`, its attention routed to `_attend` and recorded."""
    calls = []

    def override(_func, q, k, v, heads, mask=None, **kwargs):
        assert mask is None and kwargs.get("skip_reshape") is True
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, L, D) -> (B, L, H, D)
        calls.append((q.clone(), k.clone(), v.clone()))
        out = _attend(q, k, v)
        return out.reshape(out.shape[0], out.shape[1], -1)

    out = model._forward(x, timesteps, context, attention_mask=None,
                         transformer_options={"optimized_attention_override": override})
    return out, calls


def _sharded(monkeypatch, model, x, timesteps, context, *, pure_ulysses=True):
    """The adapter's bound forward, one thread and one model copy per rank."""
    kernel = _Kernel(WORLD)
    _install_stubs(monkeypatch, kernel)
    dispatch = base.make_usp_attention("TORCH_FLASH")
    drops: list = []

    def attn(*args, **kwargs):
        drops.append(kwargs.get("drop_rows"))
        return dispatch(*args, **kwargs)

    models = [copy.deepcopy(model) for _ in range(WORLD)]
    for copied in models:
        Ideogram4Adapter().inject_usp(copied, InjectionContext(
            topology_sp=WORLD, usp_attention=attn, pure_ulysses=pure_ulysses))
    outputs: list = [None] * WORLD
    errors: list = [None] * WORLD

    def body(rank):
        try:
            _LOCAL.rank = rank
            outputs[rank] = models[rank]._forward(
                x, timesteps, context, attention_mask=None, transformer_options={})
        except BaseException as exc:  # re-raised in the test body
            errors[rank] = exc
            kernel.fabric.barrier.abort()

    threads = [threading.Thread(target=body, args=(rank,)) for rank in range(WORLD)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=90)
    for error in errors:
        if error is not None:
            raise error
    return outputs, kernel, drops


def _gemm_bits_follow_row_count(model, batch, rows) -> bool:
    """Whether this CPU's GEMM gives a row other bits at the shard's row count.

    Checks every Linear the blocks run on a shard, on the same padded split the
    forward makes, so a skip names the untreated departure and nothing else.
    """
    local = -(-rows // WORLD)
    generator = torch.Generator().manual_seed(rows)
    with torch.no_grad():
        for module in model.layers.modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            padded = torch.randn(batch, local * WORLD, module.in_features, generator=generator)
            full = module(padded[:, :rows].contiguous())
            parts = torch.cat([module(part.contiguous()) for part in padded.chunk(WORLD, dim=1)],
                              dim=1)[:, :rows]
            if not torch.equal(full, parts):
                return True
    return False


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("text_rows,grid", CASES)
def test_every_kernel_sees_stock_operands_and_the_latent_is_stock(
    monkeypatch, comfy_ideogram4, text_rows, grid, batch
):
    ops, ideogram = comfy_ideogram4
    model = _model(ops, ideogram)
    rows = grid[0] * grid[1] + (text_rows or 0)
    if _gemm_bits_follow_row_count(model, batch, rows):
        pytest.skip("this CPU's GEMM changes a row's bits with the row count, which is the "
                    "untreated block-Linear departure, not the pad exclusion under test")
    x, timesteps, context = _inputs(text_rows, grid, batch)
    expected, stock_calls = _stock(model, x, timesteps, context)
    outputs, kernel, drops = _sharded(monkeypatch, model, x, timesteps, context)

    padded = rows % WORLD != 0
    local = -(-rows // WORLD)
    assert len(stock_calls) == len(model.layers)
    for rank in range(WORLD):
        calls = [call[1:] for call in kernel.calls if call[0] == rank]
        assert len(calls) == len(stock_calls)
        for (entry, q, k, v), (stock_q, stock_k, stock_v) in zip(calls, stock_calls, strict=True):
            assert tuple(k.shape) == tuple(stock_k.shape), "the kernel saw a key stock does not have"
            assert torch.equal(k, stock_k) and torch.equal(v, stock_v)
            # A padded stream reaches the kernel on the full axis past its pad;
            # an even one keeps xfuser's own call with this rank's queries.
            if entry == "xfuser":
                stock_q = stock_q[:, rank * local:(rank + 1) * local]
            assert torch.equal(q, stock_q)
            assert entry == ("exact" if padded else "xfuser")
        assert outputs[rank].shape == expected.shape
        assert (outputs[rank] - expected).abs().max().item() == 0.0
    # The pad is the tail of the one packed stream, and an even stream names none.
    assert all(drop == list(range(rows, local * WORLD)) for drop in drops)


@pytest.mark.parametrize("text_rows,grid", [(5, (2, 4)), (None, (3, 3))])
def test_without_pure_ulysses_the_pad_row_still_reaches_the_kernel(
    monkeypatch, comfy_ideogram4, text_rows, grid
):
    """The negative control on the real model, and ring's path: no drop rows
    are named, the kernel sees one key more than stock, and the latent moves.

    It also pins how the pad row acts on this model. Its modulation only
    scales and RMSNorm keeps zero at zero, so the first block's pad key and
    value are exactly zero, yet that key still takes softmax weight; its zero
    query takes the mean of the values, so the second block's pad key is not
    zero.
    """
    ops, ideogram = comfy_ideogram4
    model = _model(ops, ideogram)
    x, timesteps, context = _inputs(text_rows, grid, 1)
    expected, stock_calls = _stock(model, x, timesteps, context)
    outputs, kernel, drops = _sharded(monkeypatch, model, x, timesteps, context,
                                      pure_ulysses=False)
    rows = grid[0] * grid[1] + (text_rows or 0)
    assert drops and all(drop is None for drop in drops)
    assert all(call[3].shape[1] == rows + 1 for call in kernel.calls)
    assert stock_calls[0][1].shape[1] == rows
    assert (outputs[0] - expected).abs().max().item() > 0.0
    first, second = [call for call in kernel.calls if call[0] == 0][:2]
    assert torch.count_nonzero(first[3][:, rows]) == 0
    assert torch.count_nonzero(first[4][:, rows]) == 0
    assert torch.count_nonzero(second[3][:, rows]) > 0
