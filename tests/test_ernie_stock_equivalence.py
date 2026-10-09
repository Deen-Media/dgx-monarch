"""Ernie-Image under pure Ulysses produces exactly the stock ComfyUI output (real model, CPU).

This builds comfy's own `ErnieImageModel` at toy sizes in float32, runs its
stock forward once, then runs the adapter's bound forward on two rank threads.
Only the all-to-all, the sequence gather and the attention kernel are stubbed
(the harness of tests/test_flux_pad_exclusion.py). The stub kernel is comfy's
own `optimized_attention` on the full gathered rows, so the attention math is
stock's. Each kernel call records its q, k and v, which must equal stock's
call for the same block, and the output must equal stock's. The comfy-canary
job runs this file against comfy master, because the bound attention copies
stock's q/k preparation and no other test fails when upstream changes it.

Ernie's forward reads no reference latents, masks or guidance: model_base
`ErnieImage` passes only `c_crossattn`, and the block loop passes no mask. The
optional input covered here is `transformer_options["rope_options"]`.

On this rig's Linux ARM cores the CPU GEMMs give the same bits per row at
these row counts (checked 2026-10-06), so there "equal" means bit equality.
The canary runs on x86, whose BLAS may pick another kernel for the halved row
count, the GEMM-shape departure the adapter leaves open; there "equal" means
agreement to float32 rounding (allclose, rtol and atol 2e-5), which the pad
row's effect exceeds by orders of magnitude. The ring control asserts that an
attended pad row breaks that agreement: on this rig's ARM cores it asserts
unequal bits, and on x86 an error past the tolerance. The kernel names,
scales, eps, rotary rows and layouts carry no GEMM and stay exact everywhere,
and so does rank against rank. The fixture forces ``--cpu`` because
comfy.cli_args parses argv on import, and skips where ComfyUI is not
importable.
"""
from __future__ import annotations

import copy
import os
import platform
import sys
import threading

import pytest
import torch

from dgx_monarch.adapters import base
from dgx_monarch.adapters.base import InjectionContext
from dgx_monarch.adapters.ernie import ErnieAdapter
from module_location_helpers import from_checkout
from test_flux_pad_exclusion import _LOCAL, _Harness, _install_stubs, _rank

WORLD = 2
COMFY_DIR = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
HIDDEN, HEADS, TEXT_DIM, CHANNELS = 32, 2, 24, 4
# (latent grid, text rows): even joint, odd text, odd image, both odd.
SHAPES = [((3, 4), 4), ((3, 4), 5), ((3, 3), 4), ((3, 3), 5)]
ROPE_OPTIONS = {"scale_y": 2.0, "scale_x": 0.5, "shift_t": 1.0, "shift_y": 0.5, "shift_x": 1.5}
_EXACT_GEMM = platform.machine() == "aarch64"


def _same(candidate: torch.Tensor, reference: torch.Tensor) -> bool:
    """Equal bits where the GEMMs are known to agree per row, else float32 rounding."""
    if _EXACT_GEMM:
        return torch.equal(candidate, reference)
    return torch.equal(candidate, reference) or torch.allclose(
        candidate, reference, rtol=2e-5, atol=2e-5)


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def comfy_ernie():
    """Import real `comfy.ldm.ernie.model`, then leave sys.modules as found."""
    preserved = {n: m for n, m in sys.modules.items() if _is_comfy_module(n)}
    for name in preserved:
        sys.modules.pop(name, None)
    path_before = list(sys.path)
    argv_before = sys.argv
    if os.path.isdir(COMFY_DIR) and COMFY_DIR not in sys.path:
        sys.path.insert(0, COMFY_DIR)
    try:
        sys.argv = ["pytest-ernie-stock-equivalence", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            ops = pytest.importorskip("comfy.ops")
            ernie_model = pytest.importorskip("comfy.ldm.ernie.model")
        finally:
            sys.argv = argv_before
        yield ops, ernie_model
    finally:
        # Classify first, then pop: a namespace package's path re-resolves
        # through its parent in sys.modules while it is being read.
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or from_checkout(module, COMFY_DIR)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = path_before


def _tiny_ernie(comfy_ernie):
    ops, ernie_model = comfy_ernie
    torch.manual_seed(0)
    # A non-default eps, so a fused call with a hard-coded one cannot match stock's.
    model = ernie_model.ErnieImageModel(
        hidden_size=HIDDEN, num_attention_heads=HEADS, num_layers=2, ffn_hidden_size=64,
        in_channels=CHANNELS, out_channels=CHANNELS, patch_size=1, text_in_dim=TEXT_DIM,
        rope_theta=256, rope_axes_dim=(4, 6, 6), eps=1e-5, dtype=torch.float32, device="cpu",
        operations=ops.disable_weight_init,
    )
    # comfy builds state-dict parameters with torch.empty; a real load fills
    # them from a checkpoint, so a toy run fills them here.
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    return model


def _inputs(grid, text_rows, batch):
    torch.manual_seed(100 * batch + 10 * text_rows + grid[0] * grid[1])
    x = torch.randn(batch, CHANNELS, *grid)
    timesteps = torch.rand(batch)
    context = torch.randn(batch, text_rows, TEXT_DIM)
    return x, timesteps, context


class _StockKernelHarness(_Harness):
    """Runs comfy's attention kernel on the full rows stock would hand it."""

    def __init__(self, world, kernel):
        super().__init__(world, strict=False)
        self.kernel = kernel
        self.seen: dict[int, list] = {rank: [] for rank in range(world)}

    def attend(self, entry, q, k, v):
        rank = _rank()
        local = q.shape[1]
        if entry == "xfuser":
            # xfuser's own call holds only this rank's queries; gather them so
            # the kernel sees stock's full rows, then keep this rank's share.
            q = torch.cat(self.fabric.exchange(q), dim=1)
        batch, rows, heads, dim = q.shape
        flat = [t.reshape(batch, -1, heads * dim) for t in (q, k, v)]
        self.seen[rank].append((entry, *flat))
        out = self.kernel(*flat, heads).reshape(batch, rows, heads, dim)
        return out[:, rank * local:(rank + 1) * local] if entry == "xfuser" else out


def _plain(value):
    """The tensor behind comfy's AttentionTensorContainer, or the tensor itself."""
    return value if isinstance(value, torch.Tensor) else value.take()


def _stock(monkeypatch, comfy_ernie, model, inputs, options):
    """Stock's forward, recording every attention call's operands."""
    _ops, ernie_model = comfy_ernie
    kernel = ernie_model.optimized_attention
    calls: list[tuple] = []

    def record(q, k, v, heads, mask=None, **kwargs):
        assert mask is None
        # Comfy master wraps each operand in a single-owner container; take
        # the tensor once and hand the kernel plain tensors.
        q, k, v = (_plain(t) for t in (q, k, v))
        calls.append((q, k, v))
        return kernel(q, k, v, heads, mask=mask, **kwargs)

    monkeypatch.setattr(ernie_model, "optimized_attention", record)
    with torch.no_grad():
        out = model(*inputs, transformer_options=options)
    monkeypatch.setattr(ernie_model, "optimized_attention", kernel)
    return out, calls, kernel


def _sharded(monkeypatch, model, inputs, options, kernel, *, pure_ulysses=True):
    harness = _StockKernelHarness(WORLD, kernel)
    _install_stubs(monkeypatch, harness)
    dispatch = base.make_usp_attention("TORCH_FLASH")
    models = [copy.deepcopy(model) for _ in range(WORLD)]
    for rank_model in models:
        ErnieAdapter().inject_usp(rank_model, InjectionContext(
            topology_sp=WORLD, usp_attention=dispatch, pure_ulysses=pure_ulysses))
    outputs: list = [None] * WORLD
    errors: list = [None] * WORLD

    def body(rank):
        try:
            _LOCAL.rank = rank
            with torch.no_grad():
                outputs[rank] = models[rank](*inputs, transformer_options=options)
        except BaseException as exc:  # re-raised in the test body
            errors[rank] = exc
            harness.fabric.barrier.abort()

    threads = [threading.Thread(target=body, args=(rank,)) for rank in range(WORLD)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=90)
    for error in errors:
        if error is not None:
            raise error
    return outputs, harness, models


@pytest.mark.parametrize("rope", [False, True], ids=["stock-rope", "rope-options"])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("grid,text_rows", SHAPES)
def test_pure_ulysses_hands_every_kernel_stock_operands_and_returns_stock_bits(
    monkeypatch, comfy_ernie, grid, text_rows, batch, rope
):
    model = _tiny_ernie(comfy_ernie)
    inputs = _inputs(grid, text_rows, batch)
    options = {"rope_options": dict(ROPE_OPTIONS)} if rope else {}
    expected, stock_calls, kernel = _stock(monkeypatch, comfy_ernie, model, inputs, options)
    outputs, harness, _models = _sharded(monkeypatch, model, inputs, options, kernel)

    assert len(stock_calls) == len(model.layers)
    for rank in range(WORLD):
        seen = harness.seen[rank]
        assert len(seen) == len(stock_calls)
        for (_entry, q, k, v), (stock_q, stock_k, stock_v) in zip(seen, stock_calls, strict=True):
            # The same keys in the same order, and the same queries and values.
            assert _same(k, stock_k)
            assert _same(q, stock_q)
            assert _same(v, stock_v)
    assert torch.equal(outputs[0], outputs[1])
    assert outputs[0].shape == expected.shape
    assert _same(outputs[0], expected)


@pytest.mark.parametrize("grid,text_rows", [((3, 4), 5), ((3, 3), 4)])
def test_ring_and_hybrid_still_attend_the_pad_and_differ_from_stock(
    monkeypatch, comfy_ernie, grid, text_rows
):
    """The negative control on the real model: without pure Ulysses the odd
    joint stream's pad row reaches the kernel and the output moves, so the
    exact match above comes from the exclusion."""
    model = _tiny_ernie(comfy_ernie)
    inputs = _inputs(grid, text_rows, 1)
    expected, stock_calls, kernel = _stock(monkeypatch, comfy_ernie, model, inputs, {})
    outputs, harness, _models = _sharded(monkeypatch, model, inputs, {}, kernel,
                                         pure_ulysses=False)
    for rank in range(WORLD):
        for (_entry, _q, k, _v), (_sq, stock_k, _sv) in zip(
                harness.seen[rank], stock_calls, strict=True):
            assert k.shape[1] == stock_k.shape[1] + 1
    assert not _same(outputs[0], expected)


def _record_rope(monkeypatch, calls):
    """Wrap comfy-kitchen's two q/k kernels on the module object stock reads."""
    import comfy.quant_ops

    ck = comfy.quant_ops.ck
    for name in ("rms_rope_split_half", "apply_rope_split_half"):
        original = getattr(ck, name)

        def recorder(*args, _name=name, _original=original):
            calls.append((_rank(), _name, args))
            return _original(*args)

        monkeypatch.setattr(ck, name, recorder)


def _rope_run(monkeypatch, comfy_ernie, grid, text_rows, batch, *, pure_ulysses=True):
    model = _tiny_ernie(comfy_ernie)
    inputs = _inputs(grid, text_rows, batch)
    calls: list[tuple] = []
    _record_rope(monkeypatch, calls)
    expected, _attention, kernel = _stock(monkeypatch, comfy_ernie, model, inputs, {})
    stock_calls = list(calls)
    outputs, _harness, _models = _sharded(monkeypatch, model, inputs, {}, kernel,
                                          pure_ulysses=pure_ulysses)
    return model, expected, outputs, stock_calls, calls[len(stock_calls):]


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("grid,text_rows", [((3, 4), 4), ((3, 4), 5)])
def test_pure_ulysses_calls_stocks_fused_q_k_kernel_with_stocks_operands(
    monkeypatch, comfy_ernie, grid, text_rows, batch
):
    """On CPU the fused and unfused kernels produce identical outputs, so the output
    alone cannot tell them apart. The calls can: each rank must make stock's
    call per block, on its own rows of stock's operands."""
    model, expected, outputs, stock_calls, sharded_calls = _rope_run(
        monkeypatch, comfy_ernie, grid, text_rows, batch)
    assert [name for _rank_id, name, _args in stock_calls] == (
        ["rms_rope_split_half"] * len(model.layers))
    joint = grid[0] * grid[1] + text_rows
    local = -(-joint // WORLD)
    for rank in range(WORLD):
        mine = [(name, args) for rank_id, name, args in sharded_calls if rank_id == rank]
        assert [name for name, _args in mine] == ["rms_rope_split_half"] * len(model.layers)
        start = rank * local
        real = min(joint, start + local) - start
        for (_name, args), (_r, _n, stock_args) in zip(mine, stock_calls, strict=True):
            query, key, rotary, q_scale, k_scale, epsilon = args
            s_query, s_key, s_rotary, s_q_scale, s_k_scale, s_epsilon = stock_args
            assert query.shape[1] == key.shape[1] == rotary.shape[1] == local
            assert rotary.is_contiguous()
            assert torch.equal(rotary[:, :real], s_rotary[:, start:start + real])
            for mine_t, stock_t in ((query, s_query), (key, s_key)):
                assert _same(mine_t[:, :real], stock_t[:, start:start + real])
            assert torch.equal(q_scale, s_q_scale) and torch.equal(k_scale, s_k_scale)
            assert epsilon == s_epsilon
    assert torch.equal(outputs[0], outputs[1]) and _same(outputs[0], expected)


def test_training_runs_stocks_unfused_branch_on_both_sides(monkeypatch, comfy_ernie):
    import comfy.model_management

    monkeypatch.setattr(comfy.model_management, "in_training", True)
    _model, expected, outputs, stock_calls, sharded_calls = _rope_run(
        monkeypatch, comfy_ernie, (3, 4), 5, 1)
    assert {name for _r, name, _a in stock_calls} == {"apply_rope_split_half"}
    assert {name for _r, name, _a in sharded_calls} == {"apply_rope_split_half"}
    assert torch.equal(outputs[0], outputs[1]) and _same(outputs[0], expected)


def test_ring_and_hybrid_keep_the_unfused_q_k_branch(monkeypatch, comfy_ernie):
    _model, _expected, _outputs, stock_calls, sharded_calls = _rope_run(
        monkeypatch, comfy_ernie, (3, 4), 4, 1, pure_ulysses=False)
    assert {name for _r, name, _a in stock_calls} == {"rms_rope_split_half"}
    assert {name for _r, name, _a in sharded_calls} == {"apply_rope_split_half"}


@pytest.mark.parametrize("pure_ulysses", [True, False], ids=["pure-ulysses", "ring-hybrid"])
def test_batch_2_norms_read_stocks_contiguous_rows_under_pure_ulysses_only(
    monkeypatch, comfy_ernie, pure_ulysses
):
    """At batch 2 the token shard is a strided chunk and an odd stream's gather
    a strided narrow, where stock's first block norm and final norm read
    contiguous rows. Pure Ulysses copies to match; ring and hybrid keep the
    views. The hooks survive the per-rank deepcopy, so every rank records."""
    model = _tiny_ernie(comfy_ernie)
    seen: list[tuple[str, bool]] = []
    for name, module in (("first", model.layers[0].adaLN_sa_ln), ("final", model.final_norm.norm)):
        module.register_forward_pre_hook(
            lambda _module, args, _name=name: seen.append((_name, args[0].is_contiguous())))
    inputs = _inputs((3, 4), 5, 2)   # joint 17: odd, so the gather narrows
    _expected, _calls, kernel = _stock(monkeypatch, comfy_ernie, model, inputs, {})
    assert sorted(seen) == [("final", True), ("first", True)]
    seen.clear()
    _sharded(monkeypatch, model, inputs, {}, kernel, pure_ulysses=pure_ulysses)
    assert sorted(seen) == sorted([("final", pure_ulysses), ("first", pure_ulysses)] * WORLD)
