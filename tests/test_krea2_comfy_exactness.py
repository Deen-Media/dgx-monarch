"""Krea2 under pure Ulysses hands every kernel stock's operands (real Comfy, CPU).

This builds comfy's own `SingleStreamDiT` at toy sizes in float32, runs stock
`_forward` once, then runs the adapter's bound forward on two rank threads
through the real `base.make_usp_attention` dispatch and the real full-axis
call. Only the all-to-all, the sequence gather and the kernel are stubbed (the
harness of tests/test_flux_pad_exclusion.py); the stub all-to-all keeps every
head on every rank, so each rank's kernel call has stock's shape.

Stock and both ranks attend through one kernel, comfy's CPU
`scaled_dot_product_attention` on contiguous (B, H, L, D) tensors, reached
through an `optimized_attention_override` that records what it was handed. So
the claims are:

* every joint-block kernel call receives q, k and v bit-equal to stock's, which
  is stock's key set (no pad rows) in stock's order;
* every text-path attention and every text-path Linear receives stock's
  operands: all text rows, bit-equal to stock's, in stock's shape and strides;
* each rank's output equals stock's, max abs difference 0.0.

The joint blocks and the final layer run their Linears on local rows, and the
joint q, k and v come out of those Linears. So the joint operand checks and the
output equality both need a CPU GEMM that gives each row the same bits at any
row count, and no precondition here checks it. The text-path checks hold on any
CPU, because that path runs on all text rows before the shard. Whether cuBLAS
keeps those bits at those row counts is a hardware question this file cannot
answer. Skips where ComfyUI is not importable.
"""
from __future__ import annotations

import copy
import os
import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.adapters import base
from dgx_monarch.adapters.base import InjectionContext
from dgx_monarch.adapters.krea2 import Krea2Adapter
from module_location_helpers import from_checkout
from test_flux_pad_exclusion import _LOCAL, _Harness, _install_stubs, _rank

WORLD = 2
FEATURES, HEADS, KV_HEADS, BLOCKS = 64, 4, 2, 2
TXT_DIM, TXT_HEADS, TXT_LAYERS = 32, 2, 3
CHANNELS, PATCH = 4, 2
# (text rows, latent height and width): even text and even image (2x4), odd
# text, odd image (3x3), and both odd with a latent the patch pads and crops.
SHAPES = [(4, (4, 8)), (5, (4, 8)), (4, (6, 6)), (5, (5, 6))]
TEXT_PREFIXES = ("txtfusion.", "txtmlp.")


def _comfy_dir() -> str:
    return os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def comfy_krea2():
    """Import real comfy on its CPU path, then put sys.modules back as found."""
    preserved = {n: m for n, m in sys.modules.items() if _is_comfy_module(n)}
    for name in preserved:
        sys.modules.pop(name, None)
    original_path, original_argv = list(sys.path), sys.argv
    comfy_dir = _comfy_dir()
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    try:
        sys.argv = ["pytest-krea2-comfy-exactness", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            ops = pytest.importorskip("comfy.ops")
            model_module = pytest.importorskip("comfy.ldm.krea2.model")
        finally:
            sys.argv = original_argv
        yield ops, model_module
    finally:
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or from_checkout(module, comfy_dir)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = original_path


def _kernel(ops, q, k, v):
    """The one attention every side runs: comfy's CPU SDPA on (B, H, L, D)."""
    return ops.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())


class _Recorder:
    """Stands in for native attention; records per rank what it was handed."""

    def __init__(self, ops, *, per_rank: bool) -> None:
        self.ops, self.per_rank = ops, per_rank
        self.calls: dict[int, list] = {}
        self.lock = threading.Lock()

    def __call__(self, _func, q, k, v, heads, mask=None, **kwargs):
        assert mask is None and kwargs.get("skip_reshape") and not kwargs.get("skip_output_reshape")
        joint = "block_index" in kwargs.get("transformer_options", {})
        with self.lock:
            self.calls.setdefault(_rank() if self.per_rank else 0, []).append(
                ("joint" if joint else "text", q.clone(), k.clone(), v.clone()))
        out = _kernel(self.ops, q, k, v)
        return out.transpose(1, 2).reshape(q.shape[0], -1, heads * q.shape[-1])


class _KernelHarness(_Harness):
    """The flux harness with the real kernel in place of the tag-sorting one."""

    def __init__(self, world: int, ops) -> None:
        super().__init__(world, strict=False)
        self.ops = ops
        self.kernel_calls: dict[int, list] = {}
        self.lock = threading.Lock()

    def attend(self, entry, q, k, v):  # (B, L, H, D) over the whole axis
        with self.lock:
            self.kernel_calls.setdefault(_rank(), []).append(
                (entry, q.clone(), k.clone(), v.clone()))
        out = _kernel(self.ops, *(t.transpose(1, 2) for t in (q, k, v)))
        return out.transpose(1, 2)


def _linear_spy(model, seen: list) -> None:
    """Record every text Linear input in call order: (name, shape, strides) and a copy."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.startswith(TEXT_PREFIXES):
            def hook(_module, args, name=name):
                seen.append(((name, tuple(args[0].shape), args[0].stride()), args[0].clone()))
            module.register_forward_pre_hook(hook)


def _toy(comfy_krea2):
    ops, model_module = comfy_krea2
    torch.manual_seed(2026)
    model = model_module.SingleStreamDiT(
        features=FEATURES, tdim=32, txtdim=TXT_DIM, heads=HEADS, kvheads=KV_HEADS,
        multiplier=4, layers=BLOCKS, patch=PATCH, channels=CHANNELS,
        txtlayers=TXT_LAYERS, txtheads=TXT_HEADS, txtkvheads=TXT_HEADS,
        dtype=torch.float32, device="cpu", operations=ops.disable_weight_init)
    # comfy builds parameters with torch.empty; a checkpoint load fills them.
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, mean=0.0, std=0.05)
    return model


def _inputs(text_rows, latent_hw, batch):
    generator = torch.Generator().manual_seed(text_rows * 100 + latent_hw[0] * 10 + batch)
    x = torch.randn(batch, CHANNELS, *latent_hw, generator=generator)
    timesteps = torch.tensor([0.3, 0.7])[:batch]
    context = torch.randn(batch, text_rows, TXT_LAYERS * TXT_DIM, generator=generator)
    return x, timesteps, context


def _run(monkeypatch, comfy_krea2, model, args, kwargs, *, pure_ulysses):
    """Stock once on `model`, then the bound forward on a copy per rank."""
    ops, _ = comfy_krea2
    ranks = [copy.deepcopy(model) for _ in range(WORLD)]
    stock_linears: list = []
    _linear_spy(model, stock_linears)
    stock_attention = _Recorder(ops, per_rank=False)
    with torch.no_grad():
        stock = type(model)._forward(
            model, *args, transformer_options={"optimized_attention_override": stock_attention},
            **kwargs)

    harness = _KernelHarness(WORLD, ops)
    _install_stubs(monkeypatch, harness)
    rank_attention = _Recorder(ops, per_rank=True)
    kernel = base.make_usp_attention("TORCH_FLASH")
    rank_linears: list[list] = [[] for _ in range(WORLD)]
    for rank, replica in enumerate(ranks):
        _linear_spy(replica, rank_linears[rank])
        Krea2Adapter().inject_usp(replica, InjectionContext(
            topology_sp=WORLD, usp_attention=kernel, pure_ulysses=pure_ulysses))
    outputs: list = [None] * WORLD
    errors: list = [None] * WORLD

    def body(rank):
        try:
            _LOCAL.rank = rank
            with torch.no_grad():
                outputs[rank] = ranks[rank]._forward(
                    *args, transformer_options={"optimized_attention_override": rank_attention},
                    **kwargs)
        except BaseException as exc:  # re-raised in the test body
            errors[rank] = exc
            harness.fabric.barrier.abort()

    threads = [threading.Thread(target=body, args=(rank,)) for rank in range(WORLD)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    for error in errors:
        if error is not None:
            raise error
    return SimpleNamespace(
        stock=stock, outputs=outputs, harness=harness,
        stock_text=[call[1:] for call in stock_attention.calls[0] if call[0] == "text"],
        stock_joint=[call[1:] for call in stock_attention.calls[0] if call[0] == "joint"],
        stock_text_linears=stock_linears,
        rank_text=[[call[1:] for call in rank_attention.calls.get(rank, [])]
                   for rank in range(WORLD)],
        rank_text_linears=rank_linears,
    )


def _layouts(linears):
    return [layout for layout, _values in linears]


def _same(left, right) -> bool:
    return left.shape == right.shape and torch.equal(left, right)


def _joint_calls(run, rank):
    """The kernel calls whose row count is the length of stock's joint axis."""
    rows = run.stock_joint[0][1].shape[2]
    return [call for call in run.harness.kernel_calls.get(rank, []) if call[2].shape[1] == rows]


def _scenario(monkeypatch, comfy_krea2, text_rows, latent_hw, batch, kwargs=None,
              *, pure_ulysses=True):
    model = _toy(comfy_krea2)
    run = _run(monkeypatch, comfy_krea2, model, _inputs(text_rows, latent_hw, batch),
               kwargs or {}, pure_ulysses=pure_ulysses)
    assert len(run.stock_text) == 4 and len(run.stock_joint) == BLOCKS
    assert run.stock.shape == (batch, CHANNELS, *latent_hw)
    return run


def _assert_joint_operands_are_stock(run):
    """Every joint kernel call: stock's q, k and v, so stock's keys in stock's order."""
    for rank in range(WORLD):
        calls = _joint_calls(run, rank)
        assert len(calls) == BLOCKS, rank
        for (entry, *operands), expected in zip(calls, run.stock_joint, strict=True):
            assert entry == "exact", rank
            for got, want in zip(operands, expected, strict=True):
                assert _same(got, want.transpose(1, 2)), rank


def _assert_text_operands_are_stock(run):
    """Every text attention runs natively; it and every text Linear get stock's operands."""
    for rank in range(WORLD):
        assert _layouts(run.rank_text_linears[rank]) == _layouts(run.stock_text_linears), rank
        assert all(_same(got, want) for (_g, got), (_w, want)
                   in zip(run.rank_text_linears[rank], run.stock_text_linears, strict=True)), rank
        assert len(run.rank_text[rank]) == len(run.stock_text), rank
        for got_call, want_call in zip(run.rank_text[rank], run.stock_text, strict=True):
            assert all(_same(got, want) for got, want in zip(got_call, want_call, strict=True)), rank
        # No text attention goes through the sequence-parallel kernel.
        assert len(run.harness.kernel_calls.get(rank, [])) == BLOCKS, rank


def _assert_output_is_stock(run):
    for out in run.outputs:
        assert torch.equal(out, run.stock)
        assert (out - run.stock).abs().max().item() == 0.0


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("text_rows,latent_hw", SHAPES)
def test_joint_kernels_receive_stock_operands(monkeypatch, comfy_krea2, text_rows, latent_hw, batch):
    _assert_joint_operands_are_stock(
        _scenario(monkeypatch, comfy_krea2, text_rows, latent_hw, batch))


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("text_rows,latent_hw", SHAPES)
def test_text_path_receives_stock_operands(monkeypatch, comfy_krea2, text_rows, latent_hw, batch):
    _assert_text_operands_are_stock(
        _scenario(monkeypatch, comfy_krea2, text_rows, latent_hw, batch))


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("text_rows,latent_hw", SHAPES)
def test_each_rank_output_equals_stock(monkeypatch, comfy_krea2, text_rows, latent_hw, batch):
    _assert_output_is_stock(_scenario(monkeypatch, comfy_krea2, text_rows, latent_hw, batch))


@pytest.mark.parametrize("batch", [1, 2])
def test_inputs_stock_ignores_stay_ignored(monkeypatch, comfy_krea2, batch):
    """An attention mask and reference latents with no method are inert in stock.

    Krea2 takes no guidance and comfy builds it no mask; a reference latent
    without a resolved method never enters stock's forward. Passing both here
    checks the sharded forward ignores them the same way.
    """
    kwargs = {"attention_mask": torch.ones(batch, 5),
              "ref_latents": [torch.randn(batch, CHANNELS, 4, 4)]}
    run = _scenario(monkeypatch, comfy_krea2, 5, (5, 6), batch, kwargs)
    _assert_joint_operands_are_stock(run)
    _assert_text_operands_are_stock(run)
    _assert_output_is_stock(run)


@pytest.mark.parametrize("text_rows,latent_hw", SHAPES)
def test_without_pure_ulysses_the_operands_are_not_stock(
    monkeypatch, comfy_krea2, text_rows, latent_hw
):
    """The ring and hybrid route keeps rank-major keys and half-row text.

    This is the negative control for the operand checks above.
    """
    run = _scenario(monkeypatch, comfy_krea2, text_rows, latent_hw, 1, pure_ulysses=False)
    calls = _joint_calls(run, 0)
    assert len(calls) == BLOCKS
    assert not any(torch.equal(call[2], want[1].transpose(1, 2))
                   for call, want in zip(calls, run.stock_joint, strict=True))
    assert _layouts(run.rank_text_linears[0]) != _layouts(run.stock_text_linears)
