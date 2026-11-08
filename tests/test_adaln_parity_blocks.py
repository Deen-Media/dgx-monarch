"""Fused-norm parity for the Flux-family and Qwen Image 2.1 sharded blocks (issue #54).

comfy-kitchen's fused AdaLN picks its CUDA kernel by a call's rows, and on
2026-11-07 uly2 at 512 square read one-step NRMS 0.006 to 0.062 against one
GPU on Flux 1, Flux 2, Chroma and Qwen Image 2.1 for that reason, while 1024
square read bit-identical. The bound blocks fill each norm call to stock's row
count for its stream and keep the window's rows (adapters/adaln_parity_blocks.py).

The stand-in blocks below read their norm through a module global, as comfy's
``DoubleStreamBlock``, ``SingleStreamBlock`` and ``QwenImage21TransformerBlock``
do; the norm records each launch's rows and is row-local, so the filled call
must return the window's own values. The real-Comfy checks spawn a child that
imports the installed ComfyUI on the CPU and skip where none is named.
"""
from __future__ import annotations

import ast
import os
import sys
import types
from pathlib import Path

import pytest
import torch

from dgx_monarch.adapters import adaln_parity_blocks as parity
from dgx_monarch.adapters.base import InjectionContext
from dgx_monarch.adapters.flux_family import ChromaAdapter, Flux2Adapter, FluxAdapter, LongCatAdapter

LAUNCHES: list[int] = []


def modulated_norm(x, norm, scale, shift, modulation_dims=None):
    """Row-local, as comfy-kitchen's fused op is, and records the launch rows."""
    LAUNCHES.append(x.shape[1])
    return x * (1 + scale) + shift + norm.offset


def _modulated_norm(norm, x, scale, prefix_len, zero):
    """Qwen Image 2.1's shape: every row with the target scale, then the prefix redone."""
    s_prefix, s_target = scale
    LAUNCHES.append(x.shape[1])
    out = x * (1 + s_target) + zero
    if prefix_len:
        LAUNCHES.append(prefix_len)
        out[:, :prefix_len] = x[:, :prefix_len] * (1 + s_prefix) + zero
    return out


class _Norm:
    def __init__(self, offset: float) -> None:
        self.offset = offset


class _Double:
    """comfy's DoubleStreamBlock call pattern: two image and two text norms."""

    def __init__(self) -> None:
        self.img_norm1, self.img_norm2 = _Norm(1.0), _Norm(2.0)
        self.txt_norm1, self.txt_norm2 = _Norm(3.0), _Norm(4.0)

    def forward(self, img, txt, vec, pe, attn_mask=None, transformer_options=None):
        img = img + modulated_norm(img, self.img_norm1, vec, vec)
        txt = txt + modulated_norm(txt, self.txt_norm1, vec, vec)
        img = img + modulated_norm(img, self.img_norm2, vec, vec)
        txt = txt + modulated_norm(txt, self.txt_norm2, vec, vec)
        return img, txt

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _Single:
    def __init__(self) -> None:
        self.pre_norm = _Norm(5.0)

    def forward(self, x, vec, pe, attn_mask=None, transformer_options=None):
        return x + modulated_norm(x, self.pre_norm, vec, vec)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _Qwen21Block:
    def __init__(self) -> None:
        self.img_norm1 = _Norm(0.0)

    def forward(self, x, mod, pe, attn_fn, prefix_len, transformer_options=None):
        scale, zero = mod
        return x + _modulated_norm(self.img_norm1, x, scale, prefix_len, zero)


class _Unroutable:
    def forward(self, x, vec, pe, attn_mask=None, transformer_options=None):
        return x


@pytest.fixture(autouse=True)
def _clear_launches():
    LAUNCHES.clear()
    yield
    LAUNCHES.clear()


def test_a_bound_double_block_launches_each_norm_at_its_streams_stock_rows():
    block, plain = _Double(), _Double()
    assert parity.bind_flux_norm_parity(block) is True
    img, txt, vec = torch.randn(1, 3, 4), torch.randn(1, 2, 4), torch.randn(1, 1, 4)
    options = parity.with_norm_rows({}, img=6, txt=5)
    got = block(img.clone(), txt.clone(), vec, None, transformer_options=options)
    assert LAUNCHES == [6, 5, 6, 5]  # stock's image and text call sizes
    LAUNCHES.clear()
    want = plain(img.clone(), txt.clone(), vec, None, transformer_options={})
    assert LAUNCHES == [3, 2, 3, 2]  # an unbound block launches the window
    assert all(torch.equal(a, b) for a, b in zip(got, want, strict=True))
    for value in got:  # a fresh tensor's strides, as stock's output has
        assert value.is_contiguous()


def test_a_bound_single_block_launches_at_the_joint_rows_and_passes_through_without_them():
    block = _Single()
    parity.bind_flux_norm_parity(block)
    x, vec = torch.randn(2, 4, 3), torch.randn(2, 1, 3)
    routed = block(x, vec, None, transformer_options=parity.with_norm_rows({}, joint=9))
    unrouted = block(x, vec, None, transformer_options={})
    assert LAUNCHES == [9, 4]
    assert torch.equal(routed, unrouted)
    # A block_wrap patch may hand no options at all; the class forward runs.
    LAUNCHES.clear()
    assert torch.equal(block(x, vec, None, transformer_options=None), unrouted)
    assert LAUNCHES == [4]


def test_a_call_the_plan_does_not_name_fails_loudly():
    block = _Double()
    parity.bind_flux_norm_parity(block)
    with pytest.raises(AssertionError, match="ComfyUI changed the block"):
        block(torch.randn(1, 3, 4), torch.randn(1, 2, 4), torch.randn(1, 1, 4), None,
              transformer_options=parity.with_norm_rows({}, img=6))


def test_a_block_whose_class_reads_no_norm_global_is_left_unbound_and_logged(caplog):
    block = _Unroutable()
    assert parity.bind_flux_norm_parity(block) is False
    assert parity.bind_qwen21_norm_parity(block) is False
    assert parity.bind_flux_norm_parity(types.SimpleNamespace()) is False
    assert "forward" not in vars(block)
    single = _Single()
    with caplog.at_level("WARNING"):
        parity.bind_blocks([block, single], parity.bind_flux_norm_parity, "flux")
    assert "forward" in vars(single)
    assert "flux fused-norm parity: _Unroutable.forward reads no norm global" in caplog.text


def test_a_bound_qwen21_block_launches_the_prefix_and_whole_target_and_redoes_the_prefix():
    block, plain = _Qwen21Block(), _Qwen21Block()
    assert parity.bind_qwen21_norm_parity(block) is True
    x = torch.randn(1, 2 + 3, 4)  # a two-row prefix and a three-row target window
    mod = ((torch.randn(1, 1, 4), torch.randn(1, 1, 4)), torch.zeros(1, 1, 4))
    got = block.forward(x.clone(), mod, None, None, 2, parity.with_norm_rows({}, joint=2 + 6))
    assert LAUNCHES == [8, 2]  # stock's prefix plus whole target, then the prefix
    LAUNCHES.clear()
    want = plain.forward(x.clone(), mod, None, None, 2, {})
    assert LAUNCHES == [5, 2]
    assert torch.equal(got, want)
    LAUNCHES.clear()
    block.forward(x.clone(), mod, None, None, 2, {})
    assert LAUNCHES == [5, 2]  # no count, as on the cache-miss prefix fill: the class forward


class _FluxShaped:
    def __init__(self, blocks: list, singles: list) -> None:
        self.double_blocks, self.single_blocks = blocks, singles


@pytest.mark.parametrize("adapter_cls", [FluxAdapter, Flux2Adapter, LongCatAdapter, ChromaAdapter])
def test_the_flux_family_binds_its_blocks_under_sequence_parallelism_only(adapter_cls):
    sharded = _FluxShaped([_Double(), _Double()], [_Single(), _Unroutable()])
    adapter_cls().inject_usp(sharded, InjectionContext(topology_sp=2, usp_attention=None))
    bound = [("forward" in vars(block)) for block in (*sharded.double_blocks, *sharded.single_blocks)]
    assert bound == [True, True, True, False]
    # One GPU never reaches inject_usp, so its blocks keep the class forward.
    single = _FluxShaped([_Double()], [_Single()])
    assert not any("forward" in vars(block) for block in (*single.double_blocks, *single.single_blocks))


class _Qwen21Shaped:
    def __init__(self, blocks: list) -> None:
        self.transformer_blocks = blocks


def test_qwen21_binds_its_transformer_blocks_under_sequence_parallelism_only():
    from dgx_monarch.adapters.qwen_image21 import QwenImage21Adapter

    sharded = _Qwen21Shaped([_Qwen21Block(), _Qwen21Block()])
    QwenImage21Adapter().inject_usp(sharded, InjectionContext(topology_sp=2, usp_attention=None))
    assert all("forward" in vars(block) for block in sharded.transformer_blocks)
    single = _Qwen21Shaped([_Qwen21Block()])
    assert "forward" not in vars(single.transformer_blocks[0])


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def comfy_blocks():
    """Import the installed ComfyUI's block modules, then undo it.

    The snapshot-and-restore of tests/test_chroma_text_projections.py, so a
    later test that asks whether comfy imports gets the answer it gets alone.
    """
    preserved = {name: module for name, module in sys.modules.items() if _is_comfy_module(name)}
    for name in preserved:
        sys.modules.pop(name, None)
    original_path = list(sys.path)
    comfy_dir = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    original_argv = sys.argv
    try:
        sys.argv = ["pytest-adaln-parity-blocks", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            ops = pytest.importorskip("comfy.ops")
            flux_layers = pytest.importorskip("comfy.ldm.flux.layers")
            qwen21 = pytest.importorskip("comfy.ldm.qwen_image21.model")
        finally:
            sys.argv = original_argv
        yield ops, flux_layers, qwen21
    finally:
        for name in [name for name in sys.modules if _is_comfy_module(name)]:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = original_path


def test_the_installed_comfy_blocks_still_read_the_norm_globals(comfy_blocks):
    """A ComfyUI whose blocks stop reading these globals leaves them unrouted
    (logged at inject); this names the change wherever ComfyUI is installed."""
    from dgx_monarch.adapters.flux3_adaln_parity import loads_global

    ops, flux_layers, qwen21 = comfy_blocks
    assert loads_global(flux_layers.DoubleStreamBlock.forward, parity.NORM_GLOBAL)
    assert loads_global(flux_layers.SingleStreamBlock.forward, parity.NORM_GLOBAL)
    assert loads_global(qwen21.QwenImage21TransformerBlock.forward, parity.QWEN21_NORM_GLOBAL)
    # The norm modules the plan keys on are the ones the stock blocks build,
    # so every call a bound block makes finds its stream's count.
    operations = ops.disable_weight_init
    double = flux_layers.DoubleStreamBlock(8, 2, 2.0, operations=operations)
    single = flux_layers.SingleStreamBlock(8, 2, operations=operations)
    planned = parity._flux_plan(double, {"img": 6, "txt": 5}) + parity._flux_plan(single, {"joint": 11})
    assert [rows for _norm, rows in planned] == [6, 6, 5, 5, 11]
    assert parity.bind_flux_norm_parity(double) and parity.bind_flux_norm_parity(single)


def test_a_bound_real_flux_block_launches_at_stock_rows_and_keeps_its_output(comfy_blocks, monkeypatch):
    """comfy's own DoubleStreamBlock and SingleStreamBlock, bound: stock's code
    runs with its real ``modulated_norm`` sent through ``at_stock_rows``, the
    CPU ``ck.adaln`` takes the filled call, and the window's output is the
    unbound block's to the bit."""
    import comfy.quant_ops

    ops, flux_layers, _qwen21 = comfy_blocks
    kitchen = comfy.quant_ops.ck
    adaln = kitchen.adaln
    launches: list[int] = []

    def record(x, scale, shift, eps=1e-6):
        launches.append(int(x.shape[1]))
        return adaln(x, scale, shift, eps)

    monkeypatch.setattr(kitchen, "adaln", record)
    torch.manual_seed(7)
    operations = ops.disable_weight_init
    double = flux_layers.DoubleStreamBlock(8, 2, 2.0, operations=operations)
    single = flux_layers.SingleStreamBlock(8, 2, operations=operations)
    with torch.no_grad():
        for parameter in (*double.parameters(), *single.parameters()):
            parameter.normal_(mean=0.0, std=0.2)
    img, txt, vec = torch.randn(1, 3, 8), torch.randn(1, 2, 8), torch.randn(1, 8)
    joint = torch.cat((txt, img), dim=1)

    def run(double_options, single_options):
        with torch.no_grad():
            pair = double(img=img.clone(), txt=txt.clone(), vec=vec, pe=None,
                          transformer_options=double_options)
            return (*pair, single(joint.clone(), vec=vec, pe=None, transformer_options=single_options))

    want = run({}, {})
    assert launches == [3, 2, 3, 2, 5]
    launches.clear()
    assert parity.bind_flux_norm_parity(double) and parity.bind_flux_norm_parity(single)
    got = run(parity.with_norm_rows({}, img=6, txt=5), parity.with_norm_rows({}, joint=11))
    assert launches == [6, 5, 6, 5, 11]
    for value, reference in zip(got, want, strict=True):
        assert torch.equal(value, reference) or torch.allclose(value, reference, rtol=0, atol=1e-6)


def _named_comfy_dir() -> Path | None:
    for name in ("COMFY_DIR", "COMFYUI_DIR"):
        value = os.environ.get(name)
        if value and Path(value).is_dir():
            return Path(value)
    return None


def _gloo_qwen21_launch_worker(rank, comfy_dir, init_file, result_dir):
    """Every fused-norm launch on each rank holds stock's rows (real ComfyUI, two Gloo ranks)."""
    sys.path.insert(0, comfy_dir)
    sys.argv = ["pytest-adaln-parity-blocks", "--cpu"]
    import comfy.options

    comfy.options.args_parsing = True
    import comfy.ops
    import comfy.quant_ops
    import torch.distributed as dist
    from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel

    import dgx_monarch.adapters.base as base
    import dgx_monarch.adapters.qwen_image21 as q21

    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    base.sp_world = lambda: 2
    base.sp_rank = lambda: rank

    def gather(t, orig, dim=1):
        pieces = [torch.empty_like(t) for _ in range(2)]
        dist.all_gather(pieces, t.contiguous())
        return torch.cat(pieces, dim=dim).narrow(dim, 0, orig)

    q21.sp_gather = gather
    torch.manual_seed(31)
    model = QwenImage21Transformer2DModel(
        in_channels=64, out_channels=64, num_layers=2, attention_head_dim=6,
        num_attention_heads=1, context_in_dim=4, mlp_ratio=1,
        axes_dims_rope=(2, 2, 2), operations=comfy.ops.disable_weight_init,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    model.reset_prefix_cache(False)
    kitchen = comfy.quant_ops.ck
    adaln = kitchen.adaln
    launches: list[int] = []

    def record(x, scale, shift, eps=1e-6):
        launches.append(int(x.shape[1]))
        return adaln(x, scale, shift, eps)

    kitchen.adaln = record
    # Three text rows and five target rows, which pad to six over two ranks.
    x, t, context = torch.randn(1, 64, 1, 5), torch.tensor([0.5]), torch.randn(1, 3, 4)
    stock = model._forward(x, t, context, ref_latents=[], image_slots=[])
    stock_launches = list(launches)
    launches.clear()
    q21.QwenImage21Adapter().inject_usp(model, InjectionContext(
        topology_sp=2, usp_attention=types.SimpleNamespace(effective_kernel="TORCH_FLASH")))
    candidate = model._forward(x, t, context, ref_latents=[], image_slots=[])
    same = bool(torch.equal(candidate, stock) or torch.allclose(candidate, stock, rtol=0, atol=2e-6))
    Path(result_dir, f"rank{rank}").write_text(repr((stock_launches, list(launches), same)))
    dist.destroy_process_group()


def test_qwen21_every_rank_launches_each_fused_norm_at_stocks_rows_real_comfy():
    """Without the fill a rank's first norm call holds the prefix plus its
    three-row window, six rows, where stock's holds eight."""
    comfy_dir = _named_comfy_dir()
    if comfy_dir is None:
        pytest.skip("real Comfy canary requires COMFY_DIR or COMFYUI_DIR")
    import tempfile

    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_gloo_qwen21_launch_worker,
                 args=(str(comfy_dir), str(Path(directory) / "gloo-init"), directory),
                 nprocs=2, join=True)
        results = [ast.literal_eval(Path(directory, f"rank{rank}").read_text()) for rank in range(2)]
    for stock_launches, rank_launches, same in results:
        assert stock_launches[:4] == [8, 3, 8, 3]  # per block: prefix plus target, then prefix
        assert rank_launches == stock_launches
        assert same
