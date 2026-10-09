"""Hunyuan's text-stream Linears run on full rows under pure Ulysses.

Stock runs the four text Linears of each double block (`txt_attn.qkv`,
`txt_attn.proj`, `txt_mlp[0]`, `txt_mlp[2]`) on every text row. A shard runs
them on half, and a BLAS that picks its kernel by row count changes the bits.
The forward wraps them through chroma_text's helper with Hunyuan's selector, so
each sees all text rows in stock's layout and its output is re-sharded.
`txt_attn.proj` reads a slice of the joint `[text, image]` attention output,
so its input takes the joint batch stride. The forward runs through the toy
harness of tests/test_hunyuan_order.py.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.adapters import hunyuan
from test_hunyuan_order import (  # noqa: F401  (autouse fixture re-exported)
    DIM,
    WORLD,
    _HunyuanDouble,
    _inputs,
    _run,
    comfy_layers_stub,
)

NAMES = ("qkv", "proj", "mlp0", "mlp2")


class _SpyLinear(torch.nn.Linear):
    def __init__(self):
        super().__init__(DIM, DIM, dtype=torch.float64)
        self.seen = []

    def forward(self, value):
        self.seen.append((value.shape[1], value.stride()))
        return super().forward(value)


def _module(block, name):
    return {"qkv": lambda: block.txt_attn.qkv, "proj": lambda: block.txt_attn.proj,
            "mlp0": lambda: block.txt_mlp[0], "mlp2": lambda: block.txt_mlp[2]}[name]()


class _LinearDouble(_HunyuanDouble):
    """The toy double block plus Hunyuan's four text Linears, each called once."""

    def __init__(self, index):
        super().__init__(index)
        self.txt_attn = SimpleNamespace(qkv=_SpyLinear(), proj=_SpyLinear())
        self.txt_mlp = [_SpyLinear(), None, _SpyLinear()]

    def __call__(self, img, txt, *rest, **kwargs):
        for name in NAMES:
            _module(self, name)(txt)
        return super().__call__(img, txt, *rest, **kwargs)


def test_selection_covers_the_four_text_linears_and_skips_non_linears():
    block = _LinearDouble(0)
    selected = hunyuan._text_modules(block)
    assert [module for module, _ in selected] == [_module(block, name) for name in NAMES]
    assert [joint for _, joint in selected] == [False, True, False, False]

    class MixedPrecisionLinear(torch.nn.Module):
        """Stands in for comfy's mixed_precision_ops Linear."""

    block.txt_attn.qkv = MixedPrecisionLinear()
    block.txt_mlp[2] = MixedPrecisionLinear()
    assert [module for module, _ in hunyuan._text_modules(block)] == [
        block.txt_attn.proj, block.txt_mlp[0]]
    assert hunyuan._text_modules(SimpleNamespace()) == ()


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("qwen_rows,byt5_rows,grid", [(4, 2, (2, 4)), (3, 2, (3, 3)),
                                                      (5, 0, (2, 3))])
def test_forward_runs_every_text_linear_on_full_rows_under_pure_ulysses_only(
    monkeypatch, qwen_rows, byt5_rows, grid, batch
):
    inputs = _inputs(qwen_rows, byt5_rows, grid, batch)
    text_rows, image_rows = qwen_rows + byt5_rows, grid[0] * grid[1]
    local = -(-text_rows // WORLD)
    for pure, expected_rows in ((True, text_rows), (False, local)):
        _outputs, harness = _run(monkeypatch, inputs, text_rows + image_rows, world=WORLD,
                                 pure_ulysses=pure, double_cls=_LinearDouble)
        for model in harness.models:
            for block in model.double_blocks:
                for name in NAMES:
                    module = _module(block, name)
                    assert [rows for rows, _ in module.seen] == [expected_rows], (name, pure)
                    assert "forward" not in vars(module)
                    if pure and name == "proj":
                        # Stock slices text out of the joint attention output.
                        assert module.seen[0][1] == ((text_rows + image_rows) * DIM, DIM, 1)


@pytest.mark.parametrize("batch", [1, 2])
def test_proj_stride_counts_the_reference_rows(monkeypatch, batch):
    """Reference rows join the image stream, so stock's joint output holds
    them too and the stride `txt_attn.proj` sees counts them."""
    inputs = _inputs(3, 2, (2, 3), batch, ref_grid=(1, 3))
    _outputs, harness = _run(monkeypatch, inputs, 5 + 3 + 6, world=WORLD, pure_ulysses=True,
                             double_cls=_LinearDouble)
    for model in harness.models:
        for block in model.double_blocks:
            assert block.txt_attn.proj.seen == [(5, ((5 + 3 + 6) * DIM, DIM, 1))]


@pytest.mark.parametrize("batch", [1, 2])
def test_a_replaced_double_block_still_runs_full_text_rows(monkeypatch, batch):
    """`patches_replace["dit"]` swaps the block call, not the projection patch."""
    inputs = _inputs(3, 2, (3, 3), batch)
    replaced = []

    def replace(args, extra):
        replaced.append(args["transformer_options"]["block_index"])
        return extra["original_block"](args)

    inputs["transformer_options"] = {"patches_replace": {"dit": {("double_block", 1): replace}}}
    _outputs, harness = _run(monkeypatch, inputs, 5 + 9, world=WORLD, pure_ulysses=True,
                             double_cls=_LinearDouble)
    assert replaced == [1] * WORLD
    for model in harness.models:
        for block in model.double_blocks:
            for name in NAMES:
                module = _module(block, name)
                assert [rows for rows, _ in module.seen] == [5], (name, block.index)
                assert "forward" not in vars(module)
