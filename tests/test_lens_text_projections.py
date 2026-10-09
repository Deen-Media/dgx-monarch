"""CPU contracts for Lens's full-row text projections (pure Ulysses).

Stock runs each block's text-stream projections on every text row: `txt_qkv`,
`to_add_out` and the SwiGLU `txt_mlp` (`w1`, `w3`, `w2`). On a rank's 1/world
of the rows the BLAS picks another kernel, and the bits change.
The forward wraps them through chroma_text's shared helper, so each sees all
text rows in stock's layout and its output is re-sharded; `txt_mlp` is wrapped
whole, so its three Linears see full rows behind one gather. The first tests
drive the helper on one block, the next drive the real bound forward, and the
last builds Lens's real block from comfy's Linear families, skipping where
ComfyUI is not importable. That fixture turns on comfy's argv parsing with argv
set to ``--cpu``, so the import reads no pytest flags and comfy selects the CPU.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.adapters import chroma_text, lens
from dgx_monarch.adapters.base import pad_seq_to_multiple
from test_lens_order import (  # noqa: F401  (fixture re-exported)
    DIM,
    WORLD,
    _LensToyBlock,
    _run,
    _streams,
    comfy_lens_stub,
)

NAMES = ("txt_qkv", "to_add_out", "w1", "w3", "w2")
HIDDEN = 6


class _GateMLP(torch.nn.Module):
    """comfy's Lens GateMLP: `w2(silu(w1(x)) * w3(x))`, at a toy width."""

    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Linear(DIM, HIDDEN, bias=False, dtype=torch.float64)
        self.w2 = torch.nn.Linear(HIDDEN, DIM, bias=False, dtype=torch.float64)
        self.w3 = torch.nn.Linear(DIM, HIDDEN, bias=False, dtype=torch.float64)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x), inplace=True).mul_(self.w3(x)))


def _block():
    return SimpleNamespace(
        attn=SimpleNamespace(txt_qkv=torch.nn.Linear(DIM, DIM, dtype=torch.float64),
                             to_add_out=torch.nn.Linear(DIM, DIM, dtype=torch.float64)),
        txt_mlp=_GateMLP(),
    )


def _module(block, name):
    return getattr(block.attn, name) if name in ("txt_qkv", "to_add_out") else getattr(
        block.txt_mlp, name)


def _real_strides(stride, shape):
    return tuple(step for step, size in zip(stride, shape, strict=True) if size != 1)


def _record(block):
    seen = {name: [] for name in NAMES}
    for name in NAMES:
        module = _module(block, name)
        original = module.forward

        def spy(value, name=name, original=original):
            seen[name].append((value.shape[1], value.stride(), value.is_contiguous(),
                               tuple(value.shape)))
            return original(value)
        module.forward = spy
    return seen


def _run_helper(monkeypatch, block, source, world, image_rows):
    """Each rank's output of every wrapped projection, joined and trimmed."""
    padded, text_rows = pad_seq_to_multiple(source, world, dim=1)
    chunks = list(torch.chunk(padded, world, dim=1))
    outputs = {"txt_qkv": [], "to_add_out": [], "txt_mlp": []}
    for rank in range(world):
        def gather(value, length, dim=1, rank=rank):
            assert length == text_rows and dim == 1 and torch.equal(value, chunks[rank])
            return padded[:, :length]
        monkeypatch.setattr(chroma_text, "sp_gather", gather)
        monkeypatch.setattr(chroma_text, "sp_rank", lambda rank=rank: rank)
        monkeypatch.setattr(chroma_text, "sp_world", lambda: world)
        with chroma_text.full_row_text_projections(
                block, text_rows, image_rows, pure_ulysses=True, select=lens._text_modules):
            outputs["txt_qkv"].append(block.attn.txt_qkv(chunks[rank]))
            outputs["to_add_out"].append(block.attn.to_add_out(chunks[rank]))
            outputs["txt_mlp"].append(block.txt_mlp(chunks[rank]))
    return {name: torch.cat(parts, dim=1)[:, :text_rows] for name, parts in outputs.items()}


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("rows", [1, 14, 27, 31, 61, 62])
def test_every_projection_sees_full_rows_in_stock_layout(monkeypatch, batch, world, rows):
    torch.manual_seed(batch * 1000 + rows)
    block, image_rows = _block(), 17
    originals = {name: _module(block, name).forward for name in NAMES}
    stock_mlp = _GateMLP()
    stock_mlp.load_state_dict(block.txt_mlp.state_dict())
    seen = _record(block)
    source = torch.randn(batch, rows, DIM, dtype=torch.float64)
    actual = _run_helper(monkeypatch, block, source, world, image_rows)
    # Stock slices text out of the joint [image, text] attention output.
    stock_joint = torch.empty(batch, image_rows + rows, DIM, dtype=torch.float64)
    stock_view = stock_joint[:, image_rows:]
    stock_view.copy_(source)
    assert torch.equal(actual["txt_qkv"], originals["txt_qkv"](source))
    assert torch.equal(actual["to_add_out"], originals["to_add_out"](stock_view))
    assert torch.equal(actual["txt_mlp"], stock_mlp(source))
    for name in NAMES:
        assert len(seen[name]) == world, name
        for seen_rows, stride, contiguous, seen_shape in seen[name]:
            assert seen_rows == rows, name
            if name == "to_add_out":
                assert stride == stock_view.stride() and contiguous is stock_view.is_contiguous()
            else:
                assert contiguous, name
                width = HIDDEN if name == "w2" else DIM
                # A size-1 batch axis keeps the padded gather's stride, which
                # no GEMM reads, so only the strides of real axes must match.
                assert _real_strides(stride, seen_shape) == _real_strides(
                    (rows * width, width, 1), seen_shape), name


def test_selection_covers_qkv_out_and_the_whole_mlp_and_skips_non_linears():
    block = _block()
    selected = lens._text_modules(block)
    assert [module for module, _ in selected] == [
        block.attn.txt_qkv, block.attn.to_add_out, block.txt_mlp]
    assert [joint for _, joint in selected] == [False, True, False]

    class MixedPrecisionLinear(torch.nn.Module):
        """Stands in for comfy's mixed_precision_ops Linear."""
        def forward(self, value):
            return value
    block.attn.txt_qkv = MixedPrecisionLinear()
    block.txt_mlp.w2 = MixedPrecisionLinear()
    assert [module for module, _ in lens._text_modules(block)] == [block.attn.to_add_out]
    assert lens._text_modules(SimpleNamespace()) == ()


def test_plain_fp8_cast_linear_takes_full_rows(monkeypatch):
    class CastLinear(torch.nn.Linear):
        """Like comfy's manual_cast Linear: casts a stored fp8 weight per call."""
        def forward(self, value):
            bias = None if self.bias is None else self.bias.to(value.dtype)
            return torch.nn.functional.linear(value, self.weight.to(value.dtype), bias)

    def cast(module):
        out = CastLinear(module.in_features, module.out_features, bias=module.bias is not None,
                         dtype=torch.float64)
        out.weight = torch.nn.Parameter(module.weight.detach().to(torch.float8_e4m3fn),
                                        requires_grad=False)
        return out

    block = _block()
    block.attn.txt_qkv, block.attn.to_add_out = cast(block.attn.txt_qkv), cast(block.attn.to_add_out)
    for name in ("w1", "w2", "w3"):
        setattr(block.txt_mlp, name, cast(getattr(block.txt_mlp, name)))
    assert len(lens._text_modules(block)) == 3
    seen = _record(block)
    _run_helper(monkeypatch, block, torch.randn(1, 5, DIM, dtype=torch.float64), world=2,
                image_rows=3)
    assert all(len(calls) == 2 and all(call[0] == 5 for call in calls) for calls in seen.values())


class _SpyLinear(torch.nn.Linear):
    def __init__(self, width_in=DIM, width_out=DIM):
        super().__init__(width_in, width_out, dtype=torch.float64)
        self.seen = []

    def forward(self, value):
        self.seen.append((value.shape[1], value.stride(), value.is_contiguous()))
        return super().forward(value)


class _SpyGateMLP(_GateMLP):
    def __init__(self):
        super().__init__()
        self.w1, self.w2, self.w3 = _SpyLinear(DIM, HIDDEN), _SpyLinear(HIDDEN, DIM), _SpyLinear(
            DIM, HIDDEN)


class _LinearBlock(_LensToyBlock):
    """The toy attention block plus Lens's text projections, each called once."""

    def __init__(self, index):
        super().__init__(index)
        self.attn = SimpleNamespace(txt_qkv=_SpyLinear(), to_add_out=_SpyLinear())
        self.txt_mlp = _SpyGateMLP()

    def __call__(self, hidden_states, encoder_hidden_states, *rest, **kwargs):
        self.attn.txt_qkv(encoder_hidden_states)
        self.txt_mlp(encoder_hidden_states)
        # Stock hands to_add_out the text slice of the joint attention output.
        joint = torch.cat((hidden_states, encoder_hidden_states), dim=1)
        self.attn.to_add_out(joint[:, hidden_states.shape[1]:])
        return super().__call__(hidden_states, encoder_hidden_states, *rest, **kwargs)


def _modules_of(block):
    return {"txt_qkv": block.attn.txt_qkv, "to_add_out": block.attn.to_add_out,
            "w1": block.txt_mlp.w1, "w3": block.txt_mlp.w3, "w2": block.txt_mlp.w2,
            "txt_mlp": block.txt_mlp}


@pytest.mark.usefixtures("comfy_lens_stub")
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("grid,text_rows", [((2, 4), 4), ((3, 3), 5), ((2, 4), 7)])
def test_forward_wraps_every_block_under_pure_ulysses_only(monkeypatch, grid, text_rows, batch):
    image, text = _streams(grid, text_rows, batch)
    local = -(-text_rows // WORLD)
    for pure, expected_rows in ((True, text_rows), (False, local)):
        _outputs, harness = _run(monkeypatch, image, text, grid, world=WORLD,
                                 pure_ulysses=pure, block_cls=_LinearBlock)
        for model in harness.models:
            for block in model.transformer_blocks:
                for name, module in _modules_of(block).items():
                    if name != "txt_mlp":
                        assert [rows for rows, *_ in module.seen] == [expected_rows], (name, pure)
                    assert "forward" not in vars(module), (name, pure)
                if pure:
                    _rows, stride, _contiguous = block.attn.to_add_out.seen[0]
                    # Stock's joint output spans all image and all text rows.
                    assert stride == ((image.shape[1] + text_rows) * DIM, DIM, 1)


@pytest.mark.usefixtures("comfy_lens_stub")
@pytest.mark.parametrize("batch", [1, 2])
def test_a_replaced_double_block_still_runs_full_text_rows(monkeypatch, batch):
    """`patches_replace["dit"]` swaps the block call, not the projection patch."""
    grid, text_rows = (3, 3), 5
    image, text = _streams(grid, text_rows, batch)
    replaced = []

    def replace(args, extra):
        replaced.append(args["transformer_options"]["block_index"])
        return extra["original_block"](args)

    options = {"patches_replace": {"dit": {("double_block", 1): replace}}}
    _outputs, harness = _run(monkeypatch, image, text, grid, world=WORLD, pure_ulysses=True,
                             block_cls=_LinearBlock,
                             forward_kwargs={"transformer_options": options})
    assert replaced == [1] * WORLD
    for model in harness.models:
        for block in model.transformer_blocks:
            for name, module in _modules_of(block).items():
                if name != "txt_mlp":
                    assert [rows for rows, *_ in module.seen] == [text_rows], (name, block.index)
                assert "forward" not in vars(module)


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def comfy_ops_and_lens():
    """Import real `comfy.ops` / `comfy.ldm.lens.model`, then undo it."""
    preserved = {n: m for n, m in sys.modules.items() if _is_comfy_module(n)}
    for name in preserved:
        sys.modules.pop(name, None)
    original_path = list(sys.path)
    comfy_dir = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    original_argv = sys.argv
    try:
        sys.argv = ["pytest-lens-text-projections", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            ops = pytest.importorskip("comfy.ops")
            lens_model = pytest.importorskip("comfy.ldm.lens.model")
        finally:
            sys.argv = original_argv
        yield ops, lens_model
    finally:
        for name in [n for n in sys.modules if _is_comfy_module(n)]:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = original_path


def test_real_comfy_block_selects_qkv_out_and_mlp_for_bf16_and_plain_fp8(comfy_ops_and_lens):
    ops, lens_model = comfy_ops_and_lens

    def block(operations):
        return lens_model.LensTransformerBlock(
            dim=8, num_attention_heads=2, attention_head_dim=4,
            dtype=torch.float32, device="cpu", operations=operations)

    for family in ("disable_weight_init", "manual_cast", "fp8_ops"):
        real = block(getattr(ops, family))
        selected = lens._text_modules(real)
        assert [module for module, _ in selected] == [
            real.attn.txt_qkv, real.attn.to_add_out, real.txt_mlp], family
        assert [joint for _, joint in selected] == [False, True, False]
    mixed = block(ops.mixed_precision_ops({}, torch.float32))
    assert not isinstance(mixed.attn.txt_qkv, torch.nn.Linear)
    assert lens._text_modules(mixed) == ()
