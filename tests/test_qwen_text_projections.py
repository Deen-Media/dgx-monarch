"""CPU contracts for Qwen-Image's full-row text projections (pure Ulysses).

Stock runs the six text-stream Linears of each double block on every text row.
A shard runs them on half, and the BLAS can pick another kernel for another row
count, so the bits can change. The forward wraps them (via chroma_text's shared
helper) so each sees all text rows in stock's layout and its output is
re-sharded. The first tests drive the helper on one block; later ones drive the
real bound forward. A Linear the selector cannot find would stay sharded with
nothing logged, so the layout tests pin the warning that names it. The final
test builds Qwen's real block from comfy's Linear families and skips where
ComfyUI is not importable; the comfy-canary job runs it against comfy master.
comfy.cli_args parses argv on import, so the fixture forces ``--cpu``.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.adapters import chroma_text, qwen_image
from dgx_monarch.adapters.base import pad_seq_to_multiple
from test_qwen_image_order import (
    DIM,
    WORLD,
    _QwenToyBlock,
    _references,
    _run,
    _streams,
)

NAMES = ("add_q_proj", "add_k_proj", "add_v_proj", "to_add_out", "mlp0", "mlp2")


def _linear():
    return torch.nn.Linear(DIM, DIM, dtype=torch.float64)


def _block():
    return SimpleNamespace(
        attn=SimpleNamespace(add_q_proj=_linear(), add_k_proj=_linear(),
                             add_v_proj=_linear(), to_add_out=_linear()),
        txt_mlp=SimpleNamespace(net=[SimpleNamespace(proj=_linear()), None, _linear()]),
    )


def _module(block, name):
    return {"mlp0": lambda: block.txt_mlp.net[0].proj,
            "mlp2": lambda: block.txt_mlp.net[2]}.get(name, lambda: getattr(block.attn, name))()


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
    """Each rank's output of every projection, joined and trimmed to full rows."""
    padded, text_rows = pad_seq_to_multiple(source, world, dim=1)
    chunks = list(torch.chunk(padded, world, dim=1))
    outputs = {name: [] for name in NAMES}
    for rank in range(world):
        def gather(value, length, dim=1, rank=rank):
            assert length == text_rows and dim == 1 and torch.equal(value, chunks[rank])
            return padded[:, :length]
        monkeypatch.setattr(chroma_text, "sp_gather", gather)
        monkeypatch.setattr(chroma_text, "sp_rank", lambda rank=rank: rank)
        monkeypatch.setattr(chroma_text, "sp_world", lambda: world)
        with chroma_text.full_row_text_projections(
                block, text_rows, image_rows, pure_ulysses=True,
                select=qwen_image._text_modules):
            for name in NAMES:
                outputs[name].append(_module(block, name)(chunks[rank]))
    return {name: torch.cat(parts, dim=1)[:, :text_rows] for name, parts in outputs.items()}


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("rows", [1, 28, 44, 45, 209, 210])
def test_every_projection_sees_full_rows_in_stock_layout(monkeypatch, batch, world, rows):
    torch.manual_seed(batch * 1000 + rows)
    block, image_rows = _block(), 17
    originals = {name: _module(block, name).forward for name in NAMES}
    seen = _record(block)
    source = torch.randn(batch, rows, DIM, dtype=torch.float64)
    actual = _run_helper(monkeypatch, block, source, world, image_rows)
    stock_joint = torch.empty(batch, rows + image_rows, DIM, dtype=torch.float64)
    stock_view = stock_joint[:, :rows]
    stock_view.copy_(source)
    for name in NAMES:
        stock_input = stock_view if name == "to_add_out" else source
        assert torch.equal(actual[name], originals[name](stock_input)), name
        assert len(seen[name]) == world, name
        for seen_rows, stride, contiguous, seen_shape in seen[name]:
            assert seen_rows == rows, name
            if name == "to_add_out":
                # Stock slices text out of the joint [text, image] attention
                # output, so its batch stride spans the image rows too.
                assert stride == stock_view.stride() and contiguous is stock_view.is_contiguous()
            else:
                assert contiguous, name
                # A size-1 batch axis keeps the padded gather's stride, which
                # no GEMM reads, so only the strides of real axes must match.
                assert _real_strides(stride, seen_shape) == _real_strides(
                    source.stride(), source.shape), name


def test_selection_covers_the_six_text_linears_and_skips_non_linears():
    block = _block()
    selected = qwen_image._text_modules(block)
    assert [module for module, _ in selected] == [_module(block, name) for name in NAMES]
    assert [joint for _, joint in selected] == [False, False, False, True, False, False]

    class MixedPrecisionLinear(torch.nn.Module):
        """Stands in for comfy's mixed_precision_ops Linear."""
        in_features = DIM

        def forward(self, value):
            return value
    block.attn.add_q_proj = MixedPrecisionLinear()
    block.txt_mlp.net[2] = MixedPrecisionLinear()
    selected = qwen_image._text_modules(block)
    assert [module for module, _ in selected] == [
        block.attn.add_k_proj, block.attn.add_v_proj, block.attn.to_add_out,
        block.txt_mlp.net[0].proj]
    assert qwen_image._text_modules(SimpleNamespace()) == ()


def _warnings(monkeypatch):
    """The path each `_text_modules` warning names, with `_warned_paths` emptied first."""
    seen = []
    monkeypatch.setattr(qwen_image, "_warned_paths", set())
    monkeypatch.setattr(qwen_image.log, "warning",
                        lambda _message, path, _kind: seen.append(path))
    return seen


def _fuse_qkv(block):
    for name in ("add_q_proj", "add_k_proj", "add_v_proj"):
        delattr(block.attn, name)
    block.attn.add_qkv = _linear()


def _rename_mlp(block):
    block.txt_ff = block.txt_mlp
    del block.txt_mlp


def _wrap_out(block):
    block.attn.to_add_out = torch.nn.Sequential(block.attn.to_add_out)


PATHS = [path for path, _ in qwen_image._TEXT_PATHS]
# Upstream layout changes that move a text Linear from where the selector looks.
LAYOUT_CHANGES = {
    "fused_qkv": (_fuse_qkv, PATHS[:3]),
    "renamed_mlp": (_rename_mlp, PATHS[4:]),
    "wrapped_out": (_wrap_out, [PATHS[3]]),
    "all_moved": (lambda block: vars(block).clear(), PATHS),
}


@pytest.mark.parametrize("change", sorted(LAYOUT_CHANGES))
def test_a_changed_comfy_layout_warns_once_naming_each_lost_path(monkeypatch, change):
    seen = _warnings(monkeypatch)
    block = _block()
    edit, lost = LAYOUT_CHANGES[change]
    edit(block)
    assert len(qwen_image._text_modules(block)) == len(NAMES) - len(lost)
    assert seen == lost
    # The selector runs per block per call; the warning does not repeat.
    qwen_image._text_modules(block)
    qwen_image._text_modules(_block())
    assert seen == lost


def test_stock_and_quantized_layouts_warn_nothing(monkeypatch):
    seen = _warnings(monkeypatch)
    block = _block()
    assert len(qwen_image._text_modules(block)) == len(NAMES)

    class QuantizedLinear(torch.nn.Module):
        """Like comfy's mixed_precision_ops Linear: no nn.Linear base, keeps in_features."""
        def __init__(self):
            super().__init__()
            self.in_features = DIM
    block.attn.add_q_proj = QuantizedLinear()
    block.txt_mlp.net[2] = QuantizedLinear()
    assert len(qwen_image._text_modules(block)) == len(NAMES) - 2
    assert seen == []


def test_plain_fp8_cast_linear_takes_full_rows(monkeypatch):
    class CastLinear(torch.nn.Linear):
        """Like comfy's manual_cast Linear: casts a stored fp8 weight per call."""
        def forward(self, value):
            return torch.nn.functional.linear(
                value, self.weight.to(value.dtype), self.bias.to(value.dtype))
    block = _block()
    for name in NAMES:
        cast = CastLinear(DIM, DIM, dtype=torch.float64)
        cast.weight = torch.nn.Parameter(
            _module(block, name).weight.detach().to(torch.float8_e4m3fn), requires_grad=False)
        if name == "mlp0":
            block.txt_mlp.net[0].proj = cast
        elif name == "mlp2":
            block.txt_mlp.net[2] = cast
        else:
            setattr(block.attn, name, cast)
    assert len(qwen_image._text_modules(block)) == 6
    seen = _record(block)
    source = torch.randn(1, 5, DIM, dtype=torch.float64)
    _run_helper(monkeypatch, block, source, world=2, image_rows=3)
    assert all(len(calls) == 2 and all(call[0] == 5 for call in calls)
               for calls in seen.values())


class _SpyLinear(torch.nn.Linear):
    def __init__(self):
        super().__init__(DIM, DIM, dtype=torch.float64)
        self.seen = []

    def forward(self, value):
        self.seen.append((value.shape[1], value.stride(), value.is_contiguous()))
        return super().forward(value)


class _LinearBlock(_QwenToyBlock):
    """The toy attention block plus Qwen's six text Linears, each called once."""

    def __init__(self, index):
        super().__init__(index)
        self.attn = SimpleNamespace(add_q_proj=_SpyLinear(), add_k_proj=_SpyLinear(),
                                    add_v_proj=_SpyLinear(), to_add_out=_SpyLinear())
        self.txt_mlp = SimpleNamespace(net=[SimpleNamespace(proj=_SpyLinear()), None, _SpyLinear()])

    def __call__(self, hidden_states, encoder_hidden_states, *rest, **kwargs):
        for name in NAMES:
            _module(self, name)(encoder_hidden_states)
        return super().__call__(hidden_states, encoder_hidden_states, *rest, **kwargs)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("text_rows,grid", [(4, (2, 4)), (5, (3, 3)), (7, (2, 4))])
def test_forward_wraps_every_block_under_pure_ulysses_only(monkeypatch, text_rows, grid, batch):
    text, image = _streams(text_rows, grid, batch)
    local = -(-text_rows // WORLD)
    for pure, expected_rows in ((True, text_rows), (False, local)):
        _outputs, harness = _run(monkeypatch, text, image, grid, world=WORLD,
                                 pure_ulysses=pure, block_cls=_LinearBlock)
        for model in harness.models:
            for block in model.transformer_blocks:
                for name in NAMES:
                    module = _module(block, name)
                    assert [rows for rows, *_ in module.seen] == [expected_rows], (name, pure)
                    assert "forward" not in vars(module)
                    if pure and name == "to_add_out":
                        _rows, stride, _contig = module.seen[0]
                        # Stock joint output spans all text and all image rows.
                        assert stride == ((text_rows + image.shape[1]) * DIM, DIM, 1)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("method", ["index", "offset", "negative_index"])
@pytest.mark.parametrize("text_rows,grid,ref_rows", [(5, (3, 3), [3]), (4, (2, 4), [2, 3])])
def test_to_add_out_stride_spans_the_reference_tokens(
    monkeypatch, text_rows, grid, ref_rows, method, batch
):
    """Stock's joint output holds text, image and reference rows, so the stride
    the forward hands `to_add_out` counts all three."""
    text, image = _streams(text_rows, grid, batch)
    refs = _references(text_rows, image.shape[1], ref_rows, batch)
    _outputs, harness = _run(monkeypatch, text, image, grid, world=WORLD, pure_ulysses=True,
                             block_cls=_LinearBlock, refs=refs,
                             forward_kwargs={"ref_latents_method": method})
    joint_rows = text_rows + image.shape[1] + sum(ref_rows)
    for model in harness.models:
        for block in model.transformer_blocks:
            assert [rows for rows, *_ in block.attn.to_add_out.seen] == [text_rows]
            assert block.attn.to_add_out.seen[0][1] == (joint_rows * DIM, DIM, 1)


def test_the_forward_warns_about_a_lost_linear_under_pure_ulysses_only(monkeypatch):
    """The toy block has no text Linears; only pure Ulysses runs the selector."""
    text, image = _streams(5, (3, 3))
    for pure, expected in ((False, set()), (True, set(PATHS))):
        seen = _warnings(monkeypatch)
        _run(monkeypatch, text, image, (3, 3), world=WORLD, pure_ulysses=pure)
        # Two rank threads share one process here, so both may warn for a path.
        assert set(seen) == expected, pure


@pytest.mark.parametrize("batch", [1, 2])
def test_a_replaced_double_block_still_runs_full_text_rows(monkeypatch, batch):
    """`patches_replace["dit"]` swaps the block call, not the projection patch."""
    text_rows, grid = 5, (3, 3)
    text, image = _streams(text_rows, grid, batch)
    replaced = []

    def replace(args, extra):
        replaced.append(args["transformer_options"]["block_index"])
        return extra["original_block"](args)

    options = {"patches_replace": {"dit": {("double_block", 1): replace}}}
    _outputs, harness = _run(monkeypatch, text, image, grid, world=WORLD, pure_ulysses=True,
                             block_cls=_LinearBlock, forward_kwargs={"transformer_options": options})
    assert replaced == [1] * WORLD
    for model in harness.models:
        for block in model.transformer_blocks:
            for name in NAMES:
                module = _module(block, name)
                assert [rows for rows, *_ in module.seen] == [text_rows], (name, block.index)
                assert "forward" not in vars(module)


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def comfy_ops_and_qwen():
    """Import real `comfy.ops` / `comfy.ldm.qwen_image.model`, then undo it."""
    preserved = {n: m for n, m in sys.modules.items() if _is_comfy_module(n)}
    for name in preserved:
        sys.modules.pop(name, None)
    original_path = list(sys.path)
    comfy_dir = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    original_argv = sys.argv
    try:
        sys.argv = ["pytest-qwen-text-projections", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            ops = pytest.importorskip("comfy.ops")
            qwen_model = pytest.importorskip("comfy.ldm.qwen_image.model")
        finally:
            sys.argv = original_argv
        yield ops, qwen_model
    finally:
        for name in [n for n in sys.modules if _is_comfy_module(n)]:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = original_path


def test_real_comfy_block_selects_six_linears_for_bf16_and_plain_fp8(
    comfy_ops_and_qwen, monkeypatch
):
    """Pins comfy's attribute paths: a rename fails here, where a render only logs a warning."""
    ops, qwen_model = comfy_ops_and_qwen
    seen = _warnings(monkeypatch)

    def block(operations):
        return qwen_model.QwenImageTransformerBlock(
            8, 2, 4, dtype=torch.float32, device="cpu", operations=operations)

    for family in ("disable_weight_init", "manual_cast", "fp8_ops"):
        real = block(getattr(ops, family))
        selected = qwen_image._text_modules(real)
        assert [module for module, _ in selected] == [
            real.attn.add_q_proj, real.attn.add_k_proj, real.attn.add_v_proj,
            real.attn.to_add_out, real.txt_mlp.net[0].proj, real.txt_mlp.net[2]], family
        assert [joint for _, joint in selected] == [False, False, False, True, False, False]
    mixed = block(ops.mixed_precision_ops({}, torch.float32))
    assert not isinstance(mixed.attn.add_q_proj, torch.nn.Linear)
    assert qwen_image._text_modules(mixed) == ()
    # Quantized Linears are the intended skip, not a lost path.
    assert seen == []
