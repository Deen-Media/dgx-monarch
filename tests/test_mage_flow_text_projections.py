"""CPU contracts for Mage Flow's full-row text projections (pure Ulysses).

Stock runs each double block's text-stream Linears on every text row. A shard
runs them on half, the BLAS can pick another kernel for another row count, and
the bits change. Under pure Ulysses the forward wraps `add_q_proj`,
`add_k_proj`, `add_v_proj`, `to_add_out` and both `txt_mlp` projections
through chroma_text's helper, so each sees all text rows in stock's layout and
its output is re-sharded. The three `add_*` projections take one input, so the
forward gathers it once. The first tests drive the helper on one block; the
rest drive the real bound Mage forward through the harness in
tests/test_mage_flow_order.py. No ComfyUI import:
tests/test_mage_flow_stock_exact.py runs the real model.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.adapters import base, chroma_text, mage_flow
from dgx_monarch.adapters.base import pad_seq_to_multiple
from test_flux_pad_exclusion import _rank
from test_mage_flow_order import WORLD, _MageToyBlock, _references, _run, _streams
from test_qwen_text_projections import DIM, NAMES, _block, _module, _real_strides, _record, _SpyLinear


def _select(block):
    # Imported per call, so a run against a tree without the selector fails
    # on its assertions rather than at collection.
    from dgx_monarch.adapters.mage_flow import _text_modules

    return _text_modules(block)


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
                block, text_rows, image_rows, pure_ulysses=True, select=_select):
            for name in NAMES:
                outputs[name].append(_module(block, name)(chunks[rank]))
    return {name: torch.cat(parts, dim=1)[:, :text_rows] for name, parts in outputs.items()}


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("rows", [1, 6, 7, 27, 28, 285])
def test_every_wrapped_projection_sees_full_rows_in_stock_layout(monkeypatch, batch, world, rows):
    """6 and 28 are the shipped T2I negative and positive; 7, 27 and 285 pad."""
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
                assert _real_strides(stride, seen_shape) == _real_strides(
                    source.stride(), source.shape), name


def test_selection_is_the_six_text_linears_and_skips_other_modules():
    block = _block()
    selected = _select(block)
    assert [module for module, _ in selected] == [_module(block, name) for name in NAMES]
    assert [joint for _, joint in selected] == [False, False, False, True, False, False]

    class MixedPrecisionLinear(torch.nn.Module):
        """Stands in for comfy's mixed_precision_ops Linear."""
        layout_type = "TensorCoreNVFP4Layout"

        def forward(self, value):
            return value
    block.attn.add_k_proj = MixedPrecisionLinear()
    block.txt_mlp.net[2] = MixedPrecisionLinear()
    assert [module for module, _ in _select(block)] == [
        block.attn.add_q_proj, block.attn.add_v_proj, block.attn.to_add_out,
        block.txt_mlp.net[0].proj]
    assert _select(SimpleNamespace()) == ()


def test_selection_takes_an_unquantized_layer_of_a_quantized_checkpoint():
    """comfy loads a layer that a quantized file stores unquantized through
    its mixed-precision Linear (a torch.nn.Module with `layout_type` unset).
    That layer runs on full rows; a quantized `layout_type` keeps the shard."""
    class DenseMixedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(DIM, DIM, dtype=torch.bfloat16))

        def forward(self, value):
            return value
    block = _block()
    block.attn.to_add_out = DenseMixedLinear()
    block.txt_mlp.net[0].proj = DenseMixedLinear()
    selected = _select(block)
    assert [module for module, _ in selected] == [_module(block, name) for name in NAMES]
    assert [joint for _, joint in selected] == [False, False, False, True, False, False]
    for layout in ("TensorCoreMXFP8Layout", "TensorCoreNVFP4Layout", "TensorCoreFP8E4M3Layout"):
        block.attn.to_add_out.layout_type = layout
        block.txt_mlp.net[0].proj.layout_type = layout
        assert [module for module, _ in _select(block)] == [
            block.attn.add_q_proj, block.attn.add_k_proj, block.attn.add_v_proj,
            block.txt_mlp.net[2]], layout
    # A module with no weight tensor is not a Linear the selector can wrap.
    block.attn.to_add_out = torch.nn.Identity()
    block.txt_mlp.net[0].proj.layout_type = None
    assert [module for module, _ in _select(block)] == [
        block.attn.add_q_proj, block.attn.add_k_proj, block.attn.add_v_proj,
        block.txt_mlp.net[0].proj, block.txt_mlp.net[2]]


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
    assert len(_select(block)) == 6
    seen = _record(block)
    _run_helper(monkeypatch, block, torch.randn(1, 5, DIM, dtype=torch.float64),
                world=2, image_rows=3)
    assert all(len(seen[name]) == 2 and all(call[0] == 5 for call in seen[name])
               for name in NAMES)


def test_shared_gather_gathers_each_input_tensor_once(monkeypatch):
    calls = []

    def gather(value, length, dim=1):
        calls.append(value)
        return torch.cat((value, value), dim=dim).narrow(dim, 0, length)
    monkeypatch.setattr(mage_flow, "sp_gather", gather)
    shared = mage_flow._shared_gather()
    first = torch.randn(2, 3, DIM, dtype=torch.float64)
    gathered = [shared(first, 5, dim=1) for _ in range(3)]
    assert len(calls) == 1 and all(value is gathered[0] for value in gathered)
    assert gathered[0].is_contiguous() and gathered[0].shape == (2, 5, DIM)
    # Equal values in another tensor are another input: identity, not value.
    shared(first.clone(), 5, dim=1)
    assert len(calls) == 2
    # One slot: the earlier tensor is gathered again after another input.
    shared(first, 5, dim=1)
    assert len(calls) == 3 and calls[2] is first


def test_helper_takes_a_replacement_gather_and_defaults_to_sp_gather(monkeypatch):
    """chroma_text's helper calls `gather` in place of `sp_gather`, and
    `sp_gather` when `gather` is omitted."""
    block = _block()
    source = torch.randn(2, 5, DIM, dtype=torch.float64)
    padded, rows = pad_seq_to_multiple(source, 2, dim=1)
    local = torch.chunk(padded, 2, dim=1)[0]
    monkeypatch.setattr(chroma_text, "sp_rank", lambda: 0)
    monkeypatch.setattr(chroma_text, "sp_world", lambda: 2)
    default_calls, replacement_calls = [], []

    def default(value, length, dim=1):
        default_calls.append((length, dim))
        return padded[:, :length]

    def replacement(value, length, dim=1):
        replacement_calls.append((length, dim))
        return padded[:, :length]
    monkeypatch.setattr(chroma_text, "sp_gather", default)
    results = []
    for gather in (None, replacement):
        with chroma_text.full_row_text_projections(
                block, rows, 3, pure_ulysses=True, select=_select, gather=gather):
            results.append([_module(block, name)(local) for name in NAMES])
    assert default_calls == [(5, 1)] * len(NAMES)
    assert replacement_calls == [(5, 1)] * len(NAMES)
    assert all(torch.equal(a, b) for a, b in zip(*results, strict=True))


class _MageLinearBlock(_MageToyBlock):
    """The toy attention block plus Mage's six text Linears, each called once.

    Inputs are shared as stock shares them: the three `add_*` projections take
    one tensor, and `to_add_out` and each `txt_mlp` projection take their own.
    """

    def __init__(self, index):
        super().__init__(index)
        self.attn = SimpleNamespace(add_q_proj=_SpyLinear(), add_k_proj=_SpyLinear(),
                                    add_v_proj=_SpyLinear(), to_add_out=_SpyLinear())
        self.txt_mlp = SimpleNamespace(net=[SimpleNamespace(proj=_SpyLinear()), None, _SpyLinear()])

    def __call__(self, hidden_states, encoder_hidden_states, **kwargs):
        for name in NAMES:
            shared = name.startswith("add_")
            _module(self, name)(encoder_hidden_states if shared else encoder_hidden_states.clone())
        return super().__call__(hidden_states, encoder_hidden_states, **kwargs)


def _count_gathers(monkeypatch):
    """Every gather the Mage forward makes, by rank; chroma_text's default
    gather must not run, because the forward hands the helper its own."""
    calls: list[int] = []

    def counted(value, length, dim=1):
        calls.append(_rank())
        return base.sp_gather(value, length, dim=dim)

    def refused(*_args, **_kwargs):
        raise AssertionError("the Mage forward reached chroma_text's default gather")
    monkeypatch.setattr(mage_flow, "sp_gather", counted)
    monkeypatch.setattr(chroma_text, "sp_gather", refused)
    return calls


def _assert_rows(harness, text_rows, joint_rows, *, pure):
    local = -(-text_rows // WORLD)
    for model in harness.models:
        for block in model.transformer_blocks:
            for name in NAMES:
                module = _module(block, name)
                expected = text_rows if pure else local
                assert [rows for rows, *_ in module.seen] == [expected], (name, pure)
                assert "forward" not in vars(module)
            if pure:
                # Stock's joint output spans text, image and reference rows.
                assert block.attn.to_add_out.seen[0][1] == (joint_rows * DIM, DIM, 1)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("text_rows,grid", [(4, (2, 4)), (5, (3, 3)), (7, (2, 4))])
def test_forward_wraps_every_block_under_pure_ulysses_only(monkeypatch, text_rows, grid, batch):
    text, image = _streams(text_rows, grid, batch)
    for pure in (True, False):
        gathers = _count_gathers(monkeypatch)
        _outputs, harness = _run(monkeypatch, text, image, grid, world=WORLD,
                                 pure_ulysses=pure, block_cls=_MageLinearBlock)
        _assert_rows(harness, text_rows, text_rows + image.shape[1], pure=pure)
        # Per block: one gather for the add_* input, one each for to_add_out
        # and the two txt_mlp projections; then the forward's output gather.
        blocks = len(harness.models[0].transformer_blocks)
        per_rank = 4 * blocks + 1 if pure else 1
        assert sorted(gathers) == sorted(list(range(WORLD)) * per_rank), pure


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("text_rows,grid,ref_rows", [(5, (3, 3), [3]), (4, (2, 4), [2, 3])])
def test_to_add_out_stride_spans_the_reference_tokens(monkeypatch, text_rows, grid, ref_rows, batch):
    text, image = _streams(text_rows, grid, batch)
    refs = _references(text_rows, image.shape[1], ref_rows, batch)
    _outputs, harness = _run(monkeypatch, text, image, grid, world=WORLD, pure_ulysses=True,
                             block_cls=_MageLinearBlock, refs=refs)
    _assert_rows(harness, text_rows, text_rows + image.shape[1] + sum(ref_rows), pure=True)


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
                             block_cls=_MageLinearBlock,
                             forward_kwargs={"transformer_options": options})
    assert replaced == [1] * WORLD
    _assert_rows(harness, text_rows, text_rows + image.shape[1], pure=True)
