"""Lens joint attention sees stock key order (pure Ulysses).

Each Lens block joins `[image, text]`, so with each stream sharded apart every
rank holds `[image_r, text_r]` and Ulysses' head all-to-all hands the kernel
`[image0, text0, image1, text1]`, where stock attends `[image0, image1, text0,
text1]`. This file drives the real bound Lens forward through the real
`base.make_usp_attention` dispatch and the real full-axis call, with only the
all-to-all, the gather and the kernel stubbed: the harness of
tests/test_flux_pad_exclusion.py. Image rows carry tags 1..I and text rows
I+1..I+T, so stock order is ascending and the kernel names the order it got.

Stock Lens also hands its kernel an additive key bias, all zeros for an
unmasked prompt, so under pure Ulysses the forward passes the same bias as one
query group and the full-axis call runs comfy's SDPA wrapper, stubbed here as
`comfy.ops`, which records the bias it was handed. A uniform trailing text pad
stays under that bias there, as in stock, and is trimmed elsewhere.
"""
from __future__ import annotations

import sys
import threading
import types

import pytest
import torch

import test_flux_pad_exclusion as harness_module
from dgx_monarch.adapters import base, chroma_text, lens
from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.lens import LensAdapter, LensJointOrder
from test_flux_pad_exclusion import (
    _LOCAL,
    _TAG,
    BATCH,
    DIM,
    _attention,
    _channel_map,
    _Harness,
    _install_stubs,
    _matches,
    _Modulation,
    _rank,
    _ToyBlock,
)

WORLD = 2
# (image grid, text rows): nothing pads, odd text, odd image, both odd.
SHAPES = [((2, 4), 4), ((2, 4), 5), ((3, 3), 4), ((3, 3), 5)]


def _position_ids(frame, height, width, text_seq_len, scale_rope=True, device=None):
    """Distinct ids per joint row, so a misaligned id shard shows in the output."""
    rows = frame * height * width + text_seq_len
    return torch.arange(1, rows + 1, dtype=torch.float64).reshape(rows, 1).repeat(1, DIM - 1)


def _recording_sdpa(q, k, v, attn_mask=None, **kwargs):
    """comfy's SDPA wrapper, (B, H, L, D) in and out, over the harness kernel."""
    active = harness_module._ACTIVE
    active.masks.append(attn_mask)
    shaped = (t.transpose(1, 2) for t in (q, k, v))
    return active.attend("sdpa", *shaped).transpose(1, 2)


@pytest.fixture
def comfy_lens_stub(monkeypatch):
    """The comfy leaves the Lens forward and the query-bias kernel import."""
    comfy = types.ModuleType("comfy")
    ldm = types.ModuleType("comfy.ldm")
    lens_pkg = types.ModuleType("comfy.ldm.lens")
    model = types.ModuleType("comfy.ldm.lens.model")
    ops = types.ModuleType("comfy.ops")
    model._lens_position_ids = _position_ids
    ops.scaled_dot_product_attention = _recording_sdpa
    comfy.ldm, comfy.ops, ldm.lens, lens_pkg.model = ldm, ops, lens_pkg, model
    for name, module in (("comfy", comfy), ("comfy.ldm", ldm), ("comfy.ops", ops),
                         ("comfy.ldm.lens", lens_pkg), ("comfy.ldm.lens.model", model)):
        monkeypatch.setitem(sys.modules, name, module)
    return comfy


pytestmark = pytest.mark.usefixtures("comfy_lens_stub")


class _LensToyBlock(_ToyBlock):
    """Lens's block signature over the flux toy's elementwise attention."""

    def __call__(self, hidden_states, encoder_hidden_states, temb, freqs_cis,
                 attention_mask=None, transformer_options=None):
        img_q, img_k, img_v = self.qkv(hidden_states, temb)
        txt_q, txt_k, txt_v = self.qkv(encoder_hidden_states, temb)
        attended = _attention(torch.cat((img_q, txt_q), dim=1),
                              torch.cat((img_k, txt_k), dim=1),
                              torch.cat((img_v, txt_v), dim=1),
                              freqs_cis, transformer_options)
        rows = hidden_states.shape[1]
        return (self.residual(encoder_hidden_states, attended[:, rows:], temb),
                self.residual(hidden_states, attended[:, :rows], temb))


class _ToyLens:
    multi_layer_encoder_feature = False
    img_in = staticmethod(lambda value: value)
    txt_norm = staticmethod(lambda value: value)
    txt_in = staticmethod(lambda value: value)
    pos_embed = staticmethod(lambda ids: ids)
    norm_out = staticmethod(lambda value, temb: value)

    def __init__(self, blocks=2, block_cls=_LensToyBlock):
        self.transformer_blocks = [block_cls(index) for index in range(blocks)]

    @staticmethod
    def time_text_embed(timestep, hidden_states):
        return _Modulation(1)

    @staticmethod
    def proj_out(value):
        return _channel_map(value, 9)


def _streams(grid, text_rows, batch=1):
    """Tagged image rows 1..I and text rows I+1..I+T, with distinct features.

    Items after the first keep every tag but get a different feature body, so a
    permutation that mishandled the batch axis would show in the outputs.
    """
    image_rows = grid[0] * grid[1]
    image = torch.zeros(BATCH, image_rows, DIM, dtype=torch.float64)
    text = torch.zeros(BATCH, text_rows, DIM, dtype=torch.float64)
    image[0, :, _TAG] = torch.arange(1, image_rows + 1, dtype=torch.float64)
    text[0, :, _TAG] = torch.arange(image_rows + 1, image_rows + text_rows + 1, dtype=torch.float64)
    for column in range(1, DIM):
        image[0, :, column] = 0.2 * column - 0.05 * torch.arange(image_rows, dtype=torch.float64)
        text[0, :, column] = 0.1 * column + 0.03 * torch.arange(text_rows, dtype=torch.float64)
    image, text = image.repeat(batch, 1, 1), text.repeat(batch, 1, 1)
    for item in range(1, batch):
        image[item, :, _TAG + 1:] *= 1.0 - 0.07 * item
        text[item, :, _TAG + 1:] *= 1.0 + 0.1 * item
    return image, text


def _drive(harness, image, text, grid, *, pure_ulysses, blocks=2,
           block_cls=_LensToyBlock, forward_kwargs=None):
    """Run the real bound forward once per rank and return what each returned.

    Each rank thread gets its own model and blocks, as each rank process does,
    so a per-block patch never races with the other rank.
    """
    models = [_ToyLens(blocks, block_cls) for _ in range(harness.world)]
    kernel = base.make_usp_attention("TORCH_FLASH")
    seen: list[dict] = []

    def attn(*args, **kwargs):
        seen.append(dict(kwargs))
        return kernel(*args, **kwargs)

    for model in models:
        LensAdapter().inject_usp(model, InjectionContext(
            topology_sp=harness.world, usp_attention=attn, pure_ulysses=pure_ulysses))
    batch, (rows, cols) = image.shape[0], grid
    x = image.reshape(batch, rows, cols, DIM).permute(0, 3, 1, 2)
    outputs: list = [None] * harness.world
    errors: list = [None] * harness.world

    def body(rank):
        try:
            _LOCAL.rank = rank
            out = models[rank]._forward(x, torch.zeros(batch, dtype=torch.float64), text,
                                        **(forward_kwargs or {}))
            outputs[rank] = out.permute(0, 2, 3, 1).reshape(batch, rows * cols, DIM)
        except BaseException as exc:  # re-raised in the test body
            errors[rank] = exc
            harness.fabric.barrier.abort()

    threads = [threading.Thread(target=body, args=(rank,)) for rank in range(harness.world)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=90)
    for error in errors:
        if error is not None:
            raise error
    harness.seen, harness.models = seen, models
    return outputs


def _run(monkeypatch, image, text, grid, *, world, pure_ulysses, strict=False,
         installer=_install_stubs, **drive):
    monkeypatch.setattr(lens, "sp_rank", _rank)
    monkeypatch.setattr(chroma_text, "sp_rank", _rank)
    monkeypatch.setattr(chroma_text, "sp_world", lambda: world)
    harness = _Harness(world, strict=strict)
    harness.masks = []
    harness.expected_tags = [float(tag) for tag in range(1, image.shape[1] + text.shape[1] + 1)]
    installer(monkeypatch, harness)
    return _drive(harness, image, text, grid, pure_ulysses=pure_ulysses, **drive), harness


def _rank_major_tags(image_rows, text_rows):
    """The key order a rank-major gather hands the kernel, pad rows left out."""
    image_local, text_local = -(-image_rows // WORLD), -(-text_rows // WORLD)
    order = []
    for rank in range(WORLD):
        for start, local, total, offset in ((rank * image_local, image_local, image_rows, 0),
                                            (rank * text_local, text_local, text_rows, image_rows)):
            order += [float(offset + start + i + 1) for i in range(local) if start + i < total]
    return order


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("grid,text_rows", SHAPES)
def test_keys_arrive_in_stock_order_and_outputs_return_to_their_rank(
    monkeypatch, grid, text_rows, batch
):
    image, text = _streams(grid, text_rows, batch)
    reference, _ = _run(monkeypatch, image, text, grid, world=1, pure_ulysses=False, strict=True)
    outputs, harness = _run(monkeypatch, image, text, grid, world=WORLD, pure_ulysses=True,
                            strict=True)
    stock = [float(tag) for tag in range(1, image.shape[1] + text_rows + 1)]
    assert len(harness.calls) == WORLD * 2
    # All image rows, then all text rows, pad rows gone, through the full-axis
    # path and comfy's SDPA rather than xfuser's kernel.
    assert all(entry == "sdpa" and tags == stock for entry, _kind, _index, tags in harness.calls)
    # Stock's bias: zeros over every real key, one row per batch item.
    assert len(harness.masks) == WORLD * 2
    for mask in harness.masks:
        assert mask.shape == (batch, 1, 1, len(stock)) and mask.dtype == torch.float64
        assert not mask.any()
    orders = {(order.image_rows, order.text_rows, order.image_real, order.text_real)
              for order in (call["sequence_order"] for call in harness.seen)}
    assert orders == {(-(-image.shape[1] // WORLD), -(-text_rows // WORLD),
                       image.shape[1], text_rows)}
    # The inverse permutation returned each row to its rank: every rank holds
    # the whole image, and it is the one-rank answer for each batch item.
    assert all(torch.equal(out, outputs[0]) for out in outputs)
    assert _matches(outputs[0], reference[0])


@pytest.mark.parametrize("grid,text_rows", SHAPES)
def test_without_pure_ulysses_the_keys_stay_rank_major(monkeypatch, grid, text_rows):
    """A ring or hybrid worker context never receives the descriptor."""
    image, text = _streams(grid, text_rows)
    _outputs, harness = _run(monkeypatch, image, text, grid, world=WORLD, pure_ulysses=False)
    expected = _rank_major_tags(image.shape[1], text_rows)
    assert all(call.get("sequence_order") is None and "query_bias_groups" not in call
               for call in harness.seen)
    assert harness.masks == []
    # The negative control: rank-major differs from stock, so the test above
    # would fail on a forward that never reordered.
    assert expected != sorted(expected)
    assert harness.calls and all(tags == expected for *_head, tags in harness.calls)


@pytest.mark.parametrize("pure", [True, False])
@pytest.mark.parametrize("grid,text_rows,rank_major,canonical", [
    # Local pair per rank is [image_local, text_local]; each stream's pad is
    # its last padded row, held by rank 1. Worked by hand:
    ((2, 4), 4, [], []),               # nothing pads
    ((2, 4), 5, [13], [13]),           # image 4 + text 3 per rank: 7 + 4 + 2
    ((3, 3), 4, [11], [13]),           # image 5 + text 2 per rank: 7 + 4
    ((3, 3), 5, [12, 15], [14, 15]),   # image 5 + text 3 per rank: 8 + 4, 8 + 5 + 2
])
def test_drop_rows_are_unchanged_and_the_order_moves_them_to_the_tail(
    monkeypatch, grid, text_rows, rank_major, canonical, pure
):
    """The dispatcher gets rank-major pad coordinates in every mode, so the ring
    pad guard reads the same rows either way; the descriptor maps them past the
    last real row."""
    image, text = _streams(grid, text_rows)
    _outputs, harness = _run(monkeypatch, image, text, grid, world=WORLD, pure_ulysses=pure)
    assert harness.seen and all(call["drop_rows"] == rank_major for call in harness.seen)
    if not pure:
        return
    order = harness.seen[0]["sequence_order"]
    gathered = WORLD * (order.image_rows + order.text_rows)
    probe = torch.arange(gathered, dtype=torch.float32).reshape(1, -1, 1, 1)
    _q, _k, _v, remapped, kv_remapped, _perm = order.canonicalize(
        probe, probe, probe, rank_major, rank_major, None)
    assert remapped == kv_remapped == canonical
    assert canonical == list(range(image.shape[1] + text_rows, gathered))


@pytest.mark.parametrize("batch", [1, 2])
def test_a_replaced_double_block_keeps_stock_order(monkeypatch, batch):
    """`patches_replace["dit"]` swaps the block call, not the attention options."""
    grid, text_rows = (3, 3), 5
    image, text = _streams(grid, text_rows, batch)
    replaced = []

    def replace(args, extra):
        replaced.append(args["transformer_options"]["block_index"])
        return extra["original_block"](args)

    options = {"patches_replace": {"dit": {("double_block", 1): replace}}}
    reference, _ = _run(monkeypatch, image, text, grid, world=1, pure_ulysses=False)
    outputs, harness = _run(monkeypatch, image, text, grid, world=WORLD, pure_ulysses=True,
                            strict=True, forward_kwargs={"transformer_options": options})
    assert replaced == [1] * WORLD
    stock = [float(tag) for tag in range(1, image.shape[1] + text_rows + 1)]
    assert all(tags == stock for *_head, tags in harness.calls)
    assert _matches(outputs[0], reference[0])


def test_the_ring_pad_guard_still_refuses_a_padded_ring_render(monkeypatch):
    """With a ring degree above one, the drop rows the forward names meet the
    shared ring pad guard, which refuses, and no order is applied."""
    def with_ring(patcher, harness):
        _install_stubs(patcher, harness)
        sys.modules["xfuser.core.distributed"].get_ring_parallel_world_size = lambda: 2

    image, text = _streams((3, 3), 5)
    with pytest.raises(UnsupportedModelError, match="ring/hybrid"):
        _run(monkeypatch, image, text, (3, 3), world=WORLD, pure_ulysses=False,
             installer=with_ring)


def test_permutation_round_trips_and_handles_an_empty_text_stream():
    order = LensJointOrder(text_rows=3, image_rows=5, image_real=9, text_real=5)
    permutation = order.permutation(16, torch.device("cpu"))
    # Real image rows, real text rows, image pad, text pad, all rank-major indices.
    assert permutation.tolist() == [0, 1, 2, 3, 4, 8, 9, 10, 11, 5, 6, 7, 13, 14, 12, 15]
    values = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1, 1)
    assert torch.equal(order.restore(values[:, permutation], permutation), values)
    empty = LensJointOrder(text_rows=0, image_rows=4, image_real=8, text_real=0)
    assert empty.permutation(8, torch.device("cpu")).tolist() == list(range(8))


@pytest.mark.parametrize("fields,rows", [
    ({"text_rows": 3, "image_rows": 5, "image_real": 9, "text_real": 5}, 15),   # not a tiling
    ({"text_rows": 3, "image_rows": 5, "image_real": 9, "text_real": 5}, 8),    # one rank
    ({"text_rows": 3, "image_rows": 5, "image_real": 11, "text_real": 5}, 16),  # real past pad
])
def test_the_descriptor_refuses_a_layout_it_does_not_describe(fields, rows):
    with pytest.raises(UnsupportedModelError, match="joint Ulysses order"):
        LensJointOrder(**fields).permutation(rows, torch.device("cpu"))


def test_the_descriptor_admits_only_the_whole_sequence_bias_group():
    """The stock-bias group names every real row from row 0, which no order
    changes; any other group set needs coordinates mapped, and is refused."""
    order = LensJointOrder(text_rows=3, image_rows=5, image_real=9, text_real=5)
    probe = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1, 1)
    bias = torch.zeros(1, 1, 1, 14)
    *_qkv, drops, kv_drops, _perm = order.canonicalize(
        probe, probe, probe, [12, 15], [12, 15], ((0, 14, bias),))
    assert drops == kv_drops == [14, 15]
    for groups in (((0, 13, bias),), ((0, 7, None), (7, 14, None))):
        with pytest.raises(UnsupportedModelError, match="query-bias group"):
            order.canonicalize(probe, probe, probe, [12, 15], [12, 15], groups)


def test_the_bias_override_keeps_the_usp_mark_and_builds_one_group_per_call():
    seen = []

    def inner(func, q, k, v, heads, **kwargs):
        seen.append(kwargs)
        return q

    keep = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])
    override = lens._stock_key_bias(inner, lens._joint_key_bias(keep, 2, 3, 4, "cpu"))
    assert getattr(override, base.USP_ATTENTION_OVERRIDE_ATTR) is True
    q = torch.zeros(2, 3, 4, 5, dtype=torch.bfloat16)
    override(None, q, q, q, 3, mask=None, skip_reshape=True)
    ((start, end, bias),) = seen[0]["query_bias_groups"]
    assert (start, end) == (0, 7) and seen[0]["skip_reshape"] is True
    # Stock's float32 minimum, cast to the query dtype per call, is -inf in bf16.
    assert bias.shape == (2, 1, 1, 7) and bias.dtype == torch.bfloat16
    assert not bias[..., :6].any() and torch.isneginf(bias[..., 6]).all()


def test_the_joint_bias_is_stock_s_zeros_with_the_float32_minimum_at_masked_text():
    absent = lens._joint_key_bias(None, 2, 3, 4, "cpu")
    assert absent.shape == (2, 1, 1, 7) and absent.dtype == torch.float32 and not absent.any()
    keep = torch.tensor([[True, True, False, False]])
    bias = lens._joint_key_bias(keep, 1, 3, 4, "cpu")
    assert bias[0, 0, 0].tolist() == [0.0] * 5 + [torch.finfo(torch.float32).min] * 2


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("grid", [(2, 4), (3, 3)])
def test_a_trailing_text_pad_stays_under_stock_bias_only_under_pure_ulysses(
    monkeypatch, grid, batch
):
    """Stock attends every text key, the padded ones under the float32 minimum,
    and the kernel's bits depend on how many keys it reduces over. Pure Ulysses
    keeps them under that bias; ring and hybrid trim them."""
    text_rows, real = 7, 5
    image, text = _streams(grid, text_rows, batch)
    keep = torch.zeros(batch, text_rows, dtype=torch.long)
    keep[:, :real] = 1
    image_rows = image.shape[1]
    _outputs, harness = _run(monkeypatch, image, text, grid, world=WORLD, pure_ulysses=True,
                             strict=True, forward_kwargs={"attention_mask": keep})
    every =[float(tag) for tag in range(1, image_rows + text_rows + 1)]
    assert harness.calls and all(entry == "sdpa" and tags == every
                                 for entry, _kind, _index, tags in harness.calls)
    floor = torch.finfo(torch.float32).min
    for mask in harness.masks:
        assert mask.shape == (batch, 1, 1, image_rows + text_rows)
        assert not mask[..., :image_rows + real].any()
        assert (mask[..., image_rows + real:] == floor).all()
    orders = {(order.text_rows, order.text_real)
              for order in (call["sequence_order"] for call in harness.seen)}
    assert orders == {(-(-text_rows // WORLD), text_rows)}

    _outputs, harness = _run(monkeypatch, image, text, grid, world=WORLD, pure_ulysses=False,
                             forward_kwargs={"attention_mask": keep})
    assert harness.masks == []
    assert harness.calls and all(tags == _rank_major_tags(image_rows, real)
                                 for *_head, tags in harness.calls)
