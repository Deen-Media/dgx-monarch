"""Boogu double blocks attend in stock key order (pure Ulysses).

Each rank joins ``[instruct_r, image_r]`` for the joint attention, so Ulysses'
head all-to-all hands the kernel ``[instruct0, image0, instruct1, image1]``
where stock attends ``[instruct0, instruct1, image0, image1]``. The same block
then runs image self-attention on the same ``transformer_options``, and that
call is in stock order already. This file drives the real bound Boogu forward
through the real ``base.make_usp_attention`` dispatch and full-axis call, with
only the all-to-all, the gather and the kernel stubbed (the harness of
tests/test_flux_pad_exclusion.py). Every key row carries a tag, so the kernel
names the exact rows and order it was handed, per attention stage.
"""
from __future__ import annotations

import sys
import threading
import types

import pytest
import torch

from dgx_monarch.adapters import base, chroma_text
from dgx_monarch.adapters.attention_patches import assert_no_foreign_attention_override
from dgx_monarch.adapters.base import InjectionContext
from dgx_monarch.adapters.boogu import BooguAdapter
from dgx_monarch.adapters.boogu_ulysses import double_block_options
from test_flux_pad_exclusion import (
    _CONTEXT,
    _LOCAL,
    _TAG,
    DIM,
    HEADS,
    _add_pe,
    _channel_map,
    _Harness,
    _install_stubs,
    _keep_tag,
    _matches,
    _Modulation,
    _rank,
    _ToyBlock,
)

WORLD = 2
# (instruct rows, noise grid, reference rows): every stream even, odd
# instruct, odd noise, both odd, and references that make the image stream
# odd or even on their own.
SHAPES = [
    (4, (2, 3), ()), (5, (2, 3), ()), (4, (3, 3), ()), (5, (3, 3), ()),
    (5, (2, 3), (3,)), (4, (3, 3), (2,)), (4, (2, 3), (2,)), (5, (3, 3), (1, 2)),
]


def _attend(stage, index, q, k, v, pe, options):
    """Call the installed override the way comfy's wrap_attn does."""
    _CONTEXT.where = (stage, index)
    q = _add_pe(q, pe)
    shaped = [t.reshape(t.shape[0], t.shape[1], HEADS, -1).transpose(1, 2) for t in (q, k, v)]
    override = options["optimized_attention_override"]
    return override(None, *shaped, HEADS, skip_reshape=True, mask=None,
                    transformer_options=options, _inside_attn_wrapper=True)


class _ToyStream(_ToyBlock):
    """A refiner or single block: one self-attention over its stream."""

    def __init__(self, stage, index):
        super().__init__(index)
        self.stage = stage

    def __call__(self, x, mask, rope, temb, transformer_options):
        mod = _Modulation(self.index)
        attended = _attend(self.stage, self.index, *self.qkv(x, mod), rope, transformer_options)
        return self.residual(x, attended, mod)


class _ToyContext:
    """The replicated caption refiner: per row, and it must see no USP override."""

    def __init__(self):
        self.seen = []

    def __call__(self, x, mask, rope, transformer_options):
        self.seen.append(("optimized_attention_override" in transformer_options, mask))
        return _keep_tag(_channel_map(x, 6) + rope.sum(-1, keepdim=True), x[..., :1])


class _ToyDouble(_ToyBlock):
    """Stock's double block shape: joint [instruct, image] then image alone,
    both on the one ``transformer_options`` the loop hands the block."""

    def __call__(self, img, instruct, joint_rope, img_rope, temb,
                 joint_attention_mask=None, img_attention_mask=None, transformer_options=None):
        mod, own = _Modulation(self.index), _Modulation(self.index + 5)
        iq, ik, iv = self.qkv(instruct, mod)
        gq, gk, gv = self.qkv(img, mod)
        joint = _attend("joint", self.index, torch.cat((iq, gq), 1), torch.cat((ik, gk), 1),
                        torch.cat((iv, gv), 1), joint_rope, transformer_options)
        rows = instruct.shape[1]
        self.instruct_output(joint[:, :rows])
        alone = _attend("image", self.index, *self.qkv(img, own), img_rope, transformer_options)
        img = self.residual(self.residual(img, joint[:, rows:], mod), alone, own)
        return img, self.residual(instruct, joint[:, :rows], mod)

    def instruct_output(self, rows):
        """Stock's ``instruct_out`` operand: the instruct slice of the joint output."""


def _tagged(count, first, batch, scale):
    """Rows tagged first..first+count-1 with a distinct body per batch item."""
    rows = torch.zeros(batch, count, DIM, dtype=torch.float64)
    rows[:, :, _TAG] = torch.arange(first, first + count, dtype=torch.float64)
    for column in range(1, DIM):
        rows[:, :, column] = scale * column + 0.03 * torch.arange(count, dtype=torch.float64)
    for item in range(1, batch):
        rows[item, :, _TAG + 1:] *= 1.0 + 0.1 * item
    return rows


class _ToyBoogu:
    """Boogu's forward surface with per-row toy blocks. Tags run instruct
    1..T, reference T+1..T+R, noise T+R+1..T+R+N: stock's joint order."""

    patch_size = 2

    def __init__(self, text_rows, grid, refs=(), batch=1, doubles=2, singles=2,
                 double_cls=_ToyDouble):
        self.text_rows, self.grid, self.refs, self.batch = text_rows, grid, tuple(refs), batch
        self.noise_rows = grid[0] * grid[1]
        self.context_refiner = [_ToyContext()]
        self.noise_refiner = [_ToyStream("noise", 20)]
        self.ref_image_refiner = [_ToyStream("ref", 30)]
        self.double_stream_layers = [double_cls(index) for index in range(doubles)]
        self.single_stream_layers = [_ToyStream("single", 40 + index) for index in range(singles)]
        self.image_index_embedding = torch.zeros(5, DIM, dtype=torch.float64)
        self.image_index_embedding[:, 1:] = 0.05   # never touches the tag channel

    def time_caption_embed(self, timestep, context, dtype):
        return torch.zeros(context.shape[0], 1, dtype=torch.float64), context

    x_embedder = ref_image_patch_embedder = staticmethod(lambda value: value)
    norm_out = staticmethod(lambda value, temb: _channel_map(value, 9))

    def flat_and_pad_to_seq(self, hidden_states, ref):
        total_ref = sum(self.refs)
        noise = _tagged(self.noise_rows, self.text_rows + total_ref + 1, self.batch, 0.2)
        flat_ref = _tagged(total_ref, self.text_rows + 1, self.batch, 0.3) if ref else None
        ref_lens = [list(self.refs) if ref else [0]] * self.batch
        return (noise, flat_ref, None, None, ref_lens, [self.noise_rows] * self.batch,
                [None] * self.batch, [(2 * self.grid[0], 2 * self.grid[1])] * self.batch)

    def rope_embedder(self, batch, width, cap_lens, l_ref, l_img, ref_sizes, img_sizes, device):
        total = width + sum(l_ref[0]) + l_img[0]
        positions = torch.arange(1, total + 1, dtype=torch.float64).reshape(1, total, 1)
        full = (0.01 * positions * torch.arange(1, DIM, dtype=torch.float64)).repeat(batch, 1, 1)
        ref_end = width + sum(l_ref[0])
        return (full[:, :width], full[:, width:ref_end], full[:, ref_end:], full,
                list(cap_lens), [total] * batch)


def stage_tags(text_rows, noise_rows, refs):
    """The tags stock attends in each stage, in stock's order."""
    ref_rows = sum(refs)
    total = text_rows + ref_rows + noise_rows
    every = [float(t) for t in range(1, total + 1)]
    return {"noise": every[text_rows + ref_rows:], "ref": every[text_rows:text_rows + ref_rows],
            "joint": every, "image": every[text_rows:], "single": every}


class StageHarness(_Harness):
    """The flux harness, with the expected key set chosen per attention stage."""

    def __init__(self, world, stages, *, strict):
        super().__init__(world, strict=strict)
        self.stages = stages
        self.records: list = []   # (stage, sequence_order, drop_rows) per dispatch

    def attend(self, entry, q, k, v):
        self.expected_tags = self.stages[_CONTEXT.where[0]]
        return super().attend(entry, q, k, v)


@pytest.fixture
def comfy_leaves(monkeypatch):
    """The two comfy leaves the forward imports before its stages."""
    comfy = types.ModuleType("comfy")
    ldm = types.ModuleType("comfy.ldm")
    common_dit = types.ModuleType("comfy.ldm.common_dit")
    common_dit.pad_to_patch_size = lambda x, patch: x
    management = types.ModuleType("comfy.model_management")
    management.cast_to = lambda t, dtype=None, device=None: t
    comfy.ldm, ldm.common_dit, comfy.model_management = ldm, common_dit, management
    for name, module in (("comfy", comfy), ("comfy.ldm", ldm),
                         ("comfy.ldm.common_dit", common_dit),
                         ("comfy.model_management", management)):
        monkeypatch.setitem(sys.modules, name, module)


def drive(monkeypatch, text_rows, grid, refs=(), *, world, pure_ulysses, batch=1,
          strict=True, double_cls=_ToyDouble):
    """Run the real bound forward once per rank thread, each on its own model."""
    stages = stage_tags(text_rows, grid[0] * grid[1], refs)
    harness = StageHarness(world, stages, strict=strict)
    _install_stubs(monkeypatch, harness)
    monkeypatch.setattr(chroma_text, "sp_rank", _rank)
    monkeypatch.setattr(chroma_text, "sp_world", lambda: world)
    kernel = base.make_usp_attention("TORCH_FLASH")

    def attn(*args, **kwargs):
        harness.records.append((_CONTEXT.where[0], kwargs.get("sequence_order"),
                                kwargs.get("drop_rows")))
        return kernel(*args, **kwargs)

    models = [_ToyBoogu(text_rows, grid, refs, batch, double_cls=double_cls)
              for _ in range(world)]
    for model in models:
        BooguAdapter().inject_usp(model, InjectionContext(
            topology_sp=world, usp_attention=attn, pure_ulysses=pure_ulysses))
    x = torch.zeros(batch, 1, 2 * grid[0], 2 * grid[1], dtype=torch.float64)
    context = _tagged(text_rows, 1, batch, 0.1)
    ref_latents = [torch.zeros(batch, 1, 2, 2, dtype=torch.float64)] if refs else None
    outputs: list = [None] * world
    errors: list = [None] * world

    def body(rank):
        try:
            _LOCAL.rank = rank
            outputs[rank] = models[rank].forward(
                x, torch.zeros(batch, dtype=torch.float64), context, text_rows,
                ref_latents=ref_latents, transformer_options={})
        except BaseException as exc:  # re-raised in the test body
            errors[rank] = exc
            harness.fabric.barrier.abort()

    threads = [threading.Thread(target=body, args=(rank,)) for rank in range(world)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=90)
    for error in errors:
        if error is not None:
            raise error
    harness.models = models
    return outputs, harness


def rank_major_tags(text_rows, image_tags):
    """The joint key order a rank-major gather hands the kernel, pads as tag 0."""
    text_local, image_local = -(-text_rows // WORLD), -(-len(image_tags) // WORLD)
    text_tags = [float(t) for t in range(1, text_rows + 1)]
    order = []
    for rank in range(WORLD):
        for tags, local in ((text_tags, text_local), (image_tags, image_local)):
            chunk = tags[rank * local:(rank + 1) * local]
            order += chunk + [0.0] * (local - len(chunk))
    return order


def _calls(harness, stage):
    return [tags for _entry, kind, _index, tags in harness.calls if kind == stage]


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("text_rows,grid,refs", SHAPES)
def test_joint_keys_arrive_in_stock_order_and_the_image_call_keeps_its_own(
    monkeypatch, comfy_leaves, text_rows, grid, refs, batch
):
    outputs, harness = drive(monkeypatch, text_rows, grid, refs, world=WORLD,
                             pure_ulysses=True, batch=batch, strict=False)
    stages = harness.stages
    joint = _calls(harness, "joint")
    assert len(joint) == WORLD * 2
    # Real rows arrive in stock order. No pad row arrives at all
    # (tests/test_boogu_pad_exclusion.py checks that with the strict recorder).
    assert all([t for t in tags if t] == stages["joint"] for tags in joint)
    # The image call never sees the joint descriptor, and its keys are its own.
    assert all([t for t in tags if t] == stages["image"] for tags in _calls(harness, "image"))
    for stage, order, _drops in harness.records:
        if stage == "joint":
            assert (order.text_rows, order.image_rows) == (
                -(-text_rows // WORLD), -(-len(stages["image"]) // WORLD))
        else:
            assert order is None, stage
    assert all(torch.equal(out, outputs[0]) for out in outputs)


@pytest.mark.parametrize("text_rows,grid,refs", [(4, (2, 3), ()), (6, (2, 3), (2,))])
def test_an_unpadded_render_matches_one_rank_in_every_stage(
    monkeypatch, comfy_leaves, text_rows, grid, refs
):
    """Nothing pads, so restoring the joint order is the whole difference."""
    reference, _ = drive(monkeypatch, text_rows, grid, refs, world=1, pure_ulysses=False)
    outputs, harness = drive(monkeypatch, text_rows, grid, refs, world=WORLD,
                             pure_ulysses=True)
    for _entry, kind, _index, tags in harness.calls:
        assert tags == harness.stages[kind], kind
    assert all(torch.equal(out, outputs[0]) for out in outputs)
    assert _matches(outputs[0], reference[0])


@pytest.mark.parametrize("text_rows,grid,refs", SHAPES)
def test_without_pure_ulysses_every_stage_keeps_the_old_path(
    monkeypatch, comfy_leaves, text_rows, grid, refs
):
    """Ring and hybrid contexts get no descriptor and no drop rows anywhere."""
    _outputs, harness = drive(monkeypatch, text_rows, grid, refs, world=WORLD,
                              pure_ulysses=False, strict=False)
    assert harness.records and all(
        order is None and drops is None for _stage, order, drops in harness.records)
    expected = rank_major_tags(text_rows, harness.stages["image"])
    # The negative control: rank-major differs from stock, so the test above
    # would fail on a forward that never reordered.
    assert [t for t in expected if t] != harness.stages["joint"]
    assert all(tags == expected for tags in _calls(harness, "joint"))


def test_the_caption_refiner_stays_replicated_and_unrouted(monkeypatch, comfy_leaves):
    _outputs, harness = drive(monkeypatch, 5, (3, 3), world=WORLD, pure_ulysses=True,
                              strict=False)
    for model in harness.models:
        assert model.context_refiner[0].seen == [(False, None)]


def test_a_query_length_neither_call_owns_refuses(monkeypatch, comfy_leaves):
    """A third attention shape in a double block fails closed, never guessed."""

    class _ThirdCall(_ToyDouble):
        def __call__(self, img, instruct, *args, transformer_options=None, **kwargs):
            q, k, v = self.qkv(img[:, :1], _Modulation(1))
            _attend("image", self.index, q, k, v, args[1][:, :1], transformer_options)
            return img, instruct

    with pytest.raises(base.UnsupportedModelError, match="neither the joint"):
        drive(monkeypatch, 4, (3, 3), world=WORLD, pure_ulysses=True, strict=False,
              double_cls=_ThirdCall)


@pytest.mark.parametrize("pure_ulysses", [True, False])
def test_the_double_block_override_carries_the_build_marker(monkeypatch, pure_ulysses):
    """The foreign-override guard accepts an override only by this marker, so
    without it every uly2 Boogu render that meets the guard would refuse."""
    monkeypatch.setattr(base, "sp_world", lambda: WORLD)
    options = double_block_options({}, lambda *args, **kwargs: None, (5, 3), (9, 5),
                                   pure_ulysses=pure_ulysses)
    override = options["optimized_attention_override"]
    assert getattr(override, base.USP_ATTENTION_OVERRIDE_ATTR, False) is True
    assert_no_foreign_attention_override(options, "boogu")
