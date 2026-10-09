"""Ernie-Image joint attention under pure Ulysses, on tagged keys (no ComfyUI).

Ernie shards one joint `[image, text]` stream, so Ulysses' head all-to-all
hands the kernel `[chunk0, chunk1]`, which is already stock's key order. This
file drives the real bound Ernie forward through the real
`base.make_usp_attention` dispatch and full-axis call, with only the
all-to-all, the gather, the kernel and comfy's rope and weight-cast functions
stubbed. Every key row carries a tag, so the kernel names the exact order it
was handed. The harness is the one tests/test_flux_pad_exclusion.py builds.
"""
from __future__ import annotations

import sys
import threading
import types

import pytest
import torch

from dgx_monarch.adapters import base
from dgx_monarch.adapters.base import InjectionContext
from dgx_monarch.adapters.ernie import ErnieAdapter
from test_flux_pad_exclusion import (
    _CONTEXT,
    _LOCAL,
    _TAG,
    BATCH,
    DIM,
    HEADS,
    _channel_map,
    _Harness,
    _install_stubs,
    _keep_tag,
    _matches,
    _Modulation,
    _row_norm,
)

WORLD = 2
# (image grid, text rows): even joint, odd text, odd image, both odd.
SHAPES = [((2, 4), 4), ((2, 4), 5), ((3, 3), 4), ((3, 3), 5)]


@pytest.fixture
def comfy_rope_stub(monkeypatch):
    """The comfy functions the bound attention forward imports, recording calls.

    The rope adds the row's rotary ids to the non-tag channels, so a rotary
    shard that drifted from its token shard would change the answer. The toy
    norms are the identity, so the fused and unfused branches agree here and
    only the recorded calls tell them apart.
    """
    calls: list[tuple] = []

    def rope(query, key, rotary):
        assert rotary.shape[1] == query.shape[1] == key.shape[1]
        return (torch.cat((query[..., :1], query[..., 1:] + rotary), dim=-1),
                torch.cat((key[..., :1], key[..., 1:] + rotary), dim=-1))

    def apply_rope_split_half(query, key, rotary):
        calls.append(("apply_rope_split_half", query.shape[1]))
        return rope(query, key, rotary)

    def rms_rope_split_half(query, key, rotary, q_scale, k_scale=None, epsilon=1e-6):
        calls.append(("rms_rope_split_half", query.shape[1], q_scale, k_scale, epsilon,
                      rotary.is_contiguous()))
        return rope(query, key, rotary)

    def cast_bias_weight(s, input=None, offloadable=False):
        assert offloadable
        calls.append(("cast_bias_weight", s))
        return s.weight, None, ("stream", s)

    def uncast_bias_weight(s, weight, bias, offload_stream):
        calls.append(("uncast_bias_weight", s, weight, bias, offload_stream))

    comfy = types.ModuleType("comfy")
    quant_ops = types.ModuleType("comfy.quant_ops")
    quant_ops.ck = types.SimpleNamespace(apply_rope_split_half=apply_rope_split_half,
                                         rms_rope_split_half=rms_rope_split_half)
    ops = types.ModuleType("comfy.ops")
    ops.cast_bias_weight, ops.uncast_bias_weight = cast_bias_weight, uncast_bias_weight
    model_management = types.ModuleType("comfy.model_management")
    model_management.in_training = False
    comfy.quant_ops, comfy.ops, comfy.model_management = quant_ops, ops, model_management
    for name, module in (("comfy", comfy), ("comfy.quant_ops", quant_ops), ("comfy.ops", ops),
                         ("comfy.model_management", model_management)):
        monkeypatch.setitem(sys.modules, name, module)
    return calls


class _ToyNorm:
    """An RMSNorm stand-in: identity, with the weight and eps stock reads."""

    def __init__(self, eps: float) -> None:
        self.weight = torch.ones(DIM, dtype=torch.float64)
        self.eps = eps

    def __call__(self, value):
        return value


class _ToyAttention:
    """Ernie's attention module surface; inject_usp binds `forward` on it.

    The key keeps the row's tag and the query and value carry a zero there,
    so the tag labels a key without entering any score.
    """

    heads, head_dim = HEADS, DIM

    def __init__(self, index: int) -> None:
        self.index = index
        # Neither is comfy's 1e-6 default, and they differ, so the recorded
        # epsilon names its source: stock's fused call reads norm_q's for both.
        self.norm_q, self.norm_k = _ToyNorm(1e-5), _ToyNorm(3e-5)
        self.to_out = [lambda value: _keep_tag(_channel_map(value, 9), torch.zeros_like(value[..., :1]))]

    def _map(self, x, seed, tag):
        return _keep_tag(_channel_map(x, self.index + seed), tag)

    def to_q(self, x):
        return self._map(x, 1, torch.zeros_like(x[..., :1]))

    def to_k(self, x):
        return self._map(x, 2, x[..., :1])

    def to_v(self, x):
        return self._map(x, 3, torch.zeros_like(x[..., :1]))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _ToyBlock:
    """Stock's block shape: per-row norm, modulation, attention, gated residual."""

    def __init__(self, index: int) -> None:
        self.index = index
        self.self_attention = _ToyAttention(index)
        self.norm_inputs: list[bool] = []   # whether each call's rows were contiguous

    def __call__(self, x, rotary, temb):
        self.norm_inputs.append(x.is_contiguous())
        shift, scale, gate = temb[0], temb[1], temb[2]
        _CONTEXT.where = ("joint", self.index)
        x_norm = _keep_tag(_row_norm(x) * (1 + scale) + shift, x[..., :1])
        attended = self.self_attention(x_norm, attention_mask=None, image_rotary_emb=rotary)
        return x + gate * attended


class _ToyErnie:
    """The ErnieImageModel attributes the bound forward reads, patch size 1."""

    patch_size = 1
    out_channels = DIM

    def __init__(self, image, blocks=2):
        self.image = image
        self.text_proj = None
        self.layers = [_ToyBlock(index) for index in range(blocks)]
        self.final_inputs: list[bool] = []

    def x_embedder(self, x):
        return self.image

    @staticmethod
    def pos_embed(ids):
        # (B, S, 3) ids as a (B, S, 1, 3) rotary: one value per non-tag channel.
        # Scaled so no score dominates the softmax and a pad key keeps weight.
        return 0.05 * ids.to(torch.float64).unsqueeze(2)

    @staticmethod
    def time_proj(timesteps):
        return torch.zeros(timesteps.shape[0], DIM, dtype=torch.float64)

    @staticmethod
    def time_embedding(sample):
        return sample

    @staticmethod
    def adaLN_modulation(c):
        mod = [_Modulation(seed) for seed in (1, 2)]
        parts = [mod[0].shift, mod[0].scale, mod[0].gate, mod[1].shift, mod[1].scale, mod[1].gate]
        return torch.cat([part.reshape(1, DIM) for part in parts], dim=-1).repeat(c.shape[0], 1)

    def final_norm(self, hidden, c):
        self.final_inputs.append(hidden.is_contiguous())
        return hidden

    @staticmethod
    def final_linear(hidden):
        return _channel_map(hidden, 11)


def _streams(grid, text_rows, batch=1):
    """Tagged image rows 1..I and text rows I+1..I+T, image first as stock cats.

    Items after the first keep every tag with a different feature body, so a
    path that mixed batch items would show in the outputs.
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


def _drive(harness, image, text, grid, *, pure_ulysses, blocks=2, forward_kwargs=None):
    """Run the real bound forward once per rank and return what each returned."""
    models = [_ToyErnie(image, blocks) for _ in range(harness.world)]
    kernel = base.make_usp_attention("TORCH_FLASH")

    def attn(*args, **kwargs):
        harness.dispatch.append(kwargs.get("drop_rows"))
        return kernel(*args, **kwargs)

    for model in models:
        ErnieAdapter().inject_usp(model, InjectionContext(
            topology_sp=harness.world, usp_attention=attn, pure_ulysses=pure_ulysses))
    batch = image.shape[0]
    x = torch.zeros(batch, 1, *grid, dtype=torch.float64)
    outputs: list = [None] * harness.world
    errors: list = [None] * harness.world

    def body(rank):
        try:
            _LOCAL.rank = rank
            outputs[rank] = models[rank].forward(
                x, torch.zeros(batch, dtype=torch.float64), text, **(forward_kwargs or {}))
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
    harness.models = models
    return outputs


def _run(monkeypatch, image, text, grid, *, world, pure_ulysses, strict=False, **drive):
    harness = _Harness(world, strict=strict)
    harness.expected_tags = [float(tag) for tag in range(1, image.shape[1] + text.shape[1] + 1)]
    _install_stubs(monkeypatch, harness)
    return _drive(harness, image, text, grid, pure_ulysses=pure_ulysses, **drive), harness


pytestmark = pytest.mark.usefixtures("comfy_rope_stub")


@pytest.mark.parametrize("pure_ulysses", [True, False])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("grid,text_rows", SHAPES)
def test_joint_keys_arrive_in_stock_order_with_no_descriptor(
    monkeypatch, grid, text_rows, batch, pure_ulysses
):
    """One joint stream shards as one chunk, so the gathered keys are stock's
    `[image, text]` order on every path; no sequence_order is needed."""
    image, text = _streams(grid, text_rows, batch)
    _outputs, harness = _run(monkeypatch, image, text, grid, world=WORLD,
                             pure_ulysses=pure_ulysses)
    stock = [float(tag) for tag in range(1, image.shape[1] + text_rows + 1)]
    assert len(harness.calls) == WORLD * 2
    for _entry, _kind, _index, tags in harness.calls:
        assert [tag for tag in tags if tag] == stock


@pytest.mark.parametrize("grid,text_rows", [((2, 4), 4), ((3, 3), 5)])
def test_an_even_joint_stream_matches_the_one_rank_render(monkeypatch, grid, text_rows):
    """No pad: the plain xfuser call already sees stock's keys."""
    image, text = _streams(grid, text_rows)
    reference, _ = _run(monkeypatch, image, text, grid, world=1, pure_ulysses=False, strict=True)
    outputs, harness = _run(monkeypatch, image, text, grid, world=WORLD, pure_ulysses=True,
                            strict=True)
    stock = [float(tag) for tag in range(1, image.shape[1] + text_rows + 1)]
    assert all(tags == stock for *_head, tags in harness.calls)
    assert all(torch.equal(out, outputs[0]) for out in outputs)
    assert _matches(outputs[0], reference[0])


ROPE_OPTIONS = {"transformer_options": {"rope_options": {
    "scale_y": 2.0, "scale_x": 0.5, "shift_t": 1.0, "shift_y": 0.5, "shift_x": 1.5}}}


@pytest.mark.parametrize("forward_kwargs", [None, ROPE_OPTIONS], ids=["stock-rope", "rope-options"])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("grid,text_rows", SHAPES)
def test_pure_ulysses_attends_exactly_stock_keys_and_matches_one_rank(
    monkeypatch, grid, text_rows, batch, forward_kwargs
):
    """The strict recorder fails on a pad tag (0) or a lost real row."""
    image, text = _streams(grid, text_rows, batch)
    reference, _ = _run(monkeypatch, image, text, grid, world=1, pure_ulysses=False,
                        strict=True, forward_kwargs=forward_kwargs)
    outputs, harness = _run(monkeypatch, image, text, grid, world=WORLD, pure_ulysses=True,
                            strict=True, forward_kwargs=forward_kwargs)
    joint = image.shape[1] + text_rows
    stock = [float(tag) for tag in range(1, joint + 1)]
    assert len(harness.calls) == WORLD * 2
    assert all(tags == stock for *_head, tags in harness.calls)
    # An odd stream takes the full-axis exclusion; an even one names no row
    # and keeps xfuser's own call, unchanged.
    entry = "exact" if joint % WORLD else "xfuser"
    assert {call[0] for call in harness.calls} == {entry}
    assert all(torch.equal(out, outputs[0]) for out in outputs)
    assert _matches(outputs[0], reference[0])


@pytest.mark.parametrize("grid,text_rows,drop", [
    ((2, 4), 4, []),     # 12 rows: nothing pads
    ((2, 4), 5, [13]),   # 13 rows: 7 per rank, the pad is gathered row 13
    ((3, 3), 4, [13]),
    ((3, 3), 5, []),
])
def test_drop_rows_name_the_joint_tail_pad(monkeypatch, grid, text_rows, drop):
    image, text = _streams(grid, text_rows)
    _outputs, harness = _run(monkeypatch, image, text, grid, world=WORLD, pure_ulysses=True)
    assert harness.dispatch and all(rows == drop for rows in harness.dispatch)


@pytest.mark.parametrize("grid,text_rows", SHAPES)
def test_ring_and_hybrid_keep_attended_pads_and_never_refuse(monkeypatch, grid, text_rows):
    """Without pure Ulysses no row is named, so the ring pad guard (which
    refuses named rows) cannot fire, and any pad key reaches the kernel."""
    image, text = _streams(grid, text_rows)
    _outputs, harness = _run(monkeypatch, image, text, grid, world=WORLD, pure_ulysses=False)
    assert harness.dispatch and all(rows is None for rows in harness.dispatch)
    joint = image.shape[1] + text_rows
    stock = [float(tag) for tag in range(1, joint + 1)]
    expected = stock + [0.0] * ((-joint) % WORLD)
    assert all(entry == "xfuser" and tags == expected
               for entry, _kind, _index, tags in harness.calls)


def test_without_the_drop_the_pad_row_changes_the_answer(monkeypatch):
    """The negative control: an attended pad is material, not a rounding flip,
    so the exact match above is evidence of the exclusion."""
    grid, text_rows = (2, 4), 5
    image, text = _streams(grid, text_rows)
    reference, _ = _run(monkeypatch, image, text, grid, world=1, pure_ulysses=False)
    outputs, harness = _run(monkeypatch, image, text, grid, world=WORLD, pure_ulysses=False)
    assert any(0.0 in tags for *_head, tags in harness.calls)
    assert (outputs[0] - reference[0]).norm() / reference[0].norm() > 1e-4


def test_the_pad_rows_do_not_outlive_their_forward(monkeypatch):
    """The cell the forward fills is cleared on return, so a later attention
    call outside a forward names no row."""
    grid, text_rows = (2, 4), 5
    image, text = _streams(grid, text_rows)
    _outputs, harness = _run(monkeypatch, image, text, grid, world=WORLD, pure_ulysses=True)
    assert harness.dispatch[-1] == [13]
    attention = harness.models[0].layers[0].self_attention
    x = torch.ones(1, 3, DIM, dtype=torch.float64)
    captured = []
    harness_attend = harness.attend

    def record(entry, q, k, v):
        captured.append(entry)
        return harness_attend(entry, q, k, v)

    harness.world = 1
    harness.attend = record
    from test_flux_pad_exclusion import _Fabric
    harness.fabric = _Fabric(1)
    _LOCAL.rank = 0
    attention.forward(x, attention_mask=None, image_rotary_emb=None)
    assert harness.dispatch[-1] is None
    assert captured == ["xfuser"]


def _named(calls, name):
    return [call for call in calls if call[0] == name]


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("grid,text_rows", [((2, 4), 4), ((2, 4), 5)])
def test_pure_ulysses_builds_q_and_k_on_stocks_fused_branch(
    monkeypatch, comfy_rope_stub, grid, text_rows, batch
):
    """Stock's inference branch: cast norm weights, one fused norm-and-rope
    call per block on this rank's rows, then the casts released."""
    image, text = _streams(grid, text_rows, batch)
    _outputs, harness = _run(monkeypatch, image, text, grid, world=WORLD, pure_ulysses=True)
    norms = {id(block.self_attention.norm_q): block.self_attention
             for model in harness.models for block in model.layers}
    fused = _named(comfy_rope_stub, "rms_rope_split_half")
    assert len(fused) == WORLD * 2 and not _named(comfy_rope_stub, "apply_rope_split_half")
    local = -(-(image.shape[1] + text_rows) // WORLD)
    for _name, rows, q_scale, k_scale, epsilon, contiguous in fused:
        attention = next(a for a in norms.values() if a.norm_q.weight is q_scale)
        assert rows == local and k_scale is attention.norm_k.weight
        assert epsilon == attention.norm_q.eps
        # A batch-2 chunk is a strided view; stock's rotary is contiguous.
        assert contiguous
    casts = _named(comfy_rope_stub, "cast_bias_weight")
    releases = _named(comfy_rope_stub, "uncast_bias_weight")
    assert len(casts) == len(releases) == 2 * len(fused)
    for _name, module, weight, bias, stream in releases:
        assert weight is module.weight and bias is None and stream == ("stream", module)


@pytest.mark.parametrize("grid,text_rows", [((2, 4), 4), ((2, 4), 5)])
def test_ring_and_hybrid_keep_the_unfused_branch(monkeypatch, comfy_rope_stub, grid, text_rows):
    image, text = _streams(grid, text_rows)
    _run(monkeypatch, image, text, grid, world=WORLD, pure_ulysses=False)
    assert len(_named(comfy_rope_stub, "apply_rope_split_half")) == WORLD * 2
    assert not _named(comfy_rope_stub, "rms_rope_split_half")
    assert not _named(comfy_rope_stub, "cast_bias_weight")


def test_training_takes_stocks_unfused_branch_under_pure_ulysses(monkeypatch, comfy_rope_stub):
    """Stock's own condition: in training it runs RMSNorm then the rope."""
    monkeypatch.setattr(sys.modules["comfy.model_management"], "in_training", True)
    grid, text_rows = (2, 4), 5
    image, text = _streams(grid, text_rows)
    _run(monkeypatch, image, text, grid, world=WORLD, pure_ulysses=True)
    assert len(_named(comfy_rope_stub, "apply_rope_split_half")) == WORLD * 2
    assert not _named(comfy_rope_stub, "rms_rope_split_half")


@pytest.mark.parametrize("pure_ulysses", [True, False], ids=["pure-ulysses", "ring-hybrid"])
def test_batch_2_norms_read_contiguous_rows_under_pure_ulysses_only(monkeypatch, pure_ulysses):
    """Stock's first block norm and final norm read contiguous rows. At batch 2
    the shard is a strided chunk and an odd stream's gather a strided narrow:
    pure Ulysses copies both, and ring and hybrid keep the views."""
    grid, text_rows = (2, 4), 5   # joint 13: odd, so the gather narrows
    image, text = _streams(grid, text_rows, 2)
    _outputs, harness = _run(monkeypatch, image, text, grid, world=WORLD,
                             pure_ulysses=pure_ulysses)
    for model in harness.models:
        assert model.layers[0].norm_inputs == [pure_ulysses]
        assert model.final_inputs == [pure_ulysses]
