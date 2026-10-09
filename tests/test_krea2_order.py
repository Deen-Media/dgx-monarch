"""Krea2 joint blocks attend in stock key order (pure Ulysses).

Each rank joins `[text_r, image_r]` from its own shards, so Ulysses' head
all-to-all hands the kernel `[text0, image0, text1, image1]` where stock attends
`[text0, text1, image0, image1]`, and flash rounding depends on key order. This
file drives the real bound Krea2 forward through the real
`base.make_usp_attention` dispatch and the real full-axis call, with only the
all-to-all, the gather and the kernel stubbed. Every key row carries a tag, so
the kernel names the exact order it was handed. The harness is the one
tests/test_flux_pad_exclusion.py builds; the toy here adds Krea2's text path
(layerwise blocks, projector, refiner blocks, txtmlp) so
tests/test_krea2_text_rows.py can check which rows each part of it sees.
"""
from __future__ import annotations

import sys
import threading
import types

import pytest
import torch
from einops import rearrange

import test_flux_pad_exclusion as flux_harness
from dgx_monarch.adapters import base
from dgx_monarch.adapters.base import InjectionContext
from dgx_monarch.adapters.krea2 import Krea2Adapter
from test_flux_pad_exclusion import (
    _CONTEXT,
    _LOCAL,
    _TAG,
    BATCH,
    DIM,
    HEADS,
    _add_pe,
    _channel_map,
    _Harness,
    _install_stubs,
    _keep_tag,
    _matches,
    _Modulation,
    _ToyBlock,
)

WORLD = 2
# (text rows, image grid): even, odd text, odd image, both odd.
SHAPES = [(4, (2, 4)), (5, (2, 4)), (4, (3, 3)), (5, (3, 3))]


@pytest.fixture
def krea2_comfy_stub(monkeypatch):
    """The comfy leaves the Krea2 forward imports before its block loops."""
    comfy = types.ModuleType("comfy")
    ldm = types.ModuleType("comfy.ldm")
    common_dit = types.ModuleType("comfy.ldm.common_dit")
    flux_pkg = types.ModuleType("comfy.ldm.flux")
    layers = types.ModuleType("comfy.ldm.flux.layers")
    common_dit.pad_to_patch_size = lambda x, patch: x  # the toy patch is 1
    layers.timestep_embedding = lambda t, dim: torch.zeros(*t.shape, dim, dtype=torch.float64)
    comfy.ldm, ldm.common_dit, ldm.flux, flux_pkg.layers = ldm, common_dit, flux_pkg, layers
    for name, module in (("comfy", comfy), ("comfy.ldm", ldm),
                         ("comfy.ldm.common_dit", common_dit),
                         ("comfy.ldm.flux", flux_pkg),
                         ("comfy.ldm.flux.layers", layers)):
        monkeypatch.setitem(sys.modules, name, module)


pytestmark = pytest.mark.usefixtures("krea2_comfy_stub")


def _toy_attention(label, q, k, v, pe, options):
    """Call the installed override as comfy's wrap_attn does, else attend natively.

    Native attention is what stock runs and what the layerwise and refiner
    blocks get when no scoped override is installed: the whole key axis is on
    this rank, so the harness kernel attends it directly.
    """
    _CONTEXT.where = label
    if pe is not None:
        q = _add_pe(q, pe)
    shaped = [t.reshape(t.shape[0], t.shape[1], HEADS, -1).transpose(1, 2)
              for t in (q, k, v)]
    override = options.get("optimized_attention_override")
    if override is None:
        rows = [t.transpose(1, 2) for t in shaped]
        out = flux_harness._ACTIVE.attend("native", *rows)
        return out.reshape(out.shape[0], out.shape[1], -1)
    return override(None, *shaped, HEADS, skip_reshape=True, mask=None,
                    transformer_options=options, _inside_attn_wrapper=True)


class _RowSpy:
    """A per-row map that keeps the tag and records the shape it was handed."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.seen: list[tuple[int, ...]] = []

    def __call__(self, x, mask=None, transformer_options=None):
        self.seen.append(tuple(x.shape))
        return _keep_tag(_channel_map(x, self.seed), x[..., :1])


class _Identity(_RowSpy):
    """The projector: Linear(layers, 1) over the layer axis, which is 1 here."""

    def __call__(self, x, mask=None, transformer_options=None):
        self.seen.append(tuple(x.shape))
        return x


class _ToyTextBlock(_ToyBlock):
    """A refiner block: text-only self-attention under the options it is given."""

    def __call__(self, x, mask=None, transformer_options=None):
        mod = _Modulation(self.index)
        q, k, v = self.qkv(x, mod)
        attended = _toy_attention(("refiner", self.index), q, k, v, None,
                                  transformer_options or {})
        return self.residual(x, attended, mod)


class _ToyTextFusion:
    """Stock TextFusionTransformer.forward over spies, op for op."""

    def __init__(self) -> None:
        self.layerwise_blocks = [_RowSpy(20 + i) for i in range(2)]
        self.projector = _Identity(0)
        self.refiner_blocks = [_ToyTextBlock(10 + i) for i in range(2)]
        self.inputs: list[torch.Tensor] = []
        self.masks: list = []
        self.options: list = []

    def __call__(self, x, mask=None, transformer_options=None):
        self.inputs.append(x.clone())
        self.masks.append(mask)
        self.options.append(transformer_options)
        b, rows, n, d = x.shape
        x = x.reshape(b * rows, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), mask=None, transformer_options=transformer_options)
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=rows)
        x = self.projector(x).squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x, mask=mask, transformer_options=transformer_options)
        return x


class _ToyJointBlock(_ToyBlock):
    def __call__(self, x, vec, freqs, mask=None, transformer_options=None):
        q, k, v = self.qkv(x, vec)
        attended = _toy_attention(("joint", self.index), q, k, v, freqs,
                                  transformer_options or {})
        return self.residual(x, attended, vec)


class _ToyKrea2:
    """Krea2-shaped where it matters: one text layer of DIM features, patch 1
    and DIM channels, so the image tokens are the latent's pixels and keep
    their tags through `first`, the joint blocks and `last`."""

    patch = 1
    channels = DIM
    tdim = 4
    default_ref_method = None
    first = staticmethod(lambda img: img)
    tmlp = staticmethod(lambda t: t)
    tproj = staticmethod(lambda t: _Modulation(1))

    def __init__(self, blocks: int = 2) -> None:
        self.txtfusion = _ToyTextFusion()
        self.txtmlp = _RowSpy(7)
        self.blocks = [_ToyJointBlock(i) for i in range(blocks)]

    @staticmethod
    def _unpack_context(context):
        b, seq, fused = context.shape
        return context.reshape(b, seq, 1, fused)

    @staticmethod
    def pe_embedder(pos):
        # Folds each row's position into its query, so a misaligned id shard shows.
        return 0.01 * pos.to(torch.float64).sum(dim=-1, keepdim=True)

    @staticmethod
    def last(combined, t):
        return _channel_map(combined, 9)


def _streams(text_rows, grid, batch=1):
    """Tagged text rows 1..T and image latent whose tokens carry T+1..T+I."""
    image_rows = grid[0] * grid[1]
    text = torch.zeros(BATCH, text_rows, DIM, dtype=torch.float64)
    image = torch.zeros(BATCH, image_rows, DIM, dtype=torch.float64)
    text[0, :, _TAG] = torch.arange(1, text_rows + 1, dtype=torch.float64)
    image[0, :, _TAG] = torch.arange(text_rows + 1, text_rows + image_rows + 1, dtype=torch.float64)
    for column in range(1, DIM):
        text[0, :, column] = 0.1 * column + 0.03 * torch.arange(text_rows, dtype=torch.float64)
        image[0, :, column] = 0.2 * column - 0.05 * torch.arange(image_rows, dtype=torch.float64)
    text, image = text.repeat(batch, 1, 1), image.repeat(batch, 1, 1)
    for item in range(1, batch):
        text[item, :, _TAG + 1:] *= 1.0 + 0.1 * item
        image[item, :, _TAG + 1:] *= 1.0 - 0.07 * item
    latent = image.transpose(1, 2).reshape(batch, DIM, *grid)
    return text, latent


def _drive(harness, text, latent, *, pure_ulysses, blocks=2, attention_mask=None, options=None):
    """Run the real bound forward once per rank; each rank has its own model.

    `attention_mask` and `options` reach `_forward` as the caller's arguments,
    so a test can check which of them the text path was handed.
    """
    models = [_ToyKrea2(blocks) for _ in range(harness.world)]
    kernel = base.make_usp_attention("TORCH_FLASH")
    orders, drops = [], []

    def attn(*args, **kwargs):
        orders.append((_CONTEXT.where[0], kwargs.get("sequence_order")))
        drops.append((_CONTEXT.where[0], kwargs.get("drop_rows")))
        return kernel(*args, **kwargs)

    for model in models:
        Krea2Adapter().inject_usp(model, InjectionContext(
            topology_sp=harness.world, usp_attention=attn, pure_ulysses=pure_ulysses))
    timesteps = torch.zeros(latent.shape[0], dtype=torch.float64)
    outputs: list = [None] * harness.world
    errors: list = [None] * harness.world

    def body(rank):
        try:
            _LOCAL.rank = rank
            outputs[rank] = models[rank]._forward(latent, timesteps, text, attention_mask, None,
                                                  dict(options or {}))
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
    harness.orders, harness.drops, harness.models = orders, drops, models
    return outputs


def _run(monkeypatch, text, latent, *, world, pure_ulysses, **drive):
    harness = _Harness(world, strict=False)
    _install_stubs(monkeypatch, harness)
    return _drive(harness, text, latent, pure_ulysses=pure_ulysses, **drive), harness


def _local(rows):
    return -(-rows // WORLD)


def _rank_major_tags(text_rows, image_rows):
    """The joint key order a rank-major gather hands the kernel, pads removed."""
    order = []
    for rank in range(WORLD):
        for start, local, total, offset in (
                (rank * _local(text_rows), _local(text_rows), text_rows, 0),
                (rank * _local(image_rows), _local(image_rows), image_rows, text_rows)):
            order += [float(offset + start + i + 1) for i in range(local) if start + i < total]
    return order


def _joint(calls):
    return [call for call in calls if call[1] == "joint"]


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("text_rows,grid", SHAPES)
def test_joint_keys_arrive_in_stock_order_and_outputs_return_to_their_rank(
    monkeypatch, text_rows, grid, batch
):
    text, latent = _streams(text_rows, grid, batch)
    image_rows = grid[0] * grid[1]
    reference, _ = _run(monkeypatch, text, latent, world=1, pure_ulysses=False)
    outputs, harness = _run(monkeypatch, text, latent, world=WORLD, pure_ulysses=True)
    stock = [float(t) for t in range(1, text_rows + image_rows + 1)]
    calls = _joint(harness.calls)
    assert len(calls) == WORLD * 2
    # All text rows then all image rows, pad rows gone, through the full-axis
    # path rather than xfuser's own call.
    assert all(entry == "exact" and tags == stock for entry, _kind, _index, tags in calls)
    local = {(order.text_rows, order.image_rows)
             for kind, order in harness.orders if kind == "joint"}
    assert local == {(_local(text_rows), _local(image_rows))}
    # The inverse permutation returned each row to its rank: every rank holds
    # the whole image, and it is the one-rank answer for each batch item.
    assert all(torch.equal(out, outputs[0]) for out in outputs)
    assert _matches(outputs[0], reference[0])


@pytest.mark.parametrize("text_rows,grid", [(4, (2, 4)), (6, (2, 4))])
def test_an_evenly_sharded_render_takes_the_full_axis_path_and_matches_one_rank(
    monkeypatch, text_rows, grid
):
    """No pad anywhere: the order is the only difference."""
    text, latent = _streams(text_rows, grid)
    reference, _ = _run(monkeypatch, text, latent, world=1, pure_ulysses=False)
    outputs, harness = _run(monkeypatch, text, latent, world=WORLD, pure_ulysses=True)
    stock = [float(t) for t in range(1, text_rows + grid[0] * grid[1] + 1)]
    calls = _joint(harness.calls)
    assert calls and all(entry == "exact" and tags == stock for entry, _kind, _index, tags in calls)
    assert all(torch.equal(out, outputs[0]) for out in outputs)
    assert _matches(outputs[0], reference[0])


@pytest.mark.parametrize("text_rows,grid", SHAPES)
def test_without_pure_ulysses_the_joint_keys_stay_rank_major(monkeypatch, text_rows, grid):
    """A ring or hybrid worker context never receives the descriptor.

    The forward still names the joint and text drop rows there, so a padded
    render meets the ring pad refusal in base.py.
    """
    text, latent = _streams(text_rows, grid)
    image_rows = grid[0] * grid[1]
    _outputs, harness = _run(monkeypatch, text, latent, world=WORLD, pure_ulysses=False)
    expected = _rank_major_tags(text_rows, image_rows)
    assert all(order is None for _kind, order in harness.orders)
    # The negative control: rank-major differs from stock, so the test above
    # would fail on a forward that never reordered.
    assert expected != sorted(expected)
    assert all(tags == expected for *_head, tags in _joint(harness.calls))
    text_local, image_local = _local(text_rows), _local(image_rows)
    joint_drop = base.padded_row_indices([(text_rows, text_local), (image_rows, image_local)])
    assert {tuple(rows) for kind, rows in harness.drops if kind == "joint"} == {tuple(joint_drop)}


@pytest.mark.parametrize("text_rows,grid", SHAPES)
def test_the_refiner_never_takes_the_joint_order(monkeypatch, text_rows, grid):
    """The descriptor names a two-segment layout; the refiner attends one."""
    text, latent = _streams(text_rows, grid)
    _outputs, harness = _run(monkeypatch, text, latent, world=WORLD, pure_ulysses=True)
    assert all(order is None for kind, order in harness.orders if kind != "joint")
