"""Flux, Flux2 and LongCat double blocks attend in stock key order (pure Ulysses).

Each rank holds `[text_r, image_r]`, so Ulysses' head all-to-all hands the
kernel `[text0, image0, text1, image1]` where stock attends `[text0, text1,
image0, image1]`. The Chroma double loop restores stock order through a
`RankMajorJointOrder` descriptor; this file proves the shared flux
forward does the same for its three other families, on the real bound forward,
the real `base.make_usp_attention` dispatch and the real full-axis call, with
only the all-to-all, the gather and the kernel stubbed. Every key row carries
a tag, so the kernel can name the exact order it was handed.

The toy model, harness and stubs are the ones tests/test_flux_pad_exclusion.py
drives for pad exclusion; this file only adds what the order claim needs.
"""
from __future__ import annotations

import sys

import pytest
import torch

from dgx_monarch.adapters import flux_family
from dgx_monarch.adapters.base import usp_options
from dgx_monarch.adapters.usp_sequence_order import RankMajorJointOrder
from test_flux_pad_exclusion import (  # noqa: F401  (fixture re-exported)
    _TAG,
    ADAPTERS,
    _matches,
    _model_for,
    _reference,
    _sharded,
    _streams,
    comfy_layers_stub,
)

pytestmark = pytest.mark.usefixtures("comfy_layers_stub")

WORLD = 2
SHAPES = [(4, 8), (5, 8), (4, 9), (5, 9)]  # even, odd text, odd image, both odd


def _batched(streams, batch: int):
    """Repeat the streams along the batch axis, keeping every row's tag.

    Items after the first get a different feature body, so a permutation that
    treated the batch axis wrongly would show in the per-item outputs.
    """
    txt, txt_ids, img, img_ids = streams
    if batch == 1:
        return streams
    txt, img = txt.repeat(batch, 1, 1), img.repeat(batch, 1, 1)
    for b in range(1, batch):
        txt[b, :, _TAG + 1:] *= 1.0 + 0.1 * b
        img[b, :, _TAG + 1:] *= 1.0 - 0.07 * b
    return txt, txt_ids, img, img_ids


def _rank_major_tags(txt_len: int, img_len: int) -> list[float]:
    """The rank-major key order a double block sees with no descriptor, pads removed."""
    text_local, image_local = -(-txt_len // WORLD), -(-img_len // WORLD)
    order: list[float] = []
    for rank in range(WORLD):
        text = range(rank * text_local + 1, min(txt_len, (rank + 1) * text_local) + 1)
        image = range(txt_len + rank * image_local + 1,
                      txt_len + min(img_len, (rank + 1) * image_local) + 1)
        order += [float(tag) for tag in (*text, *image)]
    return order


@pytest.mark.parametrize("adapter_cls", ADAPTERS)
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("txt_len,img_len", SHAPES)
def test_double_block_keys_arrive_in_stock_order_and_outputs_return_to_their_rank(
    monkeypatch, adapter_cls, batch, txt_len, img_len
):
    model = _model_for(adapter_cls)
    streams = _batched(_streams(txt_len, img_len), batch)
    reference, _ = _reference(monkeypatch, adapter_cls, model, streams, txt_len, img_len)
    outputs, harness = _sharded(monkeypatch, adapter_cls, model, streams, txt_len, img_len,
                                world=WORLD, pure_ulysses=True)

    stock = [float(tag) for tag in range(1, txt_len + img_len + 1)]
    doubles = [call for call in harness.calls if call[1] == "double"]
    singles = [call for call in harness.calls if call[1] == "single"]
    assert len(doubles) == WORLD * len(model.double_blocks)
    assert len(singles) == WORLD * len(model.single_blocks)
    for entry, _kind, _index, tags in doubles:
        # [all text rows, then all image rows], every pad row dropped, and it
        # reached the kernel through the full-axis path, not xfuser's call.
        assert entry == "exact"
        assert tags == stock
    # Single blocks shard one joint stream, which is already stock order.
    assert all(tags == stock for _entry, _kind, _index, tags in singles)

    # The inverse permutation returned each row to the rank that owns it: every
    # rank holds the whole image, and it is the unsharded answer per batch item.
    assert reference.shape[0] == batch
    assert all(torch.equal(out, outputs[0]) for out in outputs)
    assert _matches(outputs[0], reference)


@pytest.mark.parametrize("adapter_cls", ADAPTERS)
def test_pure_ulysses_hands_the_descriptor_to_double_blocks_only(monkeypatch, adapter_cls):
    model = _model_for(adapter_cls)
    streams = _streams(5, 9)
    _outputs, harness = _sharded(monkeypatch, adapter_cls, model, streams, 5, 9,
                                 world=WORLD, pure_ulysses=True)
    # Local rows after the divisibility pad: 3 text and 5 image per rank.
    doubles = {(kind, order.text_rows, order.image_rows)
               for kind, order in harness.orders if isinstance(order, RankMajorJointOrder)}
    assert doubles == {("double", 3, 5)}
    assert all(order is None for kind, order in harness.orders if kind == "single")
    assert sum(1 for kind, _ in harness.orders if kind == "double") == (
        WORLD * len(model.double_blocks))


@pytest.mark.parametrize("adapter_cls", ADAPTERS)
@pytest.mark.parametrize("txt_len,img_len", SHAPES)
def test_without_pure_ulysses_the_double_blocks_keep_rank_major_order(
    monkeypatch, adapter_cls, txt_len, img_len
):
    """A ring or hybrid worker context never receives the descriptor."""
    model = _model_for(adapter_cls)
    streams = _streams(txt_len, img_len)
    reference, _ = _reference(monkeypatch, adapter_cls, model, streams, txt_len, img_len)
    outputs, harness = _sharded(monkeypatch, adapter_cls, model, streams, txt_len, img_len,
                                world=WORLD, pure_ulysses=False)

    assert all(order is None for _kind, order in harness.orders)
    expected = _rank_major_tags(txt_len, img_len)
    assert expected != sorted(expected)
    for _entry, kind, _index, tags in harness.calls:
        if kind == "double":
            assert tags == expected
    assert _matches(outputs[0], reference)


@pytest.mark.parametrize("adapter_cls", ADAPTERS)
def test_the_dispatcher_drops_the_descriptor_when_the_ring_degree_is_above_one(
    monkeypatch, adapter_cls
):
    """base.usp_attention's hybrid guard: at a ring degree above one, ring keeps its path."""
    model = _model_for(adapter_cls)
    streams = _streams(4, 8)
    # _install_stubs runs inside _sharded; patch the ring size after it ran by
    # wrapping the installer so the same stub module reports a ring of two.
    from test_flux_pad_exclusion import _install_stubs

    def with_ring(patcher, harness):
        _install_stubs(patcher, harness)
        sys.modules["xfuser.core.distributed"].get_ring_parallel_world_size = lambda: 2

    monkeypatch.setattr("test_flux_pad_exclusion._install_stubs", with_ring)
    _outputs, harness = _sharded(monkeypatch, adapter_cls, model, streams, 4, 8,
                                 world=WORLD, pure_ulysses=True)
    doubles = [call for call in harness.calls if call[1] == "double"]
    assert doubles and all(entry == "xfuser" for entry, *_rest in doubles)
    assert all(tags == _rank_major_tags(4, 8) for *_head, tags in doubles)


@pytest.mark.parametrize("adapter_cls", ADAPTERS)
def test_the_negative_control_shows_rank_major_order_without_the_descriptor(
    monkeypatch, adapter_cls
):
    """Without this the file would pass on a harness that never reordered."""
    monkeypatch.setattr(flux_family, "joint_double_options",
                        lambda options, attention, drop_rows, context, text, image:
                        usp_options(options, attention, drop_rows))
    model = _model_for(adapter_cls)
    _outputs, harness = _sharded(monkeypatch, adapter_cls, model, _streams(4, 8), 4, 8,
                                 world=WORLD, pure_ulysses=True)
    doubles = [tags for _entry, kind, _index, tags in harness.calls if kind == "double"]
    assert doubles and all(tags == _rank_major_tags(4, 8) for tags in doubles)
    assert doubles[0] != [float(tag) for tag in range(1, 13)]
