"""Boogu divisibility pad rows are not attended (pure Ulysses).

``shard_seq`` zero-pads an odd stream, and after RMSNorm and the modulation
scale a zero row is still a real key at its RoPE position. Stock one-GPU
attention has no such key. Boogu shards five streams (noise, reference,
instruct, the joined [reference, noise] image stream, and the re-joined single
stream), and each loop names its own pad rows. The harness is the one in
tests/test_boogu_ulysses_order.py; in any stage, its strict recorder fails a
test when a pad row (tag 0) reaches attention or a real row is missing.
"""
from __future__ import annotations

import pytest
import torch

from dgx_monarch.adapters import base
from dgx_monarch.adapters.usp_pad_exclusion import keep_index
from test_boogu_ulysses_order import (  # noqa: F401  (fixture re-exported)
    SHAPES,
    WORLD,
    _matches,
    comfy_leaves,
    drive,
)

pytestmark = pytest.mark.usefixtures("comfy_leaves")


def _local(rows):
    return -(-rows // WORLD)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("text_rows,grid,refs", SHAPES)
def test_every_stage_attends_exactly_stocks_rows_and_matches_one_rank(
    monkeypatch, text_rows, grid, refs, batch
):
    reference, _ = drive(monkeypatch, text_rows, grid, refs, world=1, pure_ulysses=False,
                         batch=batch)
    outputs, harness = drive(monkeypatch, text_rows, grid, refs, world=WORLD,
                             pure_ulysses=True, batch=batch)
    stages = {kind for _e, kind, _i, _t in harness.calls}
    assert stages == {"noise", "joint", "image", "single"} | ({"ref"} if refs else set())
    # The strict recorder already rejects pad tags; this pins the exact keys
    # in stock's order for every stage.
    for _entry, kind, _index, tags in harness.calls:
        assert tags == harness.stages[kind], kind
    assert all(torch.equal(out, outputs[0]) for out in outputs)
    assert _matches(outputs[0], reference[0])


@pytest.mark.parametrize("text_rows,grid,refs", SHAPES)
def test_each_call_names_its_own_pad_set(monkeypatch, text_rows, grid, refs):
    """Worked from the stream lengths, independent of the forward.

    The image call's set is the image stream's tail and nothing else, so no
    joint coordinate ever reaches it. The joint set, once the descriptor maps
    it, is stock's [instruct pads, image pads] after the real rows of each.
    """
    _outputs, harness = drive(monkeypatch, text_rows, grid, refs, world=WORLD,
                              pure_ulysses=True)
    noise, ref = grid[0] * grid[1], sum(refs)
    image, joint = ref + noise, text_rows + ref + noise
    text_local, image_local = _local(text_rows), _local(image)
    tails = {"noise": list(range(noise, WORLD * _local(noise))),
             "ref": list(range(ref, WORLD * _local(ref))),
             "image": list(range(image, WORLD * image_local)),
             "single": list(range(joint, WORLD * _local(joint)))}
    for stage, order, drops in harness.records:
        if stage != "joint":
            assert drops == tails[stage], stage
            continue
        gathered = WORLD * (text_local + image_local)
        probe = torch.arange(gathered, dtype=torch.float32).reshape(1, -1, 1, 1)
        *_qkv, canonical, _kv, _perm = order.canonicalize(probe, probe, probe, drops, None, None)
        assert canonical == (list(range(text_rows, WORLD * text_local))
                             + list(range(WORLD * text_local + image, gathered)))


@pytest.mark.parametrize("text_rows,grid,refs", SHAPES)
def test_ring_and_hybrid_name_no_pad_rows_and_never_refuse(
    monkeypatch, text_rows, grid, refs
):
    """The ring pad guard refuses named rows; without pure Ulysses none are named."""
    _outputs, harness = drive(monkeypatch, text_rows, grid, refs, world=WORLD,
                              pure_ulysses=False, strict=False)
    assert harness.records and all(drops is None for _s, _o, drops in harness.records)


def test_the_template_shapes_worked_by_hand(monkeypatch):
    """The shipped 1024 template: 109 instruct rows, 4096 image rows.

    The joint pad is rank 1's last instruct row, rank-major 2103 + 54, which
    is stock's row 109, between instruct and image. That coordinate lies inside
    the 4096-row image axis, so the joint set handed to the image call would
    drop a real image row with no error. That is why each call names its own.
    """
    monkeypatch.setattr(base, "sp_world", lambda: WORLD)
    joint = base.padded_row_indices([(109, 55), (4096, 2048)])
    assert joint == [2157]
    assert keep_index(4096, joint, "cpu").numel() == 4095   # a real row lost
    assert base.padded_row_indices([(4096, 2048)]) == []
    assert base.padded_row_indices([(109 + 4096, 2103)]) == [4205]
    assert base.padded_row_indices([(66 + 4096, 2081)]) == []


def test_the_ring_pad_guard_the_gate_avoids_still_refuses():
    """The ring pad guard refuses a named pad row."""
    with pytest.raises(base.UnsupportedModelError):
        base.assert_ulysses_only_padding(2, 1)
