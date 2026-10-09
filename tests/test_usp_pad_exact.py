"""The pad rows: their gathered coordinates, and dropping them exactly.

`padded_row_indices` gives the gathered-sequence coordinates of shard_seq's
pads; getting them wrong would drop real tokens, so the arithmetic is pinned
against hand-computed layouts. `attend_without_pads` removes them at the
full-sequence point, and the second half of this file proves in float64 that
removing them reproduces the unsharded answer, on a self-attention and on a
cross attention whose two streams padded by different amounts.
"""
import math

import pytest
import torch

import dgx_monarch.adapters.base as base
from dgx_monarch.adapters.usp_pad_exclusion import (
    assert_pads_are_a_tail,
    attend_without_pads,
    keep_index,
)


@pytest.fixture
def world2(monkeypatch):
    monkeypatch.setattr(base, "sp_world", lambda: 2)


def test_no_padding_yields_no_rows(world2):
    # krea2 at 1536: txt 512 -> 256/rank, img 9216 -> 4608/rank
    assert base.padded_row_indices([(512, 256), (9216, 4608)]) == []


def test_image_pad_is_the_final_gathered_row(world2):
    # krea2 at 1448: img 91*91 = 8281 (odd) -> local 4141, one pad row.
    # Gathered: [txt0(256), img0(4141), txt1(256), img1(4141)] = 8794 rows;
    # the pad is the last one.
    assert base.padded_row_indices([(512, 256), (8281, 4141)]) == [8793]


def test_text_pad_lands_inside_the_last_chunk(world2):
    # Odd text length 511 -> local 256, pad 1. The pad row is the tail of the
    # last rank's text slice: chunk = 256+4141 = 4397, so row 4397+256-1.
    assert base.padded_row_indices([(511, 256), (8281, 4141)]) == [4652, 8793]


def test_multi_row_pad_is_contiguous(world2):
    # A hand-built (8275, 4139) segment pads 2*4139 - 8275 = 3 rows; shard_seq
    # at world 2 never pads more than one.
    rows = base.padded_row_indices([(512, 256), (8275, 4139)])
    assert rows == [8787, 8788, 8789]
    chunk = 256 + 4139
    assert rows[-1] == 2 * chunk - 1


def test_world4_layout(monkeypatch):
    monkeypatch.setattr(base, "sp_world", lambda: 4)
    # txt 510 -> local 128 (pad 2), img 8281 -> local 2071 (pad 3).
    chunk = 128 + 2071
    base_last = 3 * chunk
    rows = base.padded_row_indices([(510, 128), (8281, 2071)])
    assert rows == [base_last + 126, base_last + 127,
                    base_last + 128 + 2071 - 3, base_last + 128 + 2071 - 2,
                    base_last + 128 + 2071 - 1]


def test_world1_never_pads(monkeypatch):
    monkeypatch.setattr(base, "sp_world", lambda: 1)
    assert base.padded_row_indices([(511, 511), (8281, 8281)]) == []


def test_single_segment_matches_tail(world2):
    # The txtfusion refiner attends text only: one segment.
    assert base.padded_row_indices([(511, 256)]) == [511]
    assert base.padded_row_indices([(512, 256)]) == []


@pytest.mark.parametrize("ring_world", [2, 3, 4])
def test_padded_sequence_on_ring_or_hybrid_refuses(ring_world):
    """Ring cannot exclude pad rows exactly; rendering them anyway would
    reproduce the left-edge corruption measured on 2026-07-10. auto reaches
    ring/hybrid on worlds ulysses does not tile (world 3 gives ring3), so the
    refusal must be typed, not a warning."""
    with pytest.raises(base.UnsupportedModelError, match="ulysses-only"):
        base.assert_ulysses_only_padding(ring_world, 1)


def test_unpadded_or_pure_ulysses_padding_passes():
    base.assert_ulysses_only_padding(1, 3)   # pure ulysses handles pads exactly
    base.assert_ulysses_only_padding(2, 0)   # ring without pads is untouched
    base.assert_ulysses_only_padding(1, 0)


def test_production_padded_ring_refuses_before_any_kernel_call(monkeypatch):
    """Drive the real usp_attention callable: with a ring topology and pad
    rows present, the typed refusal must fire before the kernel wrapper or
    any all-to-all runs. Hermetic: xfuser and yunchang are stubbed in
    sys.modules, so the test needs no process group and no accelerator."""
    import sys
    import types

    kernel_calls = []

    class StubUSP:
        def __init__(self, *args, **kwargs):
            self.captured = kwargs

        def __call__(self, attn, q_l, k_l, v_l, **kwargs):
            kernel_calls.append("usp")
            return torch.zeros_like(q_l)  # (B, L_local, H, D)

    xfuser = types.ModuleType("xfuser")
    xcore = types.ModuleType("xfuser.core")
    xdist = types.ModuleType("xfuser.core.distributed")
    xdist.get_ring_parallel_world_size = lambda: 2
    xlca = types.ModuleType("xfuser.core.long_ctx_attention")
    xlca.xFuserLongContextAttention = StubUSP
    xfuser.core = xcore
    xcore.distributed = xdist
    xcore.long_ctx_attention = xlca
    yunchang = types.ModuleType("yunchang")
    ykernels = types.ModuleType("yunchang.kernels")
    ykernels.AttnType = {"TORCH_FLASH": "TORCH_FLASH"}
    yunchang.kernels = ykernels
    for name, mod in (("xfuser", xfuser), ("xfuser.core", xcore),
                      ("xfuser.core.distributed", xdist),
                      ("xfuser.core.long_ctx_attention", xlca),
                      ("yunchang", yunchang), ("yunchang.kernels", ykernels)):
        monkeypatch.setitem(sys.modules, name, mod)

    attn = base.make_usp_attention("TORCH_FLASH")
    q = torch.zeros(1, 2, 8, 4)  # (B, H, L_local, D)
    with pytest.raises(base.UnsupportedModelError, match="ulysses-only"):
        attn(q, q, q, heads=2, skip_reshape=True, drop_rows=[15])
    assert kernel_calls == []
    # Without pad rows the same topology proceeds straight to the kernel.
    attn(q, q, q, heads=2, skip_reshape=True, drop_rows=[])
    assert len(kernel_calls) == 1


def test_krea2_auto_world4_batch1_keeps_ring_fold_until_uly4_is_validated():
    """Every Krea2 head count (48 query, 12 KV, 20 in the txtfusion refiner)
    tiles 4, but uly4 has no 4-rank hardware validation (the rig has two
    GB10s). auto must not pick an unvalidated topology, so it keeps the ring
    fold, which needs no head count to divide, and padded renders refuse there
    with the typed message. An operator can still choose `uly4` by name."""
    from dgx_monarch.topology import choose_auto_topology

    decision = choose_auto_topology(
        "krea2", "fp8", 2.1, world=4, cfg_value=1.0, batch_size=1)
    topo = decision.topology
    assert (topo.ulysses, topo.ring, topo.dp) == (2, 2, 1)


def test_krea2_world4_folds_to_uly4_once_the_degree_is_validated(monkeypatch):
    """A family-and-degree-bound grant can activate an exact Ulysses fold."""
    import dgx_monarch.topology as topology_mod

    monkeypatch.setattr(
        topology_mod,
        "VALIDATED_ULYSSES_FOLDS",
        {"krea2": {4: (48, 12, 20)}},
    )
    decision = topology_mod.choose_auto_topology(
        "krea2", "fp8", 2.1, world=4, cfg_value=1.0, batch_size=1)
    topo = decision.topology
    assert (topo.ulysses, topo.ring, topo.dp) == (4, 1, 1)
    assert "ulysses4" in decision.reason


def test_krea2_auto_world3_batch1_stays_pure_ring():
    """World 3 does not tile the degree-2 row, so auto falls back to pure
    ring3, where a padded sequence refuses with the typed error instead of
    rendering. No grant changes this: a fold acts only on leftover
    data-parallel ranks, and ring3 leaves none. Krea2 could not run uly3
    anyway: its txtfusion refiner has 20 heads, and 3 does not divide 20."""
    from dgx_monarch.topology import choose_auto_topology

    decision = choose_auto_topology(
        "krea2", "fp8", 2.1, world=3, cfg_value=1.0, batch_size=1)
    topo = decision.topology
    assert (topo.ulysses, topo.ring) == (1, 3)


def test_cfg_row_never_folds_into_an_unvalidated_composite():
    """Chroma bf16 at 1.0 MP, CFG 4, world 4, batch 1: table row 20 gives
    cfg2+dp2. Neither fold may run: the evidence covers pure cfg2 and pure
    uly2, never uly2+cfg2, and ring2+cfg2 is the same composite with the other
    kernel (refused since 2026-08-20). auto refuses and names the presets that
    tile the world; uly2+cfg2 is among them, because a typed preset is the
    operator's choice."""
    from dgx_monarch.adapters.base import UnsupportedModelError
    from dgx_monarch.topology import choose_auto_topology

    with pytest.raises(UnsupportedModelError) as excinfo:
        choose_auto_topology("chroma", "bf16", 1.0, world=4, cfg_value=4.0, batch_size=1)
    message = str(excinfo.value)
    assert "cfg-plus-sequence" in message
    assert "uly2+cfg2" in message


def test_family_without_an_exact_fold_grant_keeps_the_ring_fold():
    from dgx_monarch.topology import choose_auto_topology

    decision = choose_auto_topology(
        "chroma", "fp8", 2.1, world=4, cfg_value=1.0, batch_size=1)
    topo = decision.topology
    assert (topo.ulysses, topo.ring, topo.dp) == (2, 2, 1)


# The row algebra, in float64 against a hand-rolled softmax, so a difference
# is the algebra rather than a kernel's accumulation order. No comfy, no CUDA,
# no xfuser: attend_without_pads takes the kernel as an argument.

BATCH, HEADS, DIM = 1, 4, 8


def _tokens(rows, *, seed=0):
    """(B, L, H, D) inputs, the layout the Ulysses head scatter produces."""
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randn(BATCH, rows, HEADS, DIM, generator=generator,
                    dtype=torch.float64)
        for _ in range(3)
    )


def _attend(q, k, v, groups=None):
    q_h, k_h, v_h = (t.transpose(1, 2) for t in (q, k, v))
    scores = q_h @ k_h.transpose(-1, -2) / math.sqrt(q_h.shape[-1])
    return (torch.softmax(scores, dim=-1) @ v_h).transpose(1, 2)


def _zero_pad(t, rows):
    """What shard_seq adds: `rows` synthetic rows at the tail of the axis."""
    return torch.cat([t, t.new_zeros(t.shape[0], rows, *t.shape[2:])], dim=1)


def test_keep_index_is_none_when_there_is_nothing_to_drop():
    """Skip gathering when there are no pad rows."""
    assert keep_index(6, [], torch.device("cpu")) is None
    assert keep_index(6, None, torch.device("cpu")) is None
    assert torch.equal(keep_index(4, [1], torch.device("cpu")),
                       torch.tensor([0, 2, 3]))


def test_dropping_the_tail_pads_reproduces_the_unpadded_self_attention():
    """Dropping the tail pads reproduces the unpadded self-attention, on the shape krea2 and LTX video share.

    Here a pad row is a zero key, so its score against every query is exactly
    0, which softmax still gives real weight. In a render the pad is a zero
    embedding, which after modulation is not a zero key. Attending it is the
    left-edge corruption measured on 2026-07-10; dropping it gives the
    unsharded answer.
    """
    real, pads = 9, 1
    q, k, v = _tokens(real, seed=21)
    padded = [_zero_pad(t, pads) for t in (q, k, v)]
    drop = list(range(real, real + pads))

    excluded = attend_without_pads(*padded, drop_rows=drop, kv_drop_rows=drop,
                                   groups=None, attend=_attend)

    assert torch.allclose(excluded[:, :real], _attend(q, k, v), atol=1e-12)
    assert torch.equal(excluded[:, real:], torch.zeros_like(excluded[:, real:]))
    attended = _attend(*padded)
    assert (attended[:, :real] - _attend(q, k, v)).abs().max() > 1e-3


def test_a_cross_attention_drops_its_own_rows_on_each_side():
    """LTX audio-to-video: video queries, audio keys, each padded on its own.

    One row set cannot serve this. The key axis is a different length from the
    query axis and pads by a different amount, so the query set can index real
    audio rows and leave synthetic ones in the kernel; here it drops one of the
    three audio pads and keeps two.
    """
    video, audio = 7, 5
    q, _, _ = _tokens(video, seed=31)
    _, k, v = _tokens(audio, seed=37)
    padded = (_zero_pad(q, 1), _zero_pad(k, 3), _zero_pad(v, 3))

    out = attend_without_pads(*padded, drop_rows=[video],
                              kv_drop_rows=[audio, audio + 1, audio + 2],
                              groups=None, attend=_attend)

    assert torch.allclose(out[:, :video], _attend(q, k, v), atol=1e-12)
    assert torch.equal(out[:, video:], torch.zeros_like(out[:, video:]))
    one_set = attend_without_pads(*padded, drop_rows=[video],
                                  kv_drop_rows=[video], groups=None,
                                  attend=_attend)
    assert (one_set[:, :video] - out[:, :video]).abs().max() > 1e-3


def test_the_query_side_and_the_key_side_are_read_separately():
    """A key stream can pad while its query stream does not."""
    video, audio = 6, 5
    q, _, _ = _tokens(video, seed=41)
    _, k, v = _tokens(audio, seed=43)

    out = attend_without_pads(q, _zero_pad(k, 1), _zero_pad(v, 1),
                              drop_rows=None, kv_drop_rows=[audio],
                              groups=None, attend=_attend)

    assert out.shape[1] == video
    assert torch.allclose(out, _attend(q, k, v), atol=1e-12)


def test_a_bias_over_pads_away_from_the_tail_refuses_rather_than_sliding():
    """Group boundaries are sequence coordinates, so a middle drop moves them.

    No shipped stream has that layout. One that ever did would have to be
    measured, so it stops here instead of biasing the wrong tokens.
    """
    assert assert_pads_are_a_tail([8, 9], 10) is None
    assert assert_pads_are_a_tail([], 10) is None
    with pytest.raises(base.UnsupportedModelError, match="away from the tail"):
        assert_pads_are_a_tail([4, 5], 10)


# The dispatch, driven through the real usp_attention callable with xfuser and
# yunchang stubbed in sys.modules.

def _stub_usp(monkeypatch, ring_world=1):
    """Install the stubs and return (callable, rows the kernel was handed)."""
    import sys
    import types

    seen = {}

    class StubUSP:
        ulysses_pg = ring_pg = attn_type = attn_processor = None
        q_descale = k_descale = v_descale = None

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, attn, q_l, k_l, v_l, **kwargs):
            seen["whole"] = (q_l.shape[1], k_l.shape[1])
            return torch.zeros_like(q_l)

        @staticmethod
        def ring_attn_fn(q, k, v, **kwargs):
            seen["kept"] = (q.shape[1], k.shape[1])
            return torch.zeros_like(q)

    class StubAllToAll:
        @staticmethod
        def apply(_group, tensor, _a, _b):
            return tensor

    modules = {
        "xfuser": types.ModuleType("xfuser"),
        "xfuser.core": types.ModuleType("xfuser.core"),
        "xfuser.core.distributed": types.ModuleType("xfuser.core.distributed"),
        "xfuser.core.long_ctx_attention": types.ModuleType(
            "xfuser.core.long_ctx_attention"),
        "yunchang": types.ModuleType("yunchang"),
        "yunchang.kernels": types.ModuleType("yunchang.kernels"),
        "yunchang.comm": types.ModuleType("yunchang.comm"),
        "yunchang.comm.all_to_all": types.ModuleType("yunchang.comm.all_to_all"),
    }
    modules["xfuser.core.distributed"].get_ring_parallel_world_size = (
        lambda: ring_world)
    modules["xfuser.core.long_ctx_attention"].xFuserLongContextAttention = StubUSP
    modules["yunchang.kernels"].AttnType = {"TORCH_FLASH": "TORCH_FLASH"}
    modules["yunchang.comm.all_to_all"].SeqAllToAll4D = StubAllToAll
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return base.make_usp_attention("TORCH_FLASH"), seen


def test_a_self_attention_key_side_defaults_to_the_query_row_set(monkeypatch):
    """Self-attention callers (krea2, Lens and H3 among them) pass one set.

    Their queries and keys come from one stream, so keys drop where queries do.
    Defaulting here keeps their op sequence unchanged, so their measured
    pad-exclusion contracts (2026-07-10 onward) still hold without a new run.
    """
    attn, seen = _stub_usp(monkeypatch)
    q = torch.zeros(1, 2, 9, 4)  # (B, H, L, D)

    attn(q, q, q, heads=2, skip_reshape=True, drop_rows=[8])

    assert seen["kept"] == (8, 8)


def test_pads_on_the_key_side_alone_still_take_the_exact_path(monkeypatch):
    """LTX video-to-audio when only the video stream padded.

    Nothing drops on the query side, so a dispatch that keyed off `drop_rows`
    alone would send this straight to the maskless kernel with the pad keys in.
    """
    attn, seen = _stub_usp(monkeypatch)
    q = torch.zeros(1, 2, 6, 4)
    k = torch.zeros(1, 2, 9, 4)

    attn(q, k, k, heads=2, skip_reshape=True, kv_drop_rows=[8])

    assert seen["kept"] == (6, 8)
    assert "whole" not in seen


def test_an_unpadded_render_keeps_the_untouched_xfuser_call(monkeypatch):
    """No pads and no bias must reach xfuser's own callable unchanged."""
    attn, seen = _stub_usp(monkeypatch)
    q = torch.zeros(1, 2, 8, 4)

    attn(q, q, q, heads=2, skip_reshape=True, drop_rows=[], kv_drop_rows=[])

    assert seen == {"whole": (8, 8)}


def test_the_merged_path_with_only_drop_rows_matches_the_unpadded_reference():
    """Pins the `drop_rows`-only call on a pad set that is not a tail.

    A joint [text, image] stream puts the text pad inside the last rank's text
    slice, so its set is scattered. This drives the real usp_attention with
    (drop_rows, None, None), lets the key side default, and checks the kept rows
    against attention over the real rows alone in float64. The kernel is a stub,
    so a difference here is the row algebra rather than accumulation order.
    """
    import sys
    import types

    class StubUSP:
        ulysses_pg = ring_pg = attn_type = attn_processor = None
        q_descale = k_descale = v_descale = None

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, attn, q_l, k_l, v_l, **kwargs):
            raise AssertionError("a padded ulysses call must not reach xfuser")

        @staticmethod
        def ring_attn_fn(q, k, v, **kwargs):
            return _attend(q, k, v)

    class StubAllToAll:
        @staticmethod
        def apply(_group, tensor, _a, _b):
            return tensor

    modules = {
        "xfuser": types.ModuleType("xfuser"),
        "xfuser.core": types.ModuleType("xfuser.core"),
        "xfuser.core.distributed": types.ModuleType("xfuser.core.distributed"),
        "xfuser.core.long_ctx_attention": types.ModuleType(
            "xfuser.core.long_ctx_attention"),
        "yunchang": types.ModuleType("yunchang"),
        "yunchang.kernels": types.ModuleType("yunchang.kernels"),
        "yunchang.comm": types.ModuleType("yunchang.comm"),
        "yunchang.comm.all_to_all": types.ModuleType("yunchang.comm.all_to_all"),
    }
    modules["xfuser.core.distributed"].get_ring_parallel_world_size = lambda: 1
    modules["xfuser.core.long_ctx_attention"].xFuserLongContextAttention = StubUSP
    modules["yunchang.kernels"].AttnType = {"TORCH_FLASH": "TORCH_FLASH"}
    modules["yunchang.comm.all_to_all"].SeqAllToAll4D = StubAllToAll
    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        attn = base.make_usp_attention("TORCH_FLASH")
        # Two pad rows that are not a tail, like the scattered set of
        # test_text_pad_lands_inside_the_last_chunk.
        rows, drop = 14, [5, 13]
        q, k, v = _tokens(rows, seed=51)
        for row in drop:                       # what pad_seq_to_multiple leaves
            q[:, row] = 0
            k[:, row] = 0
            v[:, row] = 0
        # (B, L, H, D) -> the (B, H, L, D) surface skip_reshape declares.
        out = attn(*(t.transpose(1, 2) for t in (q, k, v)), heads=HEADS,
                   skip_reshape=True, skip_output_reshape=True, drop_rows=drop)
        out = out.transpose(1, 2)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    keep = [row for row in range(rows) if row not in drop]
    reference = _attend(q[:, keep], k[:, keep], v[:, keep])
    assert torch.allclose(out[:, keep], reference, atol=1e-12)
    assert torch.equal(out[:, drop], torch.zeros_like(out[:, drop]))


def test_a_pad_row_past_the_end_of_its_axis_refuses_before_the_scatter():
    """Crossed row sets abort the worker.

    The index reaches a CUDA scatter, so addressing a row the axis does not
    have raises a device-side assert, kills the process with SIGABRT and takes
    the fleet with it (seen on hardware 2026-08-13: LTX audio-to-video with
    7905 video rows sent video pad row 7905 to a 118-row audio key axis).
    Reading the range first turns that into a message naming both numbers.
    """
    with pytest.raises(base.UnsupportedModelError, match="carries 118 rows"):
        keep_index(118, [7905], torch.device("cpu"))
    with pytest.raises(base.UnsupportedModelError, match="dgx-monarch fault"):
        keep_index(4, [-1], torch.device("cpu"))
    assert keep_index(4, [3], torch.device("cpu")).tolist() == [0, 1, 2]


def test_a_padded_query_stream_against_an_unpadded_key_stream():
    """Handle padded queries against unpadded keys.

    LTX audio-to-video takes video queries, which padded, against audio keys,
    which did not. The key side must get an empty set, never None:
    usp_attention reads None as the self-attention contract and fills it from
    the query side, which caused the abort above.
    """
    video, audio = 7, 5
    q, _, _ = _tokens(video, seed=61)
    _, k, v = _tokens(audio, seed=67)

    out = attend_without_pads(_zero_pad(q, 1), k, v, drop_rows=[video],
                              kv_drop_rows=[], groups=None, attend=_attend)

    assert torch.allclose(out[:, :video], _attend(q, k, v), atol=1e-12)
    assert torch.equal(out[:, video:], torch.zeros_like(out[:, video:]))
