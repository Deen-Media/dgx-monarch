"""The padded Ulysses path builds its index tensors once and never reads the device.

A padded attention drops its pad rows by index, and a joint-order route also
permutes the gathered axis. Those indices depend only on integer row counts,
so they are built on the device once per layout and cached. Building them
with ``nonzero`` or reading the inverse back row by row made the host wait for
the stream four to six times per attention call. These tests pin the fix:
no read-back op runs on a cold or a warm call, a meta-device run (which
cannot read data back at all) completes, cached entries are keyed by layout
and device, and the cached values equal the per-call build's values, so the
output does not change.
"""
from __future__ import annotations

import math
import sys
import types

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import dgx_monarch.adapters.base as base
from dgx_monarch.adapters import usp_pad_exclusion, usp_sequence_order
from dgx_monarch.adapters.index_cache import BoundedIndexCache
from dgx_monarch.adapters.lens import LensJointOrder
from dgx_monarch.adapters.usp_full_axis import run_full_axis
from dgx_monarch.adapters.usp_pad_exclusion import attend_without_pads, keep_index
from dgx_monarch.adapters.usp_sequence_order import RankMajorJointOrder

WORLD = 2
CPU = torch.device("cpu")
META = torch.device("meta")
_aten = torch.ops.aten
# Ops that hand a device value to the host: on CUDA each one waits for the stream.
_READS = {_aten.nonzero.default: "nonzero",
          _aten._local_scalar_dense.default: "_local_scalar_dense"}


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch):
    monkeypatch.setattr(base, "sp_world", lambda: WORLD)
    usp_pad_exclusion._KEEP_INDEX.clear()
    usp_sequence_order._JOINT_INDEX.clear()
    yield
    usp_pad_exclusion._KEEP_INDEX.clear()
    usp_sequence_order._JOINT_INDEX.clear()


class _Reads(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.count = dict.fromkeys(_READS.values(), 0)

    def __torch_dispatch__(self, func, _types=(), args=(), kwargs=None):
        if func in _READS:
            self.count[_READS[func]] += 1
        return func(*args, **(kwargs or {}))


def _install_all_to_all(monkeypatch, gathered):
    class AllToAll:
        @staticmethod
        def apply(_group, tensor, scatter, gather):
            return gathered if (scatter, gather) == (2, 1) else tensor

    module = types.ModuleType("yunchang.comm.all_to_all")
    module.SeqAllToAll4D = AllToAll
    monkeypatch.setitem(sys.modules, "yunchang", types.ModuleType("yunchang"))
    monkeypatch.setitem(sys.modules, "yunchang.comm", types.ModuleType("yunchang.comm"))
    monkeypatch.setitem(sys.modules, "yunchang.comm.all_to_all", module)


def _attention(q, k, v, _groups=None):
    scores = torch.einsum("blhd,bmhd->bhlm", q, k) / q.shape[-1] ** 0.5
    return torch.einsum("bhlm,bmhd->blhd", torch.softmax(scores, dim=-1), v)


class _USP:
    ulysses_pg = ring_pg = attn_type = attn_processor = None
    q_descale = k_descale = v_descale = None

    def ring_attn_fn(self, q, k, v, **_kwargs):
        return _attention(q, k, v)


def _layout(kind, text, image):
    """(order, pad rows, gathered rows) as the adapters build them at world 2."""
    lt, li = math.ceil(text / WORLD), math.ceil(image / WORLD)
    if kind == "joint":     # flux-family double, krea2, qwen-image blocks
        return (RankMajorJointOrder(lt, li),
                base.padded_row_indices([(text, lt), (image, li)]), WORLD * (lt + li))
    if kind == "lens":      # image first, real rows before pad rows
        return (LensJointOrder(text_rows=lt, image_rows=li, image_real=image, text_real=text),
                base.padded_row_indices([(image, li), (text, lt)]), WORLD * (lt + li))
    joint = math.ceil((text + image) / WORLD)   # flux-family single block
    return None, base.padded_row_indices([(text + image, joint)]), WORLD * joint


def _full_axis(monkeypatch, kind, text, image, device, dtype=torch.float64):
    order, drop, rows = _layout(kind, text, image)
    gathered = torch.randn(1, rows, 2, 4, dtype=dtype, device=device)
    _install_all_to_all(monkeypatch, gathered)
    local = gathered[:, :rows // WORLD]
    # base.usp_attention hands a self-attention one row set for both sides.
    return lambda: run_full_axis(_USP(), local, local, local, drop, drop, None, order,
                                 _attention)


SHAPES = [("joint", 301, 4096), ("none", 301, 4096), ("lens", 301, 4096),
          ("joint", 300, 4225), ("joint", 7, 9), ("none", 7, 8), ("lens", 7, 9),
          ("joint", 300, 4096)]


def test_the_read_counter_sees_the_reads_it_guards_against():
    with _Reads() as reads:
        torch.ones(4, dtype=torch.bool).nonzero()
        int(torch.ones(()))
    assert reads.count == {"nonzero": 1, "_local_scalar_dense": 1}


@pytest.mark.parametrize("kind,text,image", SHAPES)
def test_the_padded_full_axis_path_reads_nothing_back(monkeypatch, kind, text, image):
    torch.manual_seed(5)
    call = _full_axis(monkeypatch, kind, text, image, CPU)
    for _ in ("cold", "warm"):
        with _Reads() as reads:
            out = call()
        assert reads.count == {"nonzero": 0, "_local_scalar_dense": 0}
    assert torch.equal(out, call())


@pytest.mark.parametrize("kind,text,image", [s for s in SHAPES if s[1] % 2])
def test_the_padded_full_axis_path_runs_on_the_meta_device(monkeypatch, kind, text, image):
    """Meta tensors hold no data, so any read back to the host raises."""
    call = _full_axis(monkeypatch, kind, text, image, META, torch.float32)
    for _ in ("cold", "warm"):
        out = call()
        # The stub's return all-to-all hands back the whole gathered axis.
        assert out.device == META and out.dtype == torch.float32
        assert out.shape == (1, _layout(kind, text, image)[2], 2, 4)


def test_the_keep_index_is_reused_per_layout_and_never_across_devices():
    first = keep_index(10, [3, 7], CPU)
    assert keep_index(10, [3, 7], CPU) is first
    assert keep_index(10, (3, 7), "cpu") is first
    assert keep_index(12, [3, 7], CPU) is not first
    assert keep_index(10, [3], CPU) is not first
    on_meta = keep_index(10, [3, 7], META)
    assert on_meta is not first and on_meta.device == META and first.device == CPU
    assert keep_index(10, [3, 7], "meta") is on_meta
    assert first.tolist() == [0, 1, 2, 4, 5, 6, 8, 9]


def test_a_self_attention_builds_one_keep_index_for_both_sides(monkeypatch):
    built = []
    real = usp_pad_exclusion.keep_index

    def counting(rows, drop_rows, device):
        built.append(rows)
        return real(rows, drop_rows, device)

    monkeypatch.setattr(usp_pad_exclusion, "keep_index", counting)
    q = torch.randn(1, 8, 1, 2, dtype=torch.float64)
    drop = [3, 5]
    attend_without_pads(q, q, q, drop_rows=drop, kv_drop_rows=drop, groups=None,
                        attend=_attention)
    assert built == [8]
    attend_without_pads(q, q, q, drop_rows=drop, kv_drop_rows=list(drop), groups=None,
                        attend=_attention)
    assert built == [8, 8, 8]
    k = torch.randn(1, 6, 1, 2, dtype=torch.float64)
    attend_without_pads(q, k, k, drop_rows=drop, kv_drop_rows=drop, groups=None,
                        attend=_attention)
    assert built == [8, 8, 8, 8, 6]


def test_the_keep_index_equals_the_per_call_build_on_every_layout():
    """The values the scatter and index_select see are unchanged, row for row."""
    checked = 0
    for text in range(1, 40):
        for image in (4, 9, 16, 25, 36):
            for kind in ("joint", "none", "lens"):
                _order, drop, rows = _layout(kind, text, image)
                if not drop:
                    continue
                keep = torch.ones(rows, dtype=torch.bool)
                keep[torch.as_tensor(drop)] = False
                assert torch.equal(keep_index(rows, drop, CPU), keep.nonzero(as_tuple=True)[0])
                checked += 1
    assert checked > 150
    assert keep_index(2, [0, 1], CPU).tolist() == []
    assert keep_index(5, [4, 4, 0], CPU).tolist() == [1, 2, 3]


def test_the_host_inverse_equals_the_device_inverse_for_both_orders():
    for text in range(1, 40):
        for image in (4, 9, 16, 25, 36):
            lt, li = math.ceil(text / WORLD), math.ceil(image / WORLD)
            rows = WORLD * (lt + li)
            for order in (RankMajorJointOrder(lt, li),
                          LensJointOrder(text_rows=lt, image_rows=li,
                                         image_real=image, text_real=text)):
                index = order.joint_index(rows, META)
                fresh = order.inverse(order.permutation(rows, CPU))
                assert list(index.host_inverse) == fresh.tolist()
                assert index.permutation.device == META == index.inverse.device
            parent = RankMajorJointOrder(lt, li).joint_index(rows, CPU)
            for p in range(rows):   # the closed form for the parent order
                r, j = divmod(p, lt + li)
                assert parent.host_inverse[p] == (
                    r * lt + j if j < lt else WORLD * lt + r * li + (j - lt))


def test_the_joint_index_is_reused_per_layout_and_never_across_devices():
    order = RankMajorJointOrder(3, 5)
    probe = torch.arange(16, dtype=torch.float64).reshape(1, 16, 1, 1)
    *_qkv, drop, kv_drop, permutation = order.canonicalize(probe, probe, probe, [7], [7], None)
    *_qkv, again, _kv, permutation_again = order.canonicalize(probe, probe, probe, [7], [7], None)
    assert permutation_again is permutation and again == drop == kv_drop
    rows32 = torch.zeros(1, 32, 1, 1)
    assert order.canonicalize(rows32, rows32, rows32, None, None, None)[5] is not permutation
    lens = LensJointOrder(text_rows=3, image_rows=5, image_real=10, text_real=6)
    assert lens.joint_index(16, CPU).permutation is not permutation
    assert order.joint_index(16, META).permutation is not permutation
    assert order.joint_index(16, "cpu").permutation is permutation
    same = [7]
    *_qkv, mapped, kv_mapped, _p = order.canonicalize(probe, probe, probe, same, same, None)
    assert kv_mapped is mapped


def test_restore_reuses_the_cached_inverse_and_still_inverts_a_foreign_permutation(monkeypatch):
    built = []
    real = RankMajorJointOrder.inverse

    def counting(self, permutation):
        built.append(permutation.numel())
        return real(self, permutation)

    monkeypatch.setattr(RankMajorJointOrder, "inverse", counting)
    order = RankMajorJointOrder(3, 5)
    probe = torch.arange(16, dtype=torch.float64).reshape(1, 16, 1, 1)
    canonical, *_rest, permutation = order.canonicalize(probe, probe, probe, None, None, None)
    assert built == [16]
    assert torch.equal(order.restore(canonical, permutation), probe)
    assert built == [16]
    foreign = order.permutation(16, CPU)
    assert torch.equal(order.restore(probe[:, foreign], foreign), probe)
    assert built == [16, 16]


def test_the_caches_are_bounded_and_drop_the_least_recently_used():
    cache = usp_pad_exclusion._KEEP_INDEX
    oldest = keep_index(100, [0], CPU)
    kept = keep_index(100, [1], CPU)
    for row in range(2, cache.limit + 8):
        keep_index(100, [row], CPU)
        assert keep_index(100, [1], CPU) is kept   # refreshed on every use
    assert len(cache) == cache.limit
    assert keep_index(100, [0], CPU) is not oldest
    assert usp_sequence_order._JOINT_INDEX.limit <= cache.limit

    small: BoundedIndexCache[int] = BoundedIndexCache(2)
    small.put("a", 1)
    small.put("b", 2)
    assert small.peek("a") == 1          # a peek does not refresh
    small.put("c", 3)
    assert small.get("a") is None and small.get("b") == 2 and len(small) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sync debug mode needs a CUDA device")
@pytest.mark.parametrize("kind", ["joint", "none", "lens"])
def test_the_padded_full_axis_path_never_waits_for_the_device_on_cuda(monkeypatch, kind):
    """Under sync debug mode "error" any host wait raises, cold build included."""
    torch.manual_seed(5)
    device = torch.device("cuda", torch.cuda.current_device())
    call = _full_axis(monkeypatch, kind, 301, 4096, device, torch.float32)
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        cold, warm = call(), call()
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert torch.equal(cold, warm)
