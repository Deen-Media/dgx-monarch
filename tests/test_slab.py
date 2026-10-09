"""Tests for actor/slab.py zero-copy weight slabs without comfy or a GPU.

Covers the safetensors header checks a slab relies on, slab layout and byte
fidelity, failure-atomic construction and close, reabsorb of strayed
parameters, the comfy_bridge.slab_load hook against a stubbed comfy, and the
slab and stock prices in capacity_fit and actor/store_slab_admit."""
import json
import os
import struct
import sys
import types
import weakref

import pytest
import torch
from torch import nn

from dgx_monarch.actor import slab as slab_module
from dgx_monarch.actor import unbake as unbake_module
from dgx_monarch.actor.slab import _ALIGN, WeightSlab
from dgx_monarch.safetensors_header import (
    SafetensorsHeaderError,
    UnsupportedSafetensorsDtypeError,
    read_safetensors_header,
)
from slab_lifetime_helpers import reset_slab_lifetime

_TAG = {torch.bfloat16: "BF16", torch.float32: "F32",
        torch.float8_e4m3fn: "F8_E4M3", torch.uint8: "U8", torch.int8: "I8"}


@pytest.fixture(autouse=True)
def _isolated_slab_lifetime(monkeypatch):
    reset_slab_lifetime(monkeypatch)


def _to_bytes(t: torch.Tensor) -> bytes:
    return t.detach().reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()


def write_safetensors(path, tensors, metadata=None):
    header, blobs, off = {}, [], 0
    for name, t in tensors.items():
        b = _to_bytes(t)
        header[name] = {"dtype": _TAG[t.dtype], "shape": list(t.shape),
                        "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    if metadata:
        header["__metadata__"] = metadata
    hj = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)


@pytest.fixture
def ckpt(tmp_path):
    torch.manual_seed(7)
    tensors = {
        "blocks.0.w": torch.randn(64, 64, dtype=torch.float32).to(torch.bfloat16),
        "blocks.0.b": torch.randn(64, dtype=torch.float32).to(torch.bfloat16),
        "blocks.1.w": torch.randn(48, 64, dtype=torch.float32),   # F32-stored
        "small": torch.arange(10, dtype=torch.float32).to(torch.bfloat16),
    }
    path = tmp_path / "model.safetensors"
    write_safetensors(str(path), tensors, metadata={"fmt": "test"})
    return str(path), tensors


def test_slab_bytes_and_alignment(ckpt):
    path, tensors = ckpt
    ws = WeightSlab(path)
    try:
        sd = ws.state_dict()
        assert set(sd) == set(tensors)
        for name, ref in tensors.items():
            got = sd[name]
            assert got.dtype == ref.dtype and tuple(got.shape) == tuple(ref.shape)
            assert _to_bytes(got.cpu()) == _to_bytes(ref)
            assert ws.regions[name].offset % _ALIGN == 0
        assert ws.metadata == {"fmt": "test"}
        assert ws.stat_ok()
    finally:
        del sd
        ws.close()


def test_slab_reconstructs_packed_f4_storage_shape(tmp_path):
    f4 = getattr(torch, "float4_e2m1fn_x2", None)
    if f4 is None:
        pytest.skip("torch build predates packed F4 dtype")
    path = tmp_path / "f4.safetensors"
    raw_tensor = torch.zeros((2, 4), dtype=f4)
    payload = raw_tensor.view(torch.uint8).numpy().tobytes()
    header = {
        "weight": {"dtype": "F4", "shape": [2, 8], "data_offsets": [0, len(payload)]},
    }
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)

    ws = WeightSlab(str(path))
    try:
        state = ws.state_dict()["weight"]
        assert state.dtype == f4
        assert tuple(state.shape) == (2, 4)
        assert state.view(torch.uint8).cpu().numpy().tobytes() == payload
    finally:
        del state
        ws.close()


def test_slab_reports_dtype_missing_from_torch_as_capability_error(ckpt, monkeypatch):
    path, _ = ckpt
    monkeypatch.delitem(unbake_module._ST_TO_TORCH, "F32")
    monkeypatch.setattr(
        slab_module, "_Arena",
        lambda *_args, **_kwargs: pytest.fail("unsupported dtype allocated a slab arena"),
    )

    with pytest.raises(UnsupportedSafetensorsDtypeError, match=r"F32.*torch build"):
        WeightSlab(path)

    # The dtype check runs in the header read, before any arena is allocated or registered.
    assert not any(key[0] == path for key in slab_module._OPEN_SLABS)


def test_unbake_capture_keeps_unsupported_dtype_strict(ckpt, monkeypatch):
    path, _ = ckpt
    monkeypatch.delitem(unbake_module._ST_TO_TORCH, "F32")

    with pytest.raises(UnsupportedSafetensorsDtypeError, match=r"F32.*torch build"):
        unbake_module.capture_unbake_record(object(), [], path)


@pytest.mark.parametrize("dtype", [None, "", 7, ["F32"]])
def test_malformed_dtype_field_is_not_a_capability_error(tmp_path, dtype):
    path = tmp_path / "malformed-dtype.safetensors"
    header = {
        "weight": {"dtype": dtype, "shape": [1], "data_offsets": [0, 1]},
    }
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0")

    with pytest.raises(SafetensorsHeaderError) as exc_info:
        read_safetensors_header(path)
    assert not isinstance(exc_info.value, UnsupportedSafetensorsDtypeError)


def _write_raw_safetensors(path, header, payload: bytes) -> None:
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def test_unknown_dtype_is_capability_error_only_after_valid_layout(tmp_path):
    path = tmp_path / "future-dtype.safetensors"
    _write_raw_safetensors(path, {
        "weight": {"dtype": "FUTURE4", "shape": [2], "data_offsets": [0, 1]},
    }, b"\0")

    with pytest.raises(UnsupportedSafetensorsDtypeError, match="FUTURE4"):
        read_safetensors_header(path)


@pytest.mark.parametrize(
    ("header", "payload", "match"),
    [
        (
            {"weight": {"dtype": "FUTURE", "shape": [1],
                        "data_offsets": [0, 2]}},
            b"\0",
            "exceeds file size",
        ),
        (
            {
                "a": {"dtype": "FUTURE", "shape": [1], "data_offsets": [0, 2]},
                "b": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]},
            },
            b"\0\0",
            "overlaps",
        ),
    ],
)
def test_unknown_dtype_does_not_mask_structural_corruption(
        tmp_path, header, payload, match):
    path = tmp_path / "malformed-future-dtype.safetensors"
    _write_raw_safetensors(path, header, payload)

    with pytest.raises(SafetensorsHeaderError, match=match) as exc_info:
        read_safetensors_header(path)
    assert not isinstance(exc_info.value, UnsupportedSafetensorsDtypeError)



def test_unknown_dtype_with_legal_padding_is_the_narrow_fallback_error(tmp_path):
    """Alignment padding is a legal layout, so an unknown dtype in a padded
    file surfaces as the narrow fallback-eligible error, not corruption."""
    path = tmp_path / "padded-future-dtype.safetensors"
    _write_raw_safetensors(path, {
        "a": {"dtype": "FUTURE", "shape": [1], "data_offsets": [0, 1]},
        "b": {"dtype": "U8", "shape": [1], "data_offsets": [2, 3]},
    }, b"\0\0\0")
    with pytest.raises(UnsupportedSafetensorsDtypeError):
        read_safetensors_header(path)


def test_constructor_short_read_is_failure_atomic(ckpt, monkeypatch):
    path, _ = ckpt
    created: list[int] = []
    real_memfd_create = os.memfd_create

    def tracking_memfd(name, flags):
        fd = real_memfd_create(name, flags)
        created.append(fd)
        return fd

    monkeypatch.setattr(slab_module.os, "memfd_create", tracking_memfd)
    monkeypatch.setattr(slab_module.os, "preadv", lambda *_args, **_kwargs: 0)

    with pytest.raises(RuntimeError, match="short read"):
        WeightSlab(path)

    assert len(created) == 1
    with pytest.raises(OSError):
        os.fstat(created[0])
    assert not any(key[0] == path for key in slab_module._OPEN_SLABS)


def test_arena_setup_failure_closes_memfd(ckpt, monkeypatch):
    path, _ = ckpt
    created: list[int] = []
    real_memfd_create = os.memfd_create

    def tracking_memfd(name, flags):
        fd = real_memfd_create(name, flags)
        created.append(fd)
        return fd

    monkeypatch.setattr(slab_module.os, "memfd_create", tracking_memfd)
    monkeypatch.setattr(
        slab_module.os, "ftruncate",
        lambda *_args: (_ for _ in ()).throw(OSError("ftruncate failed")),
    )

    with pytest.raises(OSError, match="ftruncate failed"):
        WeightSlab(path)

    assert len(created) == 1
    with pytest.raises(OSError):
        os.fstat(created[0])
    assert not any(key[0] == path for key in slab_module._OPEN_SLABS)


def test_close_attempts_annex_after_primary_arena_failure():
    events = []

    class Arena:
        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail

        def close(self):
            events.append(self.name)
            if self.fail:
                raise OSError("primary close failed")

    slab = WeightSlab.__new__(WeightSlab)
    slab._registered = False
    slab._arena_handoff = []
    slab._annex_handoff = []
    slab._handoff = None
    slab._close_guard = None
    slab.arena = Arena("primary", fail=True)
    slab.annex = Arena("annex")

    with pytest.raises(OSError, match="primary close failed"):
        slab.close()
    assert events == ["primary", "annex"]


@pytest.mark.parametrize("cancellation_first", [False, True])
def test_close_prefers_cancellation_across_all_arenas(cancellation_first):
    events = []
    ordinary = RuntimeError("ordinary arena close failed")
    cancellation = KeyboardInterrupt("arena close cancelled")

    class Arena:
        def __init__(self, name, error):
            self.name = name
            self.error = error

        def close(self):
            events.append(self.name)
            raise self.error

    primary_error, annex_error = (
        (cancellation, ordinary)
        if cancellation_first
        else (ordinary, cancellation)
    )
    slab = WeightSlab.__new__(WeightSlab)
    slab._registered = False
    slab._arena_handoff = []
    slab._annex_handoff = []
    slab._handoff = None
    slab._close_guard = None
    slab.arena = Arena("primary", primary_error)
    slab.annex = Arena("annex", annex_error)

    with pytest.raises(KeyboardInterrupt) as caught:
        slab.close()

    assert caught.value is cancellation
    assert caught.value.__cause__ is not cancellation
    assert events == ["primary", "annex"]


def test_close_interruption_cannot_unregister_same_checkpoint_peer(
    ckpt, monkeypatch,
):
    class CloseInterrupted(BaseException):
        def __bool__(self):
            raise AssertionError("close exception truthiness must not be evaluated")

    interruption = CloseInterrupted("interrupted after observable count update")

    class InterruptAfterCount(dict):
        armed = True

        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if self.armed and value == 1:
                self.armed = False
                raise interruption

    first = WeightSlab(ckpt[0])
    peer = WeightSlab(ckpt[0])
    key = first._open_key
    counts = InterruptAfterCount(slab_module._OPEN_SLABS)
    monkeypatch.setattr(slab_module, "_OPEN_SLABS", counts)
    try:
        assert counts[key] == 2
        with pytest.raises(CloseInterrupted) as raised:
            first.close()
        assert raised.value is interruption
        assert counts[key] == 1
        assert slab_module._OPEN_SLAB_OWNERS[key] == {peer._registration_token}

        first.close()
        first.close()
        assert counts[key] == 1
        assert peer._registered is True

        peer.close()
        assert key not in counts
        assert key not in slab_module._OPEN_SLAB_OWNERS
    finally:
        for slab in (first, peer):
            try:
                slab.close()
            except BaseException:
                pass


def test_final_slab_unregister_pop_is_repairable(ckpt, monkeypatch):
    class CloseInterrupted(BaseException):
        pass

    interruption = CloseInterrupted("interrupted after final count removal")

    class InterruptAfterPop(dict):
        armed = True

        def pop(self, key, default=None):
            value = super().pop(key, default)
            if self.armed:
                self.armed = False
                raise interruption
            return value

    slab = WeightSlab(ckpt[0])
    key = slab._open_key
    counts = InterruptAfterPop(slab_module._OPEN_SLABS)
    monkeypatch.setattr(slab_module, "_OPEN_SLABS", counts)
    try:
        with pytest.raises(CloseInterrupted) as raised:
            slab.close()
        assert raised.value is interruption
        slab.close()
        slab.close()
        assert key not in counts
        assert key not in slab_module._OPEN_SLAB_OWNERS
    finally:
        try:
            slab.close()
        except BaseException:
            pass


def test_slab_contains_and_region_by_ptr(ckpt):
    path, _ = ckpt
    ws = WeightSlab(path)
    try:
        sd = ws.state_dict()
        t = sd["blocks.0.w"]
        assert ws.contains(t.data_ptr())
        region = ws.region_by_ptr(t.data_ptr())
        assert region is ws.regions["blocks.0.w"]
        assert not ws.contains(0x1000)
        assert ws.region_by_ptr(0x1000) is None
    finally:
        del sd, t
        ws.close()


class _QuantLike(torch.Tensor):
    """Stand-in for comfy's QuantizedTensor: a Tensor subclass."""


def test_annex_cache_drain_settles_before_retire_and_trims_batch(ckpt, monkeypatch):
    """Real reabsorb drops sources only after their annex replacements exist.

    Three small strays exercise both a threshold drain after the second copy
    and a final-tail drain after the third. The cache spy sees real Parameters
    and weak references, so removing `del t` in `reabsorb` makes this fail.
    """
    path, _tensors = ckpt
    ws = WeightSlab(path)
    sd = ws.state_dict()
    module = nn.Module()
    expected: list[torch.Tensor] = []
    for index in range(3):
        source = sd["blocks.0.b"].clone().to(torch.float32)
        expected.append(source.detach().clone())
        module.register_parameter(f"p{index}", nn.Parameter(source, requires_grad=False))
        del source

    seen_sources: list[weakref.ReferenceType[torch.Tensor]] = []
    cache_calls: list[int] = []
    # Each FP32 source is 256 bytes, so the second copy crosses 300 and the third drains in the final flush.
    monkeypatch.setattr(slab_module, "_ANNEX_CACHE_DRAIN_BYTES", 300)

    def settle_copy(source):
        index = len(seen_sources)
        assert module._parameters[f"p{index}"].data_ptr() == source.data_ptr()
        seen_sources.append(weakref.ref(source))
        return True

    def empty_cache():
        processed = len(seen_sources)
        assert processed in (2, 3)
        assert ws.annex is not None
        for index in range(processed):
            assert ws.annex.contains(module._parameters[f"p{index}"].data_ptr())
            assert seen_sources[index]() is None
        cache_calls.append(processed)

    monkeypatch.setattr(slab_module._AnnexCacheDrain, "settle_copy", staticmethod(settle_copy))
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    try:
        stats = ws.reabsorb(module)
        assert stats["reabsorbed"] == 3
        assert cache_calls == [2, 3]
        for index, value in enumerate(expected):
            assert torch.equal(module._parameters[f"p{index}"].data.cpu(), value.cpu())
            assert seen_sources[index]() is None
    finally:
        del sd, expected
        module = None
        ws.close()


def test_annex_cache_drain_leaves_cpu_and_nonannex_paths_alone(monkeypatch):
    """A CPU replacement never triggers the CUDA-only batch drain."""
    events: list[str] = []
    drain = slab_module._AnnexCacheDrain()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    assert not drain.settle_copy(torch.empty(1))
    drain.flush()
    assert events == []


def test_annex_cache_drain_failure_is_not_hidden(monkeypatch):
    drain = slab_module._AnnexCacheDrain()
    drain._bytes = 1
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: (_ for _ in ()).throw(RuntimeError("trim")))
    with pytest.raises(RuntimeError, match="trim"):
        drain.flush()
    assert drain._bytes == 1


def test_reabsorb_annex_cache_failure_follows_adopted_parameter(ckpt, monkeypatch):
    """A drain failure propagates only after the completed annex copy is adopted."""
    path, _tensors = ckpt
    ws = WeightSlab(path)
    sd = ws.state_dict()
    module = nn.Module()
    cast = sd["blocks.0.b"].clone().to(torch.float32)
    old_ptr = cast.data_ptr()
    module.register_parameter("p1", nn.Parameter(cast, requires_grad=False))
    ws.key_map["p1"] = ws.regions["blocks.0.b"]
    monkeypatch.setattr(slab_module._AnnexCacheDrain, "settle_copy", staticmethod(lambda _source: True))
    monkeypatch.setattr(slab_module, "_ANNEX_CACHE_DRAIN_BYTES", 1)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: (_ for _ in ()).throw(RuntimeError("trim")))
    try:
        with pytest.raises(RuntimeError, match="trim"):
            ws.reabsorb(module)
        assert ws.annex is not None and ws.annex.contains(module.p1.data_ptr())
        assert module.p1.data_ptr() != old_ptr
        assert torch.equal(module.p1.data.cpu(), cast.cpu())
    finally:
        del sd, cast, module
        ws.close()


def test_reabsorb(ckpt):
    path, _tensors = ckpt
    ws = WeightSlab(path)
    sd = ws.state_dict()
    module = nn.Module()
    # adopt slab tensors as params (what assign=True produces)
    module.register_parameter("p0", nn.Parameter(sd["blocks.0.w"], requires_grad=False))
    module.register_parameter("p1", nn.Parameter(sd["blocks.0.b"], requires_grad=False))
    ws.key_map["p0"] = ws.regions["blocks.0.w"]
    ws.key_map["p1"] = ws.regions["blocks.0.b"]

    # stray 1: a baked replacement, same dtype and size, goes back to its region
    baked = sd["blocks.0.w"].clone() * 2.0
    module.p0 = nn.Parameter(baked, requires_grad=False)
    # stray 2: a cast key, dtype changed, goes to the annex
    cast = sd["blocks.0.b"].clone().to(torch.float32)
    module.p1 = nn.Parameter(cast, requires_grad=False)
    # stray 3: a quant-like subclass is left alone and counted
    q = torch.zeros(4, dtype=torch.bfloat16).as_subclass(_QuantLike)
    module.register_parameter("q", nn.Parameter(q, requires_grad=False))

    stats = ws.reabsorb(module)
    assert stats["reabsorbed"] == 2
    assert ws.contains(module.p0.data_ptr())
    assert ws.region_by_ptr(module.p0.data_ptr()) is ws.regions["blocks.0.w"]
    assert torch.equal(module.p0.data.cpu(), baked.cpu())
    assert ws.annex is not None and ws.annex.contains(module.p1.data_ptr())
    assert torch.equal(module.p1.data.cpu(), cast.cpu())
    assert not ws.contains(module.q.data_ptr())
    assert ws.stray_bytes == q.numel() * q.element_size()

    del sd, baked, cast, module, q
    ws.close()
    assert ws.arena.closed and ws.annex.closed


class _QuantData(torch.Tensor):
    """Stand-in for comfy_kitchen's QuantizedTensor: a Tensor subclass
    carrying the wrapper attrs the clone path reads (_qdata/_layout_cls/
    _params) and reconstructable via (qdata, layout_cls, params)."""

    @staticmethod
    def __new__(cls, qdata, layout_cls="TensorCoreInt8Layout", params=None):
        return qdata.as_subclass(cls)

    def __init__(self, qdata, layout_cls="TensorCoreInt8Layout", params=None):
        self._qdata = qdata
        self._layout_cls = layout_cls
        self._params = params


class _FakeQuantOp(nn.Module):
    def __init__(self, wrapper):
        super().__init__()
        self.weight = nn.Parameter(wrapper, requires_grad=False)


def test_reabsorb_quant_stray_clones_wrapper_into_slab(tmp_path, monkeypatch):
    """A strayed QuantizedTensor is cloned with the same layout name and params
    object around a slab-relocated payload. Never rebuild it from file
    metadata: on 2026-07-08 a metadata rebuild diverged on int8-convrot
    (identity gate FAIL, max latent diff 2.53; docs/VALIDATION.md)."""
    torch.manual_seed(3)
    pristine = torch.randint(-100, 100, (8, 8), dtype=torch.int8)
    path = str(tmp_path / "quant.safetensors")
    write_safetensors(path, {"blk.weight": pristine.clone()})

    ws = WeightSlab(path)
    root = nn.Module()
    baked = torch.randint(-100, 100, (8, 8), dtype=torch.int8)
    params = types.SimpleNamespace(scale=torch.tensor(0.5),
                                   orig_shape=(8, 8),
                                   orig_dtype=torch.bfloat16)
    wrapper = _QuantData(baked.clone(), "TensorCoreInt8Layout", params)
    root.blk = _FakeQuantOp(wrapper)
    # A real comfy_kitchen wrapper keeps its attributes through a Parameter
    # round trip; this plain fake loses them, so pin live_tensor to the wrapper.
    monkeypatch.setattr("dgx_monarch.actor.slab.live_tensor",
                        lambda m, k: wrapper if k == "blk.weight" else m.blk)

    stats = ws.reabsorb(root)
    assert ws.stray_bytes == 0
    w = root.blk.weight
    assert isinstance(w, torch.nn.Parameter)
    # Check what survives on the fake: the class and the storage pointer.
    assert isinstance(w.data, _QuantData)
    assert ws.contains(w.data.data_ptr())
    region = ws.regions["blk.weight"]
    got = bytes(ws.arena.mm[region.offset:region.offset + region.nbytes])
    assert got == _to_bytes(baked)                    # the baked bytes, not the file's
    assert stats["quant_reabsorbed_gib"] >= 0

    del w, root
    ws.close()


def test_reabsorb_quant_without_wrapper_attrs_stays_stray(tmp_path):
    """A subclass stray without the wrapper contract is left alone, counted."""
    torch.manual_seed(4)
    qdata = torch.randint(-5, 5, (4, 4), dtype=torch.int8)
    path = str(tmp_path / "bare.safetensors")
    write_safetensors(path, {"blk.weight": qdata.clone()})
    ws = WeightSlab(path)
    root = nn.Module()

    class _Bare(torch.Tensor):
        pass

    op = nn.Module()
    op.register_parameter("weight", nn.Parameter(
        qdata.clone().as_subclass(_Bare), requires_grad=False))
    root.blk = op
    stats = ws.reabsorb(root)
    assert stats["quant_reabsorbed_gib"] == 0
    assert ws.stray_bytes == qdata.numel() * qdata.element_size()
    del root, op
    ws.close()


def test_reabsorb_annex_short_still_takes_region_strays(ckpt):
    """A re-run against a full annex must still reabsorb strays whose
    original region can take them; only annex-needing strays are skipped."""
    path, _ = ckpt
    ws = WeightSlab(path)
    sd = ws.state_dict()
    module = nn.Module()
    module.register_parameter("p0", nn.Parameter(sd["blocks.0.w"], requires_grad=False))
    ws.key_map["p0"] = ws.regions["blocks.0.w"]
    # round 1: one cast stray sizes the annex at its own 256 bytes plus one 256-byte alignment pad
    cast0 = sd["blocks.0.b"].clone().to(torch.float32)
    module.register_parameter("c0", nn.Parameter(cast0, requires_grad=False))
    stats = ws.reabsorb(module)
    assert stats["reabsorbed"] == 1 and stats["skipped"] == 0
    assert ws.annex is not None

    # round 2: a baked stray with a region, and a new cast stray the annex cannot take
    baked = sd["blocks.0.w"].clone() * 3.0
    module.p0 = nn.Parameter(baked, requires_grad=False)
    cast1 = sd["blocks.1.w"].clone().to(torch.bfloat16)  # too big for the annex slack
    module.register_parameter("c1", nn.Parameter(cast1, requires_grad=False))
    stats = ws.reabsorb(module)
    assert stats["reabsorbed"] == 1          # p0 returns to its region
    assert stats["skipped"] == 1             # c1 keeps its own allocation
    assert ws.region_by_ptr(module.p0.data_ptr()) is ws.regions["blocks.0.w"]
    assert torch.equal(module.p0.data.cpu(), baked.cpu())
    assert not ws.contains(module.c1.data_ptr())

    del sd, baked, cast0, cast1, module
    ws.close()


def test_close_retains_live_export_until_retry(ckpt):
    path, _ = ckpt
    ws = WeightSlab(path)
    view = memoryview(ws.arena.mm)          # simulates a leaked reference
    with pytest.raises(RuntimeError, match="mapping is still referenced"):
        ws.close()
    assert not ws.arena.closed
    assert not ws.arena.close_uncertain
    assert ws.arena.fd >= 0
    view.release()
    ws.close()
    assert ws.arena.closed


def test_arena_fd_close_uncertainty_is_terminal_and_never_retried(monkeypatch):
    from contextlib import suppress

    arena = slab_module._Arena("dgxm-test-arena", 4096)
    arena.confirm_handoff()
    owned_fd = arena.fd
    healthy_close = slab_module.os.close
    close_calls = []

    def close_then_raise(fd):
        close_calls.append(fd)
        healthy_close(fd)
        raise OSError("descriptor close reported failure after release")

    monkeypatch.setattr(slab_module.os, "close", close_then_raise)
    with pytest.raises(OSError, match="reported failure after release"):
        arena.close()

    assert not arena.closed
    assert arena.close_uncertain
    assert arena.fd == -1
    assert close_calls == [owned_fd]

    unrelated_fd = os.open("/dev/null", os.O_RDONLY)
    try:
        if unrelated_fd != owned_fd:
            os.dup2(unrelated_fd, owned_fd)
            healthy_close(unrelated_fd)
            unrelated_fd = owned_fd
        with pytest.raises(RuntimeError, match="reset the Attached mesh"):
            arena.close()
        os.fstat(unrelated_fd)
        assert close_calls == [owned_fd]
    finally:
        with suppress(OSError):
            healthy_close(unrelated_fd)


def test_partial_weight_slab_preserves_primary_and_retains_uncertain_arena(
    ckpt,
    monkeypatch,
):
    from dgx_monarch.actor import slab_lifetime

    class ReadAbort(BaseException):
        pass

    path, _ = ckpt
    parsed_header = slab_module.read_safetensors_header(
        path, want_metadata=True, want_identity=True
    )
    primary = ReadAbort("slab read interrupted")
    healthy_close = slab_module.os.close

    reset_slab_lifetime(monkeypatch)
    monkeypatch.setattr(
        slab_module,
        "read_safetensors_header",
        lambda *_args, **_kwargs: parsed_header,
    )
    monkeypatch.setattr(
        WeightSlab,
        "_read_all",
        lambda _self: (_ for _ in ()).throw(primary),
    )

    def close_then_raise(fd):
        healthy_close(fd)
        raise OSError("arena descriptor close reported failure after release")

    monkeypatch.setattr(slab_module.os, "close", close_then_raise)
    with pytest.raises(ReadAbort) as caught:
        WeightSlab(path)

    assert caught.value is primary
    assert slab_lifetime.cleanup_poisoned()
    assert slab_lifetime.retained_count() == 1
    retained, retained_error = slab_lifetime._RETAINED_FAILED_LOAD_SLABS[0]
    assert retained_error is not None
    assert retained.arena.close_uncertain
    assert not retained.arena.closed
    assert retained.arena.fd == -1


def test_stat_ok_detects_change(ckpt, tmp_path):
    path, _ = ckpt
    ws = WeightSlab(path)
    try:
        assert ws.stat_ok()
        with open(path, "ab") as f:
            f.write(b"x")
        assert not ws.stat_ok()
    finally:
        ws.close()


class _FakeBaseModel:
    def __init__(self, module):
        self.diffusion_model = module
        self.calls = []

    def load_model_weights(self, sd, unet_prefix="", assign=False):
        self.calls.append({"assign": assign, "sd": dict(sd)})
        return self


def _stub_comfy(monkeypatch, ltf):
    utils = types.ModuleType("comfy.utils")
    utils.load_torch_file = ltf
    model_base = types.ModuleType("comfy.model_base")
    model_base.BaseModel = _FakeBaseModel
    comfy = types.ModuleType("comfy")
    comfy.utils, comfy.model_base = utils, model_base
    for name, mod in (("comfy", comfy), ("comfy.utils", utils),
                      ("comfy.model_base", model_base)):
        monkeypatch.setitem(sys.modules, name, mod)
    return utils, model_base


def test_slab_load_context(ckpt, monkeypatch):
    path, tensors = ckpt
    sentinel_ltf = lambda *a, **k: "stock"  # noqa: E731
    utils, model_base = _stub_comfy(monkeypatch, sentinel_ltf)
    from dgx_monarch.actor.comfy_bridge import slab_load

    # nested so named_parameters() yields the dotted checkpoint key
    module = nn.Module()
    module.blocks = nn.ModuleList([nn.Module(), nn.Module()])
    # the module wants bf16 where the file stores F32, so the hook must pre-cast
    module.blocks[1].register_parameter(
        "w", nn.Parameter(torch.zeros(48, 64, dtype=torch.bfloat16),
                          requires_grad=False))
    fake = _FakeBaseModel(module)

    with slab_load(path) as ws:
        # other files keep the stock loader
        assert utils.load_torch_file("other.safetensors") == "stock"
        # this file returns the slab state_dict, with its metadata on request
        sd, md = utils.load_torch_file(path, return_metadata=True)
        assert md == {"fmt": "test"}
        assert ws.contains(sd["blocks.0.w"].data_ptr())
        # load_model_weights is forced to assign=True and pre-casts F32 to bf16
        model_base.BaseModel.load_model_weights(fake, dict(sd))
        assert fake.calls[-1]["assign"] is True
        loaded = fake.calls[-1]["sd"]["blocks.1.w"]
        assert loaded.dtype == torch.bfloat16
        assert torch.equal(loaded.cpu(),
                           tensors["blocks.1.w"].to(torch.bfloat16))
        assert "blocks.1.w" in ws.cast_keys
        # stripped-key map recorded for reabsorb
        assert ws.key_map["blocks.0.w"] is ws.regions["blocks.0.w"]

    # patches restored on exit
    assert utils.load_torch_file is sentinel_ltf
    assert model_base.BaseModel.load_model_weights is _FakeBaseModel.load_model_weights
    del sd
    ws.close()


def test_slab_load_hands_the_checkpoint_config_to_the_model_builder(tmp_path, monkeypatch):
    # Comfy builds the DiT from __metadata__["config"], and its constructor
    # defaults differ from the shipped file (LTX connector depth 2 against the
    # checkpoint's 8). A reader that returned tensors alone would build a
    # different model and load most of the weights into it in silence.
    config = {"transformer": {"connector_num_layers": 8,
                              "av_ca_timestep_scale_multiplier": 1000.0}}
    path = str(tmp_path / "dit.safetensors")
    write_safetensors(path, {"blocks.0.w": torch.zeros(4, 4, dtype=torch.bfloat16)},
                      metadata={"config": json.dumps(config), "model_version": "2.5.0"})
    utils, _model_base = _stub_comfy(monkeypatch, lambda *a, **k: "stock")
    from dgx_monarch.actor.comfy_bridge import slab_load

    with slab_load(path) as ws:
        sd, md = utils.load_torch_file(path, return_metadata=True)
        assert json.loads(md["config"]) == config
        assert md["model_version"] == "2.5.0"

    del sd
    ws.close()


def test_slab_load_failure_closes_and_restores(ckpt, monkeypatch):
    path, _ = ckpt
    sentinel_ltf = lambda *a, **k: "stock"  # noqa: E731
    utils, _model_base = _stub_comfy(monkeypatch, sentinel_ltf)
    from dgx_monarch.actor.comfy_bridge import slab_load

    with pytest.raises(RuntimeError, match="boom"):
        with slab_load(path) as ws:
            raise RuntimeError("boom")
    assert ws.arena.closed
    assert utils.load_torch_file is sentinel_ltf


def test_slab_load_preexposure_close_failure_prepublishes_and_poisons(
    ckpt,
    monkeypatch,
):
    path, _ = ckpt
    _utils, _model_base = _stub_comfy(monkeypatch, lambda *a, **k: "stock")
    from dgx_monarch.actor import slab_lifetime
    from dgx_monarch.actor.comfy_bridge import slab_load

    class CloseAbort(BaseException):
        pass

    reset_slab_lifetime(monkeypatch)
    primary = RuntimeError("pre-exposure load failed")
    healthy_close = WeightSlab.close
    close_observations = []

    def interrupt_close(slab):
        close_observations.append((
            slab_lifetime.cleanup_poisoned(),
            tuple(owned for owned, _error in (
                slab_lifetime._RETAINED_FAILED_LOAD_SLABS
            )),
        ))
        raise CloseAbort("slab close interrupted")

    monkeypatch.setattr(WeightSlab, "close", interrupt_close)
    with pytest.raises(CloseAbort) as caught:
        with slab_load(path) as ws:
            raise primary

    assert caught.value.__cause__ is primary
    assert close_observations == [(True, (ws,))]
    assert slab_lifetime.cleanup_poisoned()
    assert slab_lifetime.retained_count() == 1

    monkeypatch.setattr(WeightSlab, "close", healthy_close)
    healthy_close(ws)
    slab_lifetime._RETAINED_FAILED_LOAD_SLABS.clear()
    monkeypatch.setattr(slab_lifetime, "_FAILED_LOAD_CLEANUP_POISONED", False)


def test_slab_load_exposed_baseexception_has_durable_owner(ckpt, monkeypatch):
    """Raw-pointer tensors forbid context-exit close until an explicit unload."""
    import gc
    import weakref

    path, _ = ckpt
    utils, _model_base = _stub_comfy(monkeypatch, lambda *a, **k: "stock")
    from dgx_monarch.actor import slab_lifetime
    from dgx_monarch.actor.comfy_bridge import slab_load

    class Abort(BaseException):
        pass

    monkeypatch.setattr(slab_lifetime, "_RETAINED_FAILED_LOAD_SLABS", [])
    mm = types.ModuleType("comfy.model_management")
    mm.unload_all_models = lambda: None
    mm.soft_empty_cache = lambda: None
    sys.modules["comfy"].model_management = mm
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

    with pytest.raises(Abort, match="load interrupted") as caught:
        with slab_load(path) as ws:
            slab_ref = weakref.ref(ws)
            arena = ws.arena
            sd = utils.load_torch_file(path)
            raise Abort("load interrupted")
    del ws, sd
    caught.value.__traceback__ = None
    del caught
    gc.collect()
    assert slab_lifetime.retained_count() == 1
    assert slab_ref() is not None
    assert not arena.closed

    slab_lifetime.release_after_explicit_unload()
    gc.collect()
    assert arena.closed
    assert slab_lifetime.retained_count() == 0
    assert slab_ref() is None


def test_slab_keeps_quant_json_on_cpu_when_cuda_is_available(tmp_path, monkeypatch):
    """Comfy decodes comfy_quant markers through Tensor.numpy()."""
    marker = torch.tensor(
        list(json.dumps({"format": "float8_e4m3fn"}).encode("utf-8")),
        dtype=torch.uint8,
    )
    path = str(tmp_path / "quant-metadata.safetensors")
    write_safetensors(path, {
        "weight": torch.arange(4, dtype=torch.float32),
        "block.weight_scale": torch.ones(1, dtype=torch.float32),
        "comfy_quant": marker,
        "block.comfy_quant": marker,
    })
    ws = WeightSlab(path)
    real_tensor = ws._tensor
    names_by_region = {id(region): name for name, region in ws.regions.items()}
    cuda_requests = {}
    sd = None

    def observe_tensor(region, *, cuda):
        cuda_requests[names_by_region[id(region)]] = cuda
        # Each CUDA request gets a meta tensor, so a marker routed to CUDA
        # cannot stay NumPy-readable on a CPU-only runner.
        if cuda:
            return torch.empty(0, device="meta")
        return real_tensor(region, cuda=False)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ws, "_tensor", observe_tensor)
    try:
        sd = ws.state_dict()
        assert cuda_requests == {
            "weight": True,
            "block.weight_scale": True,
            "comfy_quant": False,
            "block.comfy_quant": False,
        }
        assert sd["weight"].device.type == "meta"
        assert sd["block.weight_scale"].device.type == "meta"
        for key in ("comfy_quant", "block.comfy_quant"):
            assert sd[key].device.type == "cpu"
            assert json.loads(sd[key].numpy().tobytes()) == {
                "format": "float8_e4m3fn",
            }
    finally:
        del sd
        ws.close()


def test_the_arena_is_the_file_within_one_page(ckpt):
    """The slab price of one times the file holds only while the arena is the file within a page.

    ``slab_load_fit`` prices ``os.path.getsize`` rather than the header's
    aligned total, so both prices read one measurement. This fails once ``_ALIGN``
    pads the regions more than 4096 bytes past the file; it reads the regions, not the arena.
    """
    path, _tensors = ckpt
    ws = WeightSlab(path)
    try:
        aligned = sum((region.nbytes + _ALIGN - 1) // _ALIGN * _ALIGN + _ALIGN
                      for region in ws.regions.values())
        file_size = os.path.getsize(path)
        assert abs(aligned - file_size) <= 4096
    finally:
        ws.close()


def test_the_slab_price_skips_a_cast_and_names_the_annex(ckpt, monkeypatch):
    """A cast does not shrink the arena; it adds a second one this probe cannot size."""
    from dgx_monarch import capacity_fit

    # The price applies on unified memory only; fake that on a CI box.
    monkeypatch.setattr(capacity_fit.mesh_safety, "gpu_is_integrated", lambda: True)

    path, _tensors = ckpt
    fit = capacity_fit.slab_load_fit(path, {"dtype": torch.bfloat16})
    assert (fit.applies, fit.fits) == (False, True)
    assert "annex" in fit.skipped_reason
    assert capacity_fit.slab_load_fit(path, {}).applies


# The 2026-09-03 flux2 bf16 hold legs at ComfyUI's partial-load threshold (docs/VALIDATION.md).
_M1_FILE_BYTES = 64_446_596_128            # flux2-dev bf16, 60.02 GiB
_M1_HOLD66_AVAIL = int(65.4 * (1 << 30))   # what the worker read on the leg
_M1_HOLD64_AVAIL = 68_170_149_888          # 63.5 GiB, mem_available_bytes in the leg's refusal


def _price_against(monkeypatch, size, avail, **kwargs):
    from dgx_monarch import capacity_fit

    monkeypatch.setattr(capacity_fit.mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(capacity_fit.os.path, "getsize", lambda _p: size)
    monkeypatch.setattr(capacity_fit.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(capacity_fit, "read_safetensors_header", lambda _p: None)
    monkeypatch.setattr(capacity_fit.mesh_safety, "mem_available_bytes", lambda: avail)
    return capacity_fit.slab_load_fit("held.safetensors", {}, **kwargs)


def test_the_absolute_floor_still_admits_the_leg_that_rendered(monkeypatch):
    """The floor may not refuse a leg that provably ran.

    The 2026-09-03 hold-66 leg priced 64.0 GiB against 65.4 GiB available and
    rendered. That log figure carries one decimal, so the reading behind it is
    at least 65.35 and any floor over about 5.33 GiB refuses the leg. 5 GiB is
    the largest round figure under that; a higher floor fails this test.
    """
    from dgx_monarch import capacity_fit

    fit = _price_against(monkeypatch, _M1_FILE_BYTES, _M1_HOLD66_AVAIL)
    assert fit.floor_bytes == capacity_fit.ABSOLUTE_HOST_FLOOR_BYTES
    assert fit.fits, "the hold-66 leg rendered; the price may not refuse it"
    assert fit.required_gib == 65.0


def test_the_absolute_floor_refuses_where_the_shipped_wall_refused(monkeypatch):
    """The other bracket: the 2026-09-03 hold-64 leg refused at 63.5 GiB and still does."""
    fit = _price_against(monkeypatch, _M1_FILE_BYTES, _M1_HOLD64_AVAIL)
    assert not fit.fits


def test_the_absolute_floor_is_the_outer_bound_on_the_slab_price(monkeypatch):
    """Neither the arena cap nor a small reserve prices a load under the floor.

    A 2 GiB file caps the reserve term at 1.7 GiB, but what ComfyUI needs left
    over does not shrink with the checkpoint, so the absolute floor holds.
    """
    from dgx_monarch import capacity_fit

    small = _price_against(monkeypatch, 2 * (1 << 30), 100 * (1 << 30))
    assert small.floor_bytes == capacity_fit.ABSOLUTE_HOST_FLOOR_BYTES
    bigger = _price_against(monkeypatch, 60 * (1 << 30), 100 * (1 << 30),
                            reserve_bytes=9 * (1 << 30))
    assert bigger.floor_bytes == 9 * (1 << 30)


def test_the_stock_price_carries_the_same_absolute_floor(monkeypatch):
    """The stock price charges the same absolute floor as the slab price.

    It stays the larger of the two for one file, so slab residency can still
    admit a load the stock price refuses.
    """
    from dgx_monarch import capacity_fit

    monkeypatch.setattr(capacity_fit.mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(capacity_fit.os.path, "getsize", lambda _p: _M1_FILE_BYTES)
    monkeypatch.setattr(capacity_fit.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(capacity_fit.mesh_safety, "mem_available_bytes",
                        lambda: int(114.5 * (1 << 30)))
    stock = capacity_fit.stock_load_fit("held.safetensors", {})
    expected_required_bytes = (
        int(_M1_FILE_BYTES * capacity_fit.STOCK_LOAD_TRANSIENT_FACTOR)
        + capacity_fit.ABSOLUTE_HOST_FLOOR_BYTES)
    assert stock.required_bytes == expected_required_bytes
    assert stock.required_gib == capacity_fit.gib(expected_required_bytes)
    # 114.5 GiB is about the most a clean box of this size reports available,
    # and the stock price of this 60 GiB bf16 file is above it. No priced stock
    # admission of this file is on record that stayed whole and survived: in the
    # 2026-09-02 pair incident (capacity_floor.py) one box was admitted at about
    # 113 GiB against the 111.0 GiB the price charged then and crashed, and the
    # other reads rung=stock_fits at 108.75 GiB, which no price could have produced.
    assert not stock.fits


def test_the_slab_card_names_the_partial_load_regime(monkeypatch):
    """The card names the floor as ComfyUI's threshold for keeping a model
    whole: a bare figure reads as a round number, and the fault the floor
    prevents is a mid-render offload, not an OOM."""
    from dgx_monarch import capacity_fit
    from dgx_monarch.actor import store_slab_admit

    fit = _price_against(monkeypatch, _M1_FILE_BYTES, _M1_HOLD64_AVAIL)
    text = store_slab_admit.card(fit, "flux2-dev.safetensors", "this host")
    assert "a 60.0 GiB file plus a 5.0 GiB floor for the rest of the box" in text
    assert "ComfyUI stops keeping a model whole and offloads part of the weights" in text
    # The card's stock figure is the stock wall's own price, floor included;
    # a lower figure would understate the wall a stock retry meets.
    assert f"it needs {capacity_fit.gib(capacity_fit.stock_required_bytes(_M1_FILE_BYTES))} GiB" in text


def test_a_reserve_smaller_than_the_absolute_floor_is_named_as_outranked(monkeypatch):
    """A reserve under the absolute floor is named beside the floor that outranks it.

    With 4 GiB set and 5 GiB charged, a card that printed only 5.0 GiB would
    read as a host that never read its setting. The card names both figures
    and words this cause apart from the arena-cap clip the next test covers.
    """
    from dgx_monarch import capacity_fit
    from dgx_monarch.actor import store_slab_admit

    gib = 1 << 30
    fit = _price_against(monkeypatch, 60 * gib, 40 * gib, reserve_bytes=4 * gib)
    assert fit.floor_bytes == capacity_fit.ABSOLUTE_HOST_FLOOR_BYTES
    text = store_slab_admit.card(fit, "big.safetensors", "this host",
                                 reserve_bytes=4 * gib)
    assert "plus a 5.0 GiB floor for the rest of the box, above the 4.0 GiB" in text
    assert "this host reserves (uma_reserve_gb), which it outranks" in text
    # The other cause keeps its own words, so the card never blames the cap for
    # a reserve no cap touched.
    assert "arena cap" not in text


def test_a_reserve_the_absolute_floor_outranks_is_still_named(monkeypatch):
    """A reserve the arena cap clips under the absolute floor is still named.

    A 4 GiB file caps the reserve term at 3.4 GiB, so a 10 GiB `uma_reserve_gb`
    is clipped below the 5 GiB floor and the floor is charged. A card that
    printed only 5.0 GiB would read as a host that never read the setting.
    """
    from dgx_monarch import capacity_fit
    from dgx_monarch.actor import store_slab_admit

    gib = 1 << 30
    fit = _price_against(monkeypatch, 4 * gib, 6 * gib, reserve_bytes=10 * gib)
    assert fit.floor_bytes == capacity_fit.ABSOLUTE_HOST_FLOOR_BYTES
    text = store_slab_admit.card(fit, "small.safetensors", "this host",
                                 reserve_bytes=10 * gib)
    assert "plus a 5.0 GiB floor for the rest of the box, above the 10.0 GiB" in text
    assert "which the arena cap clipped" in text
    # A host with no reserve set says nothing about one.
    plain = _price_against(monkeypatch, 4 * gib, 6 * gib)
    assert "uma_reserve_gb" not in store_slab_admit.card(
        plain, "small.safetensors", "this host")


def _write_minimal_safetensors(path, tensors: dict) -> None:
    """A header-valid file whose tensors never need their bytes read."""
    header, cursor = {}, 0
    for key, (dtype, shape) in tensors.items():
        numel = 1
        for dim in shape:
            numel *= dim
        from dgx_monarch.safetensors_header import DTYPE_BITS

        nbytes = numel * DTYPE_BITS[dtype] // 8
        header[key] = {"dtype": dtype, "shape": list(shape),
                       "data_offsets": [cursor, cursor + nbytes]}
        cursor += nbytes
    body = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(body)))
        f.write(body)
    os.truncate(path, 8 + len(body) + cursor)


def test_a_slab_quote_with_a_stack_under_low_rss_adds_the_lora_term(tmp_path, monkeypatch):
    from dgx_monarch import capacity_fit

    ckpt = tmp_path / "m.safetensors"
    lora = tmp_path / "l.safetensors"
    _write_minimal_safetensors(ckpt, {"blocks.0.attn.wq.weight": ("BF16", [64, 64])})
    _write_minimal_safetensors(lora, {
        "diffusion_model.blocks.0.attn.wq.lora_down.weight": ("BF16", [4, 4]),
        "diffusion_model.blocks.0.attn.wq.lora_up.weight": ("BF16", [4, 4]),
    })
    monkeypatch.setattr(capacity_fit.mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(capacity_fit.mesh_safety, "mem_available_bytes", lambda: 100 << 30)
    stack = [{"name": "l.safetensors", "strength": 1.0}]

    bare = capacity_fit.slab_load_fit(str(ckpt), {})
    with_lora = capacity_fit.slab_load_fit(
        str(ckpt), {}, lora_stack=stack, resolve_lora_path=lambda _n: str(lora))

    assert bare.lora_bytes == 0
    assert with_lora.lora_bytes == 64 * 64 * 2
    assert with_lora.required_bytes == bare.required_bytes + with_lora.lora_bytes


def test_an_empty_stack_leaves_the_slab_price_unchanged(tmp_path, monkeypatch):
    from dgx_monarch import capacity_fit

    ckpt = tmp_path / "m.safetensors"
    _write_minimal_safetensors(ckpt, {"blocks.0.attn.wq.weight": ("BF16", [64, 64])})
    monkeypatch.setattr(capacity_fit.mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(capacity_fit.mesh_safety, "mem_available_bytes", lambda: 100 << 30)

    bare = capacity_fit.slab_load_fit(str(ckpt), {})
    still_bare = capacity_fit.slab_load_fit(
        str(ckpt), {}, lora_stack=None, resolve_lora_path=lambda _n: "unused")
    assert still_bare == bare


def test_both_slab_walls_price_the_same_lora_term(tmp_path, monkeypatch):
    """``capacity_quote.price``'s rung and ``preload_capacity_check``'s
    re-price must agree, or one wall admits what the other would refuse."""
    from dgx_monarch import capacity_fit
    from dgx_monarch.actor import capacity_quote as cq
    from dgx_monarch.actor import store_residency

    ckpt = tmp_path / "m.safetensors"
    lora = tmp_path / "l.safetensors"
    _write_minimal_safetensors(ckpt, {"blocks.0.attn.wq.weight": ("BF16", [64, 64])})
    _write_minimal_safetensors(lora, {
        "diffusion_model.blocks.0.attn.wq.lora_down.weight": ("BF16", [4, 4]),
        "diffusion_model.blocks.0.attn.wq.lora_up.weight": ("BF16", [4, 4]),
    })
    monkeypatch.setattr(cq.mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(capacity_fit.mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(capacity_fit.mesh_safety, "mem_available_bytes", lambda: 100 << 30)
    stack = [{"name": "l.safetensors", "strength": 1.0}]

    row = cq.price(
        path=str(ckpt), unet_name="m.safetensors", model_options={},
        slab_weights=True, slab_capable_path=True, lora_low_rss=True,
        fsdp_launch=False, blocked_reason="", authoritative_slab_retry=False,
        memoized_family=lambda _p: None, vouched_families=frozenset(),
        lora_stack=stack, resolve_lora_path=lambda _n: str(lora),
    )
    assert row.slab_measured["lora_bytes"] == 64 * 64 * 2

    class _Decision:
        use_slab = True

    admitted = []
    monkeypatch.setattr(
        "dgx_monarch.actor.store_slab_admit.admit",
        lambda fit, *_a, **_kw: admitted.append(fit))
    store_residency.preload_capacity_check(
        _Decision(), str(ckpt), "m.safetensors", {}, preflight=lambda *_a, **_kw: None,
        lora_stack=stack, lora_low_rss=True,
        resolve_lora_path=lambda _n: str(lora))
    assert admitted[0].lora_bytes == row.slab_measured["lora_bytes"]


def test_the_bake_backstop_refuses_typed_before_the_bake_runs(monkeypatch):
    """``admit_bake`` compares the exact patched set, not the header estimate,
    and raises before ``_merge_and_free`` would allocate anything."""
    from dgx_monarch.actor import store_slab_admit
    from dgx_monarch.mesh_safety import StockLoadCapacityError

    class _FakeModel:
        def __getattr__(self, name):
            return torch.zeros(1 << 20, dtype=torch.bfloat16)  # 2 MiB/key

    active = types.SimpleNamespace(model=_FakeModel(), patches={f"w{i}": [] for i in range(40)})
    from dgx_monarch import mesh_safety as _ms
    monkeypatch.setattr(_ms, "mem_available_bytes", lambda: 5 << 30)

    with pytest.raises(StockLoadCapacityError, match="LoRA bake"):
        store_slab_admit.admit_bake(active, "m.safetensors", "this host")


def test_the_bake_backstop_admits_a_patch_set_that_fits(monkeypatch):
    from dgx_monarch.actor import store_slab_admit

    class _FakeModel:
        def __getattr__(self, name):
            return torch.zeros(4, dtype=torch.bfloat16)

    active = types.SimpleNamespace(model=_FakeModel(), patches={"w": []})
    from dgx_monarch import mesh_safety as _ms
    monkeypatch.setattr(_ms, "mem_available_bytes", lambda: 100 << 30)

    store_slab_admit.admit_bake(active, "m.safetensors", "this host")  # no raise


def test_the_bake_backstop_charges_the_floor_the_slab_price_charged(monkeypatch):
    """A configured reserve above the absolute floor holds at the backstop too."""
    from dgx_monarch import capacity_fit
    from dgx_monarch import mesh_safety as _ms
    from dgx_monarch.actor import store_slab_admit
    from dgx_monarch.mesh_safety import StockLoadCapacityError

    class _FakeModel:
        def __getattr__(self, name):
            return torch.zeros(1 << 20, dtype=torch.bfloat16)  # 2 MiB/key

    active = types.SimpleNamespace(model=_FakeModel(), patches={"w": []})
    # 7 GiB free: enough over the 5 GiB absolute floor, short of a 12 GiB reserve.
    monkeypatch.setattr(_ms, "mem_available_bytes", lambda: 7 << 30)
    store_slab_admit.admit_bake(active, "m.safetensors", "host")  # the absolute floor fits
    floor = capacity_fit.slab_floor_bytes(40 << 30, 12 << 30)
    assert floor == 12 << 30
    with pytest.raises(StockLoadCapacityError, match="the bake did not run"):
        store_slab_admit.admit_bake(active, "m.safetensors", "host", floor)


def test_the_bake_backstop_makes_no_claim_on_a_stub_active(monkeypatch):
    """A stub ``active`` with no real patch set carries no claim either way."""
    from dgx_monarch import mesh_safety as _ms
    from dgx_monarch.actor import store_slab_admit

    monkeypatch.setattr(_ms, "mem_available_bytes", lambda: 1)  # would refuse if it priced
    store_slab_admit.admit_bake(types.SimpleNamespace(model="STUB"), "m.safetensors", "host")
