"""CPU contracts for deferred, generation-bound cuDNN Ring preparation."""
from __future__ import annotations

import pytest

from dgx_monarch import adapters
from dgx_monarch.actor.attention_dispatch import _AttentionDispatch
from dgx_monarch.adapters import cudnn_ring_attention
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.topology import Topology

RING = {"dp": 1, "cfg": 1, "ulysses": 1, "ring": 2, "fsdp": False}


@pytest.fixture
def builders(monkeypatch):
    generic, native = [], []

    def stock(kernel, sync):
        generic.append((kernel, sync))
        return lambda value: ("stock", kernel, value)

    def prepared(sync):
        native.append(sync)
        return lambda value: ("native", value)

    monkeypatch.setattr(adapters, "make_usp_attention", stock)
    monkeypatch.setattr(cudnn_ring_attention, "make_cudnn_ring_usp_attention", prepared)
    return generic, native


def configured(kernel="TORCH_CUDNN", topology=None, generation=1):
    dispatch = _AttentionDispatch()
    dispatch.configure(kernel, True, topology=topology or RING,
                       world=2, setup_generation=generation)
    return dispatch


def test_native_probe_is_deferred_and_cached_per_implementation(builders):
    generic, native = builders
    dispatch = configured()
    assert generic == [("TORCH_CUDNN", True)] and native == []
    with pytest.raises(UnsupportedModelError, match="normalization check"):
        dispatch("qkv")
    dispatch.prepare_native_attention()
    assert dispatch("qkv") == ("native", "qkv")
    dispatch.prepare_native_attention()
    assert native == [True]
    assert (dispatch.kernel, dispatch.effective_kernel) == ("TORCH_CUDNN", "TORCH_CUDNN")


@pytest.mark.parametrize("kernel,topology", [
    ("TORCH_FLASH", RING),
    ("TORCH_CUDNN", {"ulysses": 2, "ring": 1, "fsdp": False}),
    ("TORCH_CUDNN", {"ulysses": 2, "ring": 1, "fsdp": True}),
])
def test_other_attention_paths_do_not_probe_native_ring(builders, kernel, topology):
    dispatch = configured(kernel, topology)
    dispatch.prepare_native_attention()
    assert builders[1] == []
    assert dispatch("qkv") == ("stock", kernel, "qkv")


def test_late_head_binding_skips_substituted_flash_then_prepares_cudnn(builders):
    dispatch = configured()
    dispatch.bind_capability("ideogram4", 256)
    dispatch.prepare_native_attention()
    assert dispatch.effective_kernel == "TORCH_FLASH" and builders[1] == []
    dispatch.bind_capability("anima", 128)
    dispatch.prepare_native_attention()
    assert dispatch("qkv") == ("native", "qkv")
    assert builders[1] == [True]


def test_warm_kernel_switch_and_new_generation_reprepare(builders):
    dispatch = configured()
    dispatch.prepare_native_attention()
    dispatch.configure("TORCH_FLASH", True, topology=RING, world=2, setup_generation=1)
    assert dispatch("qkv")[0] == "stock"
    dispatch.configure("TORCH_CUDNN", True, topology=RING, world=2, setup_generation=1)
    with pytest.raises(UnsupportedModelError):
        dispatch("qkv")
    dispatch.prepare_native_attention()
    dispatch.invalidate()
    dispatch.configure("TORCH_CUDNN", True, topology=RING, world=2, setup_generation=2)
    with pytest.raises(UnsupportedModelError):
        dispatch("qkv")
    dispatch.prepare_native_attention()
    assert builders[1] == [True, True, True]


def test_failed_probe_cannot_leave_zero_lse_fallback_callable(builders, monkeypatch):
    dispatch = configured()

    def broken(_sync):
        raise RuntimeError("native LSE unavailable")

    monkeypatch.setattr(cudnn_ring_attention, "make_cudnn_ring_usp_attention", broken)
    with pytest.raises(RuntimeError, match="native LSE unavailable"):
        dispatch.prepare_native_attention()
    with pytest.raises(UnsupportedModelError, match="normalization check"):
        dispatch("qkv")


def test_changed_setup_during_probe_does_not_publish_stale_binding(builders, monkeypatch):
    dispatch = configured()

    def retired(_sync):
        dispatch.invalidate()
        return lambda value: value

    monkeypatch.setattr(cudnn_ring_attention, "make_cudnn_ring_usp_attention", retired)
    with pytest.raises(RuntimeError, match="changed during"):
        dispatch.prepare_native_attention()
    assert dispatch._impl is None and dispatch._cudnn_ring_impl is None


def test_wan_view_uses_prepared_cudnn_and_keeps_generation_guard(builders):
    dispatch = configured()
    view = dispatch.for_wan()
    with pytest.raises(UnsupportedModelError, match="normalization check"):
        view("qkv")
    dispatch.prepare_native_attention()
    assert view("qkv") == ("native", "qkv")
    dispatch.invalidate()
    with pytest.raises(RuntimeError, match="stale setup"):
        view("qkv")


def test_ring_fsdp_refusal_remains_in_force():
    with pytest.raises(ValueError, match="cannot combine with FSDP"):
        Topology(world=2, ring=2, fsdp=True).validate()
