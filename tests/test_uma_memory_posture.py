"""Loop-hosted workers on unified memory must inherit the validated driver memory posture.

The Init node does not pass the whole driver baseline (--disable-pinned-memory
--disable-async-offload --reserve-vram) to the workers, so _uma_memory_defaults
applies it on UMA by default. These tests fail if that posture regresses or
stops respecting explicit overrides.
"""
from dgx_monarch.actor.comfy_bridge import _uma_memory_defaults


def test_uma_posture_applied_on_integrated():
    d = _uma_memory_defaults({}, integrated=True)
    assert d["disable_pinned_memory"] is True
    assert d["disable_async_offload"] is True
    assert d["reserve_vram_gb"] == 8.0
    assert d["safetensors_backend"] == "pread"


def test_mmap_fallback_opts_out_of_pread():
    d = _uma_memory_defaults({"mmap_fallback": True}, integrated=True)
    assert d.get("safetensors_backend") != "pread"
    assert d["disable_pinned_memory"] is True


def test_explicit_worker_args_win():
    d = _uma_memory_defaults(
        {"disable_pinned_memory": False, "reserve_vram_gb": 4.0}, integrated=True)
    assert d["disable_pinned_memory"] is False
    assert d["reserve_vram_gb"] == 4.0
    assert d["disable_async_offload"] is True


def test_discrete_gpu_untouched():
    # On a discrete GPU, pinned host RAM is not GPU memory and async offload
    # hides weight-transfer time, so the UMA posture (and pread) must not be
    # forced there.
    assert _uma_memory_defaults({"foo": 1}, integrated=False) == {"foo": 1}


def test_pread_shim_is_idempotent_and_defaults_backend():
    import safetensors

    from dgx_monarch.actor.comfy_bridge import _enable_pread_backend

    orig = safetensors.safe_open
    captured = {}

    def _spy(*a, **kw):
        captured.clear()
        captured.update(kw)
        return None

    safetensors.safe_open = _spy   # install before wrapping so the shim forwards to it
    try:
        _enable_pread_backend()
        shim = safetensors.safe_open
        assert getattr(shim, "_dgxm_pread", False) is True
        _enable_pread_backend()
        assert safetensors.safe_open is shim
        shim("f", "pt", device="cpu")
        assert captured["backend"] == "pread"
        shim("f", "pt", device="cpu", backend="mmap")
        assert captured["backend"] == "mmap"
    finally:
        safetensors.safe_open = orig


def test_low_rss_defaults_on_for_integrated():
    d = _uma_memory_defaults({}, integrated=True)
    assert d["lora_low_rss"] is True


def test_low_rss_explicit_off_wins():
    d = _uma_memory_defaults({"lora_low_rss": False}, integrated=True)
    assert d["lora_low_rss"] is False             # Init `off` is an override


def test_low_rss_not_forced_on_discrete():
    d = _uma_memory_defaults({}, integrated=False)
    assert "lora_low_rss" not in d


def test_probe_failure_warns_and_keeps_stock_defaults(monkeypatch, caplog):
    """A failed integration probe keeps stock posture and emits a warning."""
    import logging
    import sys
    import types

    mm_stub = types.ModuleType("comfy.model_management")

    def broken_device():
        raise RuntimeError("CUDA property query failed")

    mm_stub.get_torch_device = broken_device
    comfy = types.ModuleType("comfy")
    comfy.model_management = mm_stub
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm_stub)

    with caplog.at_level(logging.WARNING, logger="dgx_monarch.actor.comfy_bridge"):
        d = _uma_memory_defaults({"foo": 1}, integrated=None)
    assert d == {"foo": 1}
    assert "UMA posture probe failed" in caplog.text


def test_slab_loader_releases_base_before_close_when_offload_pin_fails(monkeypatch):
    """The returned patcher must be released before its raw-pointer slab is closed."""
    import gc
    import sys
    import types
    import weakref
    from contextlib import contextmanager

    import pytest

    from dgx_monarch.actor import comfy_bridge, slab_lifetime

    monkeypatch.setattr(slab_lifetime, "_RETAINED_FAILED_LOAD_SLABS", [])
    events = []
    live_models = []
    base_refs = []
    slab_refs = []

    class Base:
        offload_device = "cpu"

        def __del__(self):
            events.append("base released")

    class Slab:
        def close(self):
            assert base_refs[0]() is None
            events.append("slab closed")

    @contextmanager
    def fake_slab_load(_path, *, handoff=None):
        slab = Slab()
        if handoff is not None:
            handoff.append(slab)
        slab_refs.append(weakref.ref(slab))
        yield slab

    mm_stub = types.ModuleType("comfy.model_management")

    def broken_device():
        raise RuntimeError("device query exploded")

    mm_stub.get_torch_device = broken_device
    def unload_all_models():
        events.append("global unload")
        live_models.clear()

    mm_stub.unload_all_models = unload_all_models
    mm_stub.soft_empty_cache = lambda: events.append("soft empty")
    comfy = types.ModuleType("comfy")
    comfy.model_management = mm_stub
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm_stub)
    monkeypatch.setattr(comfy_bridge, "slab_load", fake_slab_load)

    def loader(_path, model_options=None):
        base = Base()
        base_refs.append(weakref.ref(base))
        live_models.append(base)
        return base

    with pytest.raises(RuntimeError, match="device query exploded") as caught:
        comfy_bridge.load_diffusion_model_slab("/fake/m.safetensors", {}, loader)
    assert events.index("global unload") < events.index("slab closed")
    assert events.index("base released") < events.index("slab closed")
    assert slab_lifetime.retained_count() == 0
    caught.value.__traceback__ = None
    del caught
    gc.collect()
    assert slab_refs[0]() is None


def test_pin_cleanup_baseexception_retains_slab_until_explicit_unload(monkeypatch):
    """A cleanup interruption cannot mask the pin error or orphan the mapping."""
    import gc
    import sys
    import types
    import weakref
    from contextlib import contextmanager

    import pytest

    from dgx_monarch.actor import comfy_bridge, slab_lifetime
    from dgx_monarch.actor.model_store import ModelStore

    class PinAbort(BaseException):
        pass

    class CleanupAbort(BaseException):
        pass

    monkeypatch.setattr(slab_lifetime, "_RETAINED_FAILED_LOAD_SLABS", [])
    live_models = []
    refs = {}
    close_order = []

    class Base:
        offload_device = "cpu"

    class Slab:
        def close(self):
            assert refs["base"]() is None
            close_order.append("close")

    @contextmanager
    def fake_slab_load(_path, *, handoff=None):
        slab = Slab()
        if handoff is not None:
            handoff.append(slab)
        refs["slab"] = weakref.ref(slab)
        yield slab

    mm_stub = types.ModuleType("comfy.model_management")
    mm_stub.get_torch_device = lambda: (_ for _ in ()).throw(PinAbort("pin failed"))
    mm_stub.unload_all_models = lambda: (_ for _ in ()).throw(
        CleanupAbort("cleanup interrupted"))
    mm_stub.soft_empty_cache = lambda: None
    comfy = types.ModuleType("comfy")
    comfy.model_management = mm_stub
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm_stub)
    monkeypatch.setattr(comfy_bridge, "slab_load", fake_slab_load)

    def loader(_path, model_options=None):
        base = Base()
        refs["base"] = weakref.ref(base)
        live_models.append(base)
        return base

    with pytest.raises(PinAbort, match="pin failed") as caught:
        comfy_bridge.load_diffusion_model_slab("/fake/m.safetensors", {}, loader)
    caught.value.__traceback__ = None
    del caught
    gc.collect()
    assert slab_lifetime.retained_count() == 1
    assert refs["slab"]() is not None
    assert close_order == []

    def healthy_unload():
        live_models.clear()

    monkeypatch.setattr(mm_stub, "unload_all_models", healthy_unload)
    ModelStore().unload_all()
    gc.collect()
    assert close_order == ["close"]
    assert slab_lifetime.retained_count() == 0
    assert refs["slab"]() is None
