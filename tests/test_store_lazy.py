"""ModelStore low_rss routing: fused lazy swap or full reload (comfy stubbed).

The lazy branch runs exactly when it is safe (a usable record, or a base never
consumed) and reuses record entries for keys still patched, capturing only novel
keys. A failed swap, mid-pipeline included, drops the slot and reloads; a typed
UnsupportedModelError propagates instead. A failed capture lets the swap or bake
finish but voids the unbake record, so the next stack change reloads. An
interruption discards the slot and propagates; a failed discard poisons the slot."""
import sys
import types
from types import SimpleNamespace

import pytest

from dgx_monarch.actor import model_store as ms
from dgx_monarch.actor.model_store import ModelStore, StoredModel, lora_signature
from dgx_monarch.actor.store_detect import LivePrecisionEvidence
from dgx_monarch.actor.unbake import UnbakeRecord
from dgx_monarch.safetensors_header import UnsupportedSafetensorsDtypeError
from slab_lifetime_helpers import reset_slab_lifetime

STACK_A = [{"name": "a.safetensors", "strength": 1.0}]
STACK_A2 = [{"name": "a.safetensors", "strength": 0.5}]     # strength-only change
STACK_B = [{"name": "b.safetensors", "strength": 0.5}]
_REAL_DROP = ModelStore._drop


def _record(keys):
    return UnbakeRecord(path="/fake/m.safetensors", file_size=1, file_mtime_ns=1,
                        mapped={k: SimpleNamespace(tag=f"FT[{k}]", cast_to=None, dtype=None)
                                for k in keys})


@pytest.fixture
def rig(monkeypatch):
    reset_slab_lifetime(monkeypatch)
    calls = SimpleNamespace(restore=0, merge=0, drop=0, load=0, build=0,
                            capture=0, bake=0, verify=0, captured_keys=None)

    comfy = types.ModuleType("comfy")
    comfy_sd = types.ModuleType("comfy.sd")

    def load_diffusion_model(path, model_options=None):
        calls.load += 1
        return SimpleNamespace(model="FRESH_MODEL")

    comfy_sd.load_diffusion_model = load_diffusion_model
    comfy.sd = comfy_sd
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd)

    monkeypatch.setattr(ms, "resolve_model_path", lambda kind, name: f"/fake/{name}")
    monkeypatch.setattr(ms, "_detect_checkpoint_kind", lambda _path, _kind: None)
    monkeypatch.setattr(
        "dgx_monarch.gate_ledger.artifact_signature", lambda path: path)
    monkeypatch.setattr(ms, "_detect_quant_kind", lambda patcher, kind: "bf16")
    monkeypatch.setattr(ms, "_detect_family", lambda patcher: "krea2")

    def fake_merge_and_free(active, base_path=None):
        calls.merge += 1
        return _record(list(active.patches)) if base_path else None

    monkeypatch.setattr(ms, "_merge_and_free", fake_merge_and_free)

    def fake_build_active(self, base_patcher, lora_stack):
        calls.build += 1
        return SimpleNamespace(tag="ACTIVE", loras=lora_signature(lora_stack),
                               model="LIVE_MODEL",
                               patches={e["name"]: [] for e in lora_stack or []})

    monkeypatch.setattr(ModelStore, "_build_active", fake_build_active)

    def fake_bake_key(active, key):
        calls.bake += 1

    monkeypatch.setattr(ModelStore, "_bake_key", staticmethod(fake_bake_key))

    def fake_verify(self, active, key):
        calls.verify += 1
        fake_bake_key(active, key)      # the real one bakes after the oracle

    monkeypatch.setattr(ModelStore, "_verify_key", fake_verify)

    def fake_drop(self, slot):
        calls.drop += 1
        if slot == "uncond":
            self.uncond = None
        else:
            self.current = None

    monkeypatch.setattr(ModelStore, "_drop", fake_drop)

    from dgx_monarch.actor import unbake

    def fake_restore(model, record, after_key=None):
        calls.restore += 1
        if after_key is not None:
            for key in list(record.mapped) + list(record.quant) + list(record.resident):
                after_key(key)
        return {"restored_keys": 1, "file_backed_gib": 0.0}

    monkeypatch.setattr(unbake, "restore_pristine", fake_restore)

    def fake_capture(model, keys, base_path, **kw):
        calls.capture += 1
        calls.captured_keys = list(keys)
        return _record(keys)

    monkeypatch.setattr(unbake, "capture_unbake_record", fake_capture)

    store = ModelStore()
    store.lora_low_rss = True
    return store, calls


def _stored(lora_stack, quant="bf16", unbake="auto"):
    if unbake == "auto":
        unbake = _record([e["name"] for e in lora_stack]) if lora_stack else None
    base_key, _request_key = ModelStore.make_keys("m.safetensors", None, lora_stack, quant)
    return StoredModel(
        base_key=base_key,
        request_key=(*base_key, lora_signature(lora_stack), quant),
        base_patcher=SimpleNamespace(model="MODEL"),
        active_patcher=SimpleNamespace(tag="OLD_ACTIVE"),
        quant_kind=quant,
        precision_evidence=LivePrecisionEvidence(
            quant_kind=quant,
            live_dtype_profile=f"uniform_or_quantized_{quant}",
        ),
        family="krea2",
        artifact_identity=ms.request_artifact_identity("m.safetensors", lora_stack),
        lora_sig=lora_signature(lora_stack),
        unbake=unbake,
    )


def test_unsupported_dtype_voids_unbake_record_but_bake_continues(monkeypatch, caplog):
    from dgx_monarch.actor import store_bake, unbake

    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    loaded = []
    mm.load_models_gpu = lambda models, force_full_load=False: loaded.append(
        (list(models), force_full_load)
    )
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

    def unsupported_capture(*_args, **_kwargs):
        raise UnsupportedSafetensorsDtypeError("unsupported FUTURE dtype")

    monkeypatch.setattr(unbake, "capture_unbake_record", unsupported_capture)
    active = SimpleNamespace(
        model=object(), patches={"weight": []}, backup={"weight": object()},
        backup_buffers={"buffer": object()},
    )

    record = store_bake._merge_and_free(active, base_path="/fake/model.safetensors")

    assert record is None
    assert loaded == [([active], True)]
    assert active.patches == {} and active.backup == {} and active.backup_buffers == {}
    assert "unbake capture skipped" in caplog.text


def test_strength_change_reuses_record_no_capture(rig):
    store, calls = rig
    store.current = _stored(STACK_A)
    _active, transition = store.ensure("m.safetensors", None, STACK_A2)
    assert transition == "hot-swap"
    assert calls.restore == 1 and calls.drop == 0 and calls.load == 0
    assert calls.capture == 0                      # every key reused from the old record
    assert calls.bake == 1                         # baked during the restore pipeline
    assert "a.safetensors" in store.current.unbake.mapped
    assert store.current.lora_sig == lora_signature(STACK_A2)


def test_stack_change_captures_only_novel_keys(rig):
    store, calls = rig
    store.current = _stored(STACK_A)
    _, transition = store.ensure("m.safetensors", None, STACK_B)
    assert transition == "hot-swap"
    assert calls.restore == 1 and calls.load == 0
    assert calls.capture == 1 and calls.captured_keys == ["b.safetensors"]
    assert calls.bake == 1                         # only the novel key needed baking
    assert set(store.current.unbake.mapped) == {"b.safetensors"}


def test_lazy_swap_to_empty_stack(rig):
    store, calls = rig
    store.current = _stored(STACK_A)
    _, transition = store.ensure("m.safetensors", None, [])
    assert transition == "hot-swap"
    assert calls.restore == 1 and calls.capture == 0 and calls.bake == 0
    assert store.current.unbake is None
    assert store.current.lora_sig == ()


def test_pristine_base_captures_before_bake(rig):
    store, calls = rig
    store.current = _stored([], unbake=None)      # loaded without loras: never consumed
    _, transition = store.ensure("m.safetensors", None, STACK_A)
    assert transition == "hot-swap"
    assert calls.restore == 0 and calls.load == 0
    assert calls.capture == 1 and calls.bake == 1


def test_restore_failure_falls_back_to_full_reload(rig):
    store, calls = rig
    from dgx_monarch.actor import unbake

    def boom(model, record, after_key=None):
        raise unbake.UnbakeError("checkpoint changed")

    unbake.restore_pristine = boom               # the fixture's monkeypatch undoes this at teardown
    store.current = _stored(STACK_A)
    _, transition = store.ensure("m.safetensors", None, STACK_B)
    assert transition == "load"
    assert calls.drop >= 1 and calls.load == 1


def test_restore_failure_releases_outgoing_model_before_drop(rig, monkeypatch):
    import gc
    import weakref

    store, calls = rig
    from dgx_monarch.actor import unbake

    outgoing = _stored(STACK_A)
    ref = weakref.ref(outgoing)
    store.current = outgoing
    del outgoing

    def boom(model, record, after_key=None):
        raise unbake.UnbakeError("checkpoint changed")

    monkeypatch.setattr(unbake, "restore_pristine", boom)

    def checking_drop(self, slot):
        calls.drop += 1
        self.current = None
        gc.collect()
        assert ref() is None, "failed lazy-swap traceback still pins outgoing model"

    monkeypatch.setattr(ModelStore, "_drop", checking_drop)
    _, transition = store.ensure("m.safetensors", None, STACK_B)
    assert transition == "load"


def test_interrupted_lazy_swap_discards_old_request_before_reraise(rig, monkeypatch):
    """A non-Exception abort after mutation must not leave OLD_ACTIVE reusable."""
    import gc
    import weakref

    class SwapAbort(BaseException):
        pass

    store, calls = rig
    outgoing = _stored(STACK_A)
    outgoing.base_patcher.model = SimpleNamespace(weight="pristine")
    outgoing_ref = weakref.ref(outgoing)
    store.current = outgoing
    del outgoing
    failure = SwapAbort("lazy swap interrupted after mutation")

    def abort_bake(_active, _key):
        calls.bake += 1
        store.current.base_patcher.model.weight = "half-mutated"
        raise failure

    monkeypatch.setattr(ModelStore, "_bake_key", staticmethod(abort_bake))
    store.swap_verify = 0

    with pytest.raises(SwapAbort) as caught:
        store.ensure("m.safetensors", None, STACK_A2)

    assert caught.value is failure
    assert store.current is None
    assert calls.restore == calls.bake == calls.drop == 1 and calls.load == 0
    caught.value.__traceback__ = None
    del caught
    gc.collect()
    assert outgoing_ref() is None

    _active, transition = store.ensure("m.safetensors", None, STACK_A)
    assert transition == "load"
    assert calls.load == 1


@pytest.mark.parametrize("broken_boundary", ["logger", "frame-clear"])
def test_lazy_swap_recovery_diagnostics_cannot_mask_or_bypass_discard(
    rig, monkeypatch, broken_boundary,
):
    class SwapAbort(BaseException):
        pass

    class DiagnosticAbort(BaseException):
        pass

    store, calls = rig
    original = _stored(STACK_A)
    original.base_patcher.model = SimpleNamespace(weight="pristine")
    store.current = original
    primary = SwapAbort("primary lazy-swap abort")

    def abort_bake(_active, _key):
        calls.bake += 1
        store.current.base_patcher.model.weight = "half-mutated"
        raise primary

    monkeypatch.setattr(ModelStore, "_bake_key", staticmethod(abort_bake))
    if broken_boundary == "logger":
        monkeypatch.setattr(
            ms.log, "warning",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                DiagnosticAbort("logger interrupted recovery")
            ),
        )
    else:
        monkeypatch.setattr(
            ms.slab_lifetime, "_clear_exception_frames",
            lambda *_args: (_ for _ in ()).throw(
                DiagnosticAbort("frame clearing interrupted recovery")
            ),
        )
    store.swap_verify = 0

    with pytest.raises(SwapAbort) as caught:
        store.ensure("m.safetensors", None, STACK_A2)

    assert caught.value is primary
    assert store.current is None
    assert store.snapshot()["cleanup_failed_slots"] == []
    assert calls.restore == calls.bake == calls.drop == 1


def test_interrupted_lazy_swap_preserves_primary_when_drop_fails(
    rig, monkeypatch, caplog,
):
    """A failed discard poisons the slot and adds its evidence as a note; the swap abort stays the raised error."""

    class SwapAbort(BaseException):
        pass

    class CleanupAbort(BaseException):
        pass

    store, calls = rig
    original = _stored(STACK_A)
    original.base_patcher.model = SimpleNamespace(weight="pristine")
    store.current = original
    primary = SwapAbort("primary lazy-swap abort")
    cleanup = CleanupAbort("drop interrupted")

    def abort_bake(_active, _key):
        calls.bake += 1
        store.current.base_patcher.model.weight = "half-mutated"
        raise primary

    def abort_drop(_self, _slot):
        calls.drop += 1
        raise cleanup

    monkeypatch.setattr(ModelStore, "_bake_key", staticmethod(abort_bake))
    monkeypatch.setattr(ModelStore, "_drop", abort_drop)
    store.swap_verify = 0

    with pytest.raises(SwapAbort) as caught:
        store.ensure("m.safetensors", None, STACK_A2)

    assert caught.value is primary
    assert store.current is original  # retained ownership, but poisoned below
    assert calls.restore == calls.bake == calls.drop == 1
    assert store.snapshot()["cleanup_failed_slots"] == ["cond"]
    assert any("drop interrupted" in note for note in primary.__notes__)
    assert "drop interrupted" in caplog.text
    assert cleanup.__traceback__ is None
    with pytest.raises(RuntimeError, match="previous unload failed"):
        store.ensure("m.safetensors", None, STACK_A)
    assert calls.load == 0


def test_lazy_swap_cleanup_cancellation_outranks_ordinary_failure(
    rig,
    monkeypatch,
):
    store, calls = rig
    original = _stored(STACK_A)
    original.base_patcher.model = SimpleNamespace(weight="pristine")
    store.current = original
    primary = RuntimeError("ordinary lazy-swap failure")
    cancellation = KeyboardInterrupt("lazy-swap discard cancelled")

    def fail_bake(_active, _key):
        calls.bake += 1
        raise primary

    def cancel_drop(_self, _slot):
        calls.drop += 1
        raise cancellation

    monkeypatch.setattr(ModelStore, "_bake_key", staticmethod(fail_bake))
    monkeypatch.setattr(ModelStore, "_drop", cancel_drop)
    store.swap_verify = 0

    with pytest.raises(KeyboardInterrupt) as caught:
        store.ensure("m.safetensors", None, STACK_A2)

    assert caught.value is cancellation
    assert caught.value.__cause__ is primary
    assert store.current is original
    assert store.snapshot()["cleanup_failed_slots"] == ["cond"]
    assert calls.restore == calls.bake == calls.drop == 1


def test_lazy_swap_hostile_add_note_lookup_cannot_mask_primary(
    rig,
    monkeypatch,
):
    class HostilePrimary(RuntimeError):
        def __getattribute__(self, name):
            if name == "add_note":
                raise KeyboardInterrupt("diagnostic attribute lookup failed")
            return super().__getattribute__(name)

    store, calls = rig
    original = _stored(STACK_A)
    original.base_patcher.model = SimpleNamespace(weight="pristine")
    store.current = original
    primary = HostilePrimary("ordinary lazy-swap failure")
    cleanup = RuntimeError("ordinary discard failure")

    monkeypatch.setattr(
        ModelStore,
        "_bake_key",
        staticmethod(lambda *_args: (_ for _ in ()).throw(primary)),
    )
    monkeypatch.setattr(
        ModelStore,
        "_drop",
        lambda *_args: (_ for _ in ()).throw(cleanup),
    )
    store.swap_verify = 0

    with pytest.raises(HostilePrimary) as caught:
        store.ensure("m.safetensors", None, STACK_A2)

    assert caught.value is primary
    assert caught.value.__cause__ is cleanup
    assert store.current is original
    assert store.snapshot()["cleanup_failed_slots"] == ["cond"]
    assert calls.restore == 1


def test_lazy_swap_double_failure_survives_all_broken_diagnostics(
    rig,
    monkeypatch,
):
    class DiagnosticAbort(BaseException):
        pass

    class SwapAbort(BaseException):
        def add_note(self, _note):
            raise DiagnosticAbort("note attachment interrupted")

    class CleanupAbort(BaseException):
        def __str__(self):
            raise DiagnosticAbort("cleanup formatting interrupted")

    store, calls = rig
    original = _stored(STACK_A)
    original.base_patcher.model = SimpleNamespace(weight="pristine")
    store.current = original
    primary = SwapAbort("primary lazy-swap abort")
    cleanup = CleanupAbort("drop interrupted")

    def abort_bake(_active, _key):
        calls.bake += 1
        store.current.base_patcher.model.weight = "half-mutated"
        raise primary

    def abort_drop(_self, _slot):
        calls.drop += 1
        raise cleanup

    def abort_diagnostic(*_args, **_kwargs):
        raise DiagnosticAbort("diagnostic boundary interrupted")

    monkeypatch.setattr(ModelStore, "_bake_key", staticmethod(abort_bake))
    monkeypatch.setattr(ModelStore, "_drop", abort_drop)
    monkeypatch.setattr(ms.log, "warning", abort_diagnostic)
    monkeypatch.setattr(
        ms.slab_lifetime, "_clear_exception_frames", abort_diagnostic)
    store.swap_verify = 0

    with pytest.raises(SwapAbort) as caught:
        store.ensure("m.safetensors", None, STACK_A2)

    assert caught.value is primary
    assert store.current is original
    assert store.snapshot()["cleanup_failed_slots"] == ["cond"]
    assert calls.restore == calls.bake == calls.drop == 1
    assert cleanup.__traceback__ is None
    with pytest.raises(RuntimeError, match="previous unload failed"):
        store.ensure("m.safetensors", None, STACK_A)


def test_lazy_swap_drop_interruption_after_unpublish_retains_exact_owner(
    rig, monkeypatch,
):
    """When _drop is interrupted as it clears the slot, after the global unload, it
    puts the same resident back and blocks reuse."""
    import gc
    import weakref

    class SwapAbort(BaseException):
        pass

    class CleanupAbort(BaseException):
        pass

    primary = SwapAbort("primary lazy-swap abort")
    cleanup = CleanupAbort("interrupted immediately after slot unpublish")

    class InterruptingStore(ModelStore):
        def __init__(self):
            self._current_value = None
            self.interrupt_unpublish = False
            super().__init__()

        @property
        def current(self):
            return self._current_value

        @current.setter
        def current(self, value):
            self._current_value = value
            if self.interrupt_unpublish and value is None:
                self.interrupt_unpublish = False
                raise cleanup

    store, calls = rig
    store = InterruptingStore()
    store.lora_low_rss = True
    monkeypatch.setattr(InterruptingStore, "_drop", _REAL_DROP)

    mm = types.ModuleType("comfy.model_management")
    mm.unload_all_models = lambda: None
    mm.soft_empty_cache = lambda: None
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setattr(sys.modules["comfy"], "model_management", mm, raising=False)

    close_calls = []

    class OwnedSlab:
        def close(self):
            close_calls.append("closed")

    slab = OwnedSlab()
    outgoing = _stored(STACK_A)
    outgoing.active_patcher.cleanup = lambda: None
    outgoing.base_patcher.model = SimpleNamespace(weight="pristine")
    outgoing.slab = slab
    outgoing_ref = weakref.ref(outgoing)
    slab_ref = weakref.ref(slab)
    store.current = outgoing
    del outgoing, slab

    def abort_bake(_active, _key):
        calls.bake += 1
        store.current.base_patcher.model.weight = "half-mutated"
        store.interrupt_unpublish = True
        raise primary

    monkeypatch.setattr(InterruptingStore, "_bake_key", staticmethod(abort_bake))
    store.swap_verify = 0

    with pytest.raises(SwapAbort) as caught:
        store.ensure("m.safetensors", None, STACK_A2)

    assert caught.value is primary
    assert store.current is outgoing_ref()
    assert outgoing_ref() is not None and slab_ref() is not None
    assert store.snapshot()["cleanup_failed_slots"] == ["cond"]
    assert any("immediately after slot unpublish" in note for note in primary.__notes__)
    assert cleanup.__traceback__ is None
    with pytest.raises(RuntimeError, match="previous model load"):
        store.ensure("m.safetensors", None, STACK_A)

    caught.value.__traceback__ = None
    del caught
    store._drop("cond")
    gc.collect()
    assert close_calls == ["closed"]
    assert outgoing_ref() is None and slab_ref() is None


def test_lazy_swap_drop_interrupt_after_global_unload_retains_exact_owner(
    rig, monkeypatch,
):
    """An interruption on the first line after the global unload cannot orphan the resident slab."""
    import gc
    import inspect
    import weakref

    from dgx_monarch.actor import slab_lifetime

    class SwapAbort(BaseException):
        pass

    class CleanupAbort(BaseException):
        pass

    class TracingStore(ModelStore):
        pass

    _fixture_store, calls = rig
    monkeypatch.setattr(TracingStore, "_drop", _REAL_DROP)
    reset_slab_lifetime(monkeypatch)
    store = TracingStore()
    store.lora_low_rss = True

    unloads = []
    mm = types.ModuleType("comfy.model_management")
    mm.unload_all_models = lambda: unloads.append("unload")
    mm.soft_empty_cache = lambda: None
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setattr(sys.modules["comfy"], "model_management", mm, raising=False)

    close_calls = []

    class OwnedSlab:
        def close(self):
            close_calls.append("closed")

    primary = SwapAbort("primary lazy-swap abort")
    cleanup = CleanupAbort("interrupted after confirmed global unload")
    slab = OwnedSlab()
    outgoing = _stored(STACK_A)
    outgoing.active_patcher.cleanup = lambda: None
    outgoing.base_patcher.model = SimpleNamespace(weight="pristine")
    outgoing.slab = slab
    outgoing_ref, slab_ref = weakref.ref(outgoing), weakref.ref(slab)
    store.current = outgoing
    del outgoing, slab

    def abort_bake(_active, _key):
        calls.bake += 1
        store.current.base_patcher.model.weight = "half-mutated"
        raise primary

    monkeypatch.setattr(TracingStore, "_bake_key", staticmethod(abort_bake))
    store.swap_verify = 0
    lines, first = inspect.getsourcelines(_REAL_DROP)
    interrupt_line = first + next(
        index for index, line in enumerate(lines)
        if "self._drop_failed.add(slot)" in line
    )
    armed = True

    def interrupt(frame, event, _arg):
        nonlocal armed
        if (armed and event == "line" and frame.f_code is _REAL_DROP.__code__
                and frame.f_lineno == interrupt_line):
            armed = False
            raise cleanup
        return interrupt

    sys.settrace(interrupt)
    try:
        with pytest.raises(SwapAbort) as caught:
            store.ensure("m.safetensors", None, STACK_A2)
    finally:
        sys.settrace(None)

    assert caught.value is primary
    assert store.current is outgoing_ref() and slab_ref() is not None
    assert unloads == ["unload"] and close_calls == []
    assert store.snapshot()["cleanup_failed_slots"] == ["cond"]
    assert not slab_lifetime.cleanup_pending()
    assert any("confirmed global unload" in note for note in primary.__notes__)
    assert cleanup.__traceback__ is None
    with pytest.raises(RuntimeError, match="previous unload failed"):
        store.ensure("m.safetensors", None, STACK_A)

    caught.value.__traceback__ = None
    del caught
    store._drop("cond")
    gc.collect()
    assert unloads == ["unload", "unload"] and close_calls == ["closed"]
    assert outgoing_ref() is None and slab_ref() is None
    assert not slab_lifetime.cleanup_pending()
    _active, transition = store.ensure("m.safetensors", None, STACK_A)
    assert transition == "load" and calls.load == 1


def test_quant_stack_change_without_record_reloads(rig):
    store, calls = rig
    store.current = _stored(STACK_A, quant="fp8", unbake=None)
    _, transition = store.ensure("m.safetensors", None, STACK_B)
    assert transition == "load"
    assert calls.restore == 0 and calls.drop == 1


def test_missing_record_keeps_reload_contract(rig):
    store, calls = rig
    store.current = _stored(STACK_A, unbake=None)  # e.g. capture failed at load
    _, transition = store.ensure("m.safetensors", None, STACK_B)
    assert transition == "load"
    assert calls.restore == 0 and calls.drop == 1


def test_exact_request_still_reuses(rig):
    store, calls = rig
    store.current = _stored(STACK_A)
    active, transition = store.ensure("m.safetensors", None, STACK_A)
    assert transition == "reuse"
    assert active.tag == "OLD_ACTIVE"
    assert calls.restore == 0 and calls.load == 0


def test_bake_failure_drops_slot_and_reloads(rig, monkeypatch):
    store, calls = rig

    def boom_bake(active, key):
        calls.bake += 1
        raise RuntimeError("bake exploded")

    monkeypatch.setattr(ModelStore, "_bake_key", staticmethod(boom_bake))
    store.swap_verify = 0                        # route through the plain bake
    store.current = _stored(STACK_A)
    _, transition = store.ensure("m.safetensors", None, STACK_A2)
    # The bake failed mid-pipeline, so the slot drops and the full reload runs
    # comfy's stock load, then merge-and-free.
    assert transition == "load"
    assert calls.drop >= 1 and calls.load == 1 and calls.merge == 1
    assert "a.safetensors" in store.current.unbake.mapped


def test_fsdp_tagged_quantized_key_refusal_still_falls_through_to_reload(
    rig, monkeypatch,
):
    """actor/fsdp_lora.py's class P quantized-key and dtype-mismatch refusals keep the reload contract.

    They raise UnbakeError, not UnsupportedModelError, so store_bake.lazy_swap_transition's
    ``except UnsupportedModelError: raise`` does not match them, and a tagged UnbakeError falls
    through to a fresh reload like an untagged one. The operator meets the typed error from that
    reload's own bake, or on the first FSDP LoRA render."""
    from dgx_monarch.actor import fsdp_lora
    from dgx_monarch.actor.unbake import UnbakeError
    from dgx_monarch.refusal import RefusalClass, refusal

    store, calls = rig
    stored = _stored(STACK_A)
    stored.base_patcher = SimpleNamespace(model="MODEL", _dgxm_fsdp=True)
    store.current = stored

    def boom_fsdp_lazy_swap(_store, _stored, _active, _unet_name):
        raise UnbakeError(refusal(
            RefusalClass.PHYSICS,
            "LoRA on quantized shards is not admitted. Use a resident "
            "topology for this file, or a bf16, fp16, or plain-dtype fp8 "
            "checkpoint under FSDP."))

    monkeypatch.setattr(fsdp_lora, "lazy_swap", boom_fsdp_lazy_swap)
    _, transition = store.ensure("m.safetensors", None, STACK_B)
    assert transition == "load"
    assert calls.drop >= 1 and calls.load == 1


def test_novel_capture_failure_returns_reload_contract(rig, monkeypatch):
    store, calls = rig
    from dgx_monarch.actor import unbake

    def boom_capture(model, keys, base_path, **kw):
        calls.capture += 1
        raise OSError("disk went away")

    monkeypatch.setattr(unbake, "capture_unbake_record", boom_capture)
    store.current = _stored(STACK_A)
    _, transition = store.ensure("m.safetensors", None, STACK_B)
    # The swap completes with correct weights, but no record covers the new
    # set, so the next stack change reloads.
    assert transition == "hot-swap"
    assert store.current.unbake is None
    assert calls.bake == 1                        # novel key still baked


def test_ambient_verify_samples_and_delegates(rig):
    store, calls = rig
    store.current = _stored(STACK_A)
    _, transition = store.ensure("m.safetensors", None, STACK_A2)
    assert transition == "hot-swap"
    # swap_verify defaults to 2, so the one patched key takes the verified bake
    assert calls.verify == 1 and calls.bake == 1


def test_ambient_verify_mismatch_reloads_cleanly(rig, monkeypatch):
    store, calls = rig
    from dgx_monarch.actor import unbake

    def boom_verify(self, active, key):
        store.verify_failures += 1
        raise unbake.UnbakeError("ambient bake verify MISMATCH")

    monkeypatch.setattr(ModelStore, "_verify_key", boom_verify)
    store.current = _stored(STACK_A)
    _, transition = store.ensure("m.safetensors", None, STACK_A2)
    assert transition == "load"                   # the slot drops, then a stock reload
    assert store.verify_failures == 1
    assert calls.load == 1


def test_ambient_verify_off_skips_oracle(rig, monkeypatch):
    store, calls = rig

    def never(self, active, key):
        raise AssertionError("verify must not run when swap_verify=0")

    monkeypatch.setattr(ModelStore, "_verify_key", never)
    store.swap_verify = 0
    store.current = _stored(STACK_A)
    _, transition = store.ensure("m.safetensors", None, STACK_A2)
    assert transition == "hot-swap" and calls.bake == 1


def _uuid_active(monkeypatch):
    """A built patcher whose model carries comfy's loaded-uuid field."""
    model = SimpleNamespace(current_weight_patches_uuid="OLD_UUID")

    def build(self, base_patcher, lora_stack):
        return SimpleNamespace(tag="ACTIVE", model=model, patches_uuid="NEW_UUID",
                               patches={e["name"]: [] for e in lora_stack or []})

    monkeypatch.setattr(ModelStore, "_build_active", build)
    return model


def test_a_stock_hot_swap_names_the_new_patch_uuid(rig, monkeypatch):
    """Otherwise comfy reads the old uuid, unpatches the whole model to the host
    and reloads it on the next render (store_bake._adopt_patch_uuid gives the cost)."""
    store, _calls = rig
    model = _uuid_active(monkeypatch)
    store.current = _stored(STACK_A)
    _, transition = store.ensure("m.safetensors", None, STACK_A2)
    assert transition == "hot-swap"
    assert model.current_weight_patches_uuid == "NEW_UUID"


def test_slab_and_fsdp_hot_swaps_keep_the_loaded_uuid(rig, monkeypatch):
    """Slab and FSDP residents pin their offload device, so _adopt_patch_uuid leaves their uuid alone."""
    from dgx_monarch.actor import store_bake

    for stored in (SimpleNamespace(slab=object(), base_patcher=SimpleNamespace()),
                   SimpleNamespace(slab=None, base_patcher=SimpleNamespace(_dgxm_fsdp=True))):
        model = SimpleNamespace(current_weight_patches_uuid="OLD_UUID")
        store_bake._adopt_patch_uuid(stored, SimpleNamespace(model=model, patches_uuid="NEW_UUID"))
        assert model.current_weight_patches_uuid == "OLD_UUID"
