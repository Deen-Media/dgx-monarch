"""ModelStore.ensure() ownership handoffs and failed-load resource ownership.

test_store_lazy.py covers the lazy-swap branch. The rest of the full-load
surface sits beside this file: the strict FSDP launch preflight in
test_store_fsdp_preflight.py, slab routing and slab teardown in
test_store_slab_routing.py, reuse routing and artifact identity in
test_store_reuse_identity.py, and the family memo in test_store_family_memo.py.
"""
import dis
import os
import sys
from types import SimpleNamespace

import pytest

from dgx_monarch.actor import model_store as ms
from dgx_monarch.constants import TRANSITION_REUSE
from slab_lifetime_helpers import reset_slab_lifetime
from store_ensure_helpers import (
    STACK,
    _FakeSlab,
    _run_with_instruction_abort,
    _run_with_recovery_abort,
    _target_after_named_call,
    _write_one_tensor_safetensors,
)
from store_ensure_helpers import rig as rig  # a fixture tests ask for by name.


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize(
    "field",
    ["_adoption_model", "_adoption_attr", "_adoption_store", "_adoption_ready"],
)
def test_partial_adoption_record_never_mistakes_an_empty_slot_for_publication(
    monkeypatch,
    field,
):
    from dgx_monarch.actor import slab_lifetime
    from dgx_monarch.actor.store_load_ownership import FreshLoadOwnership

    class AdoptAbort(BaseException):
        pass

    class Resource:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    reset_slab_lifetime(monkeypatch)
    ownership = FreshLoadOwnership()
    resource = Resource()
    ownership.pin_handoff.append(resource)
    store = SimpleNamespace(current=None)
    stored = object()
    code = FreshLoadOwnership.prepare_adoption.__code__
    instructions = list(dis.get_instructions(code))
    store_index = next(
        index for index, instruction in enumerate(instructions)
        if instruction.opname == "STORE_ATTR" and instruction.argval == field
    )
    target = instructions[store_index + 1]
    primary = AdoptAbort(f"{field} publication interrupted")

    with pytest.raises(AdoptAbort) as caught:
        _run_with_instruction_abort(
            code,
            target.offset,
            primary,
            lambda: ownership.prepare_adoption(store, "current", stored),
        )

    assert caught.value is primary
    assert not ownership._is_published()
    ownership.close()
    assert resource.closed == 1
    assert ownership.pin_handoff == []
    slab_lifetime.confirm_prepublished_resource(
        ownership, ownership._poisoned_before)
    assert not slab_lifetime.cleanup_pending()


def test_prepublished_transaction_close_never_closes_published_children(
    monkeypatch,
):
    from dgx_monarch.actor import slab_lifetime
    from dgx_monarch.actor.store_load_ownership import FreshLoadOwnership

    class Resource:
        def __init__(self):
            self.closed = 0
            self.confirmed = 0

        def close(self):
            self.closed += 1

        def confirm_handoff(self):
            self.confirmed += 1

    reset_slab_lifetime(monkeypatch)
    ownership = FreshLoadOwnership()
    pin = Resource()
    slab = Resource()
    ownership.pin_handoff.append(pin)
    ownership.slab_handoff.append(slab)
    store = SimpleNamespace(current=None)
    stored = object()
    ownership.prepare_adoption(store, "current", stored)
    store.current = stored

    ownership.close()

    assert pin.closed == slab.closed == 0
    assert slab.confirmed == 1
    assert ownership.pin_handoff == ownership.slab_handoff == []
    slab_lifetime.confirm_prepublished_resource(
        ownership, ownership._poisoned_before)
    assert not slab_lifetime.cleanup_pending()


@pytest.mark.parametrize("cancellation_first", [False, True])
def test_fresh_load_child_close_is_cancellation_first(
    monkeypatch,
    cancellation_first,
):
    from dgx_monarch.actor.store_load_ownership import FreshLoadOwnership

    reset_slab_lifetime(monkeypatch)
    events = []
    ordinary = RuntimeError("ordinary child close failed")
    cancellation = KeyboardInterrupt("child close cancelled")

    class Resource:
        def __init__(self, name, error):
            self.name = name
            self.error = error

        def close(self):
            events.append(self.name)
            raise self.error

    ownership = FreshLoadOwnership()
    slab_error, pin_error = (
        (cancellation, ordinary)
        if cancellation_first
        else (ordinary, cancellation)
    )
    ownership.slab_handoff.append(Resource("slab", slab_error))
    ownership.pin_handoff.append(Resource("pin", pin_error))

    with pytest.raises(KeyboardInterrupt) as caught:
        ownership.close()

    assert caught.value is cancellation
    assert caught.value.__cause__ is not cancellation
    assert events == ["slab", "pin"]


@pytest.mark.parametrize("cancellation_first", [False, True])
def test_stored_model_close_is_cancellation_first_and_attempts_both(
    cancellation_first,
):
    events = []
    ordinary = RuntimeError("ordinary stored resource close failed")
    cancellation = KeyboardInterrupt("stored resource close cancelled")

    class Resource:
        def __init__(self, name, error):
            self.name = name
            self.error = error

        def close(self):
            events.append(self.name)
            raise self.error

    pin_error, slab_error = (
        (cancellation, ordinary)
        if cancellation_first
        else (ordinary, cancellation)
    )
    stored = ms.StoredModel.__new__(ms.StoredModel)
    stored.fsdp_checkpoint_pin = Resource("pin", pin_error)
    stored.slab = Resource("slab", slab_error)

    with pytest.raises(KeyboardInterrupt) as caught:
        stored.close()

    assert caught.value is cancellation
    assert caught.value.__cause__ is not cancellation
    assert events == ["pin", "slab"]


def test_published_adoption_retry_cancellation_outranks_ordinary_failure(
    rig,
    monkeypatch,
):
    from dgx_monarch.actor.store_load_ownership import FreshLoadOwnership

    store, calls = rig
    primary = RuntimeError("adoption confirmation failed")
    cancellation = KeyboardInterrupt("adoption confirmation cancelled")
    attempts = 0

    def fail_confirmation(self):
        nonlocal attempts
        attempts += 1
        assert self._is_published()
        raise primary if attempts == 1 else cancellation

    monkeypatch.setattr(
        FreshLoadOwnership,
        "disarm_and_confirm",
        fail_confirmation,
    )

    with pytest.raises(KeyboardInterrupt) as caught:
        store.ensure("m.safetensors", None, STACK)

    assert caught.value is cancellation
    assert caught.value.__cause__ is primary
    assert attempts == 2
    assert calls.load == 1
    assert store.current is not None
    assert store.snapshot()["cleanup_failed_slots"] == ["cond"]


def test_unpublished_fresh_load_cleanup_cancellation_is_not_swallowed(
    rig,
    monkeypatch,
):
    from dgx_monarch.actor import slab_lifetime
    from dgx_monarch.actor.store_load_ownership import FreshLoadOwnership

    store, calls = rig
    primary = RuntimeError("ordinary Comfy load failure")
    cancellation = KeyboardInterrupt("failed-load cleanup cancelled")
    monkeypatch.setattr(
        sys.modules["comfy.sd"],
        "load_diffusion_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
    )

    def cancel_cleanup(self, error):
        slab_lifetime.retain_failed_load_resource(self, error)
        raise cancellation

    monkeypatch.setattr(FreshLoadOwnership, "cleanup", cancel_cleanup)

    with pytest.raises(KeyboardInterrupt) as caught:
        store.ensure("m.safetensors", None, STACK)

    assert caught.value is cancellation
    assert caught.value.__cause__ is primary
    assert calls.load == 0
    assert store.current is None
    assert store.snapshot()["cleanup_failed_slots"] == ["cond"]
    assert slab_lifetime.cleanup_pending()


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_fsdp_pin_call_return_interrupt_is_recovered_from_caller_handoff(
    rig,
    monkeypatch,
    tmp_path,
):
    from dgx_monarch.actor import fsdp_checkpoint_pin, slab_lifetime

    class LoadAbort(BaseException):
        pass

    reset_slab_lifetime(monkeypatch)
    monkeypatch.setattr(fsdp_checkpoint_pin, "_PIN_POISON_OWNERS", [])
    store, calls = rig
    path = tmp_path / "model.safetensors"
    _write_one_tensor_safetensors(path, "BF16")
    monkeypatch.setattr(ms, "resolve_model_path", lambda *_args: str(path))
    monkeypatch.setattr(ms, "stock_load_preflight", lambda *_args: None)
    healthy_pin = ms.store_detect.pin_fsdp_checkpoint
    observed = {}

    def record_pin(pin_path, proof, *, handoff=None):
        observed["handoff"] = handoff
        observed["pin"] = healthy_pin(pin_path, proof, handoff=handoff)
        observed["handoff_snapshot"] = tuple(handoff)
        return observed["pin"]

    monkeypatch.setattr(ms.store_detect, "pin_fsdp_checkpoint", record_pin)
    code = ms.store_load.fresh_load.__code__
    target = _target_after_named_call(code, "pin_fsdp_checkpoint")
    assert (target.opname, target.argval) == ("POP_TOP", None)
    primary = LoadAbort("pin caller handoff interrupted")

    with pytest.raises(LoadAbort) as caught:
        _run_with_instruction_abort(
            code,
            target.offset,
            primary,
            lambda: store.ensure(
                "model.safetensors", None, None, fsdp_launch=True),
        )

    assert caught.value is primary
    pin = observed["pin"]
    assert observed["handoff_snapshot"] == (pin,)
    assert observed["handoff"] == []
    assert pin._closed
    assert pin._fd == -1
    assert not os.path.exists(pin.loader_path)
    assert fsdp_checkpoint_pin._PIN_POISON_OWNERS == []
    assert store.current is None
    assert store.snapshot()["cleanup_failed_slots"] == []
    assert not slab_lifetime.cleanup_pending()
    assert calls.load == 0


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_post_load_recovery_interrupt_retains_transaction_until_unload(
    rig,
    monkeypatch,
    tmp_path,
):
    from dgx_monarch.actor import fsdp_checkpoint_pin, slab_lifetime

    class LoadAbort(BaseException):
        pass

    class RecoveryAbort(BaseException):
        pass

    reset_slab_lifetime(monkeypatch)
    monkeypatch.setattr(fsdp_checkpoint_pin, "_PIN_POISON_OWNERS", [])
    store, calls = rig
    path = tmp_path / "model.safetensors"
    _write_one_tensor_safetensors(path, "BF16")
    monkeypatch.setattr(ms, "resolve_model_path", lambda *_args: str(path))
    monkeypatch.setattr(ms, "stock_load_preflight", lambda *_args: None)
    healthy_pin = ms.store_detect.pin_fsdp_checkpoint
    observed = {}

    def record_pin(pin_path, proof, *, handoff=None):
        observed["pin"] = healthy_pin(pin_path, proof, handoff=handoff)
        return observed["pin"]

    monkeypatch.setattr(ms.store_detect, "pin_fsdp_checkpoint", record_pin)
    code = ms.store_load.fresh_load.__code__
    primary_target = _target_after_named_call(
        code, "load_diffusion_model", latest_source=True)
    recovery_target = next(
        instruction for instruction in dis.get_instructions(code)
        if instruction.opname == "LOAD_ATTR" and instruction.argval == "cleanup"
    )
    assert (primary_target.opname, primary_target.argval) == ("STORE_FAST", "base")
    primary = LoadAbort("Comfy load return interrupted")
    recovery = RecoveryAbort("failed-load recovery interrupted")

    with pytest.raises(LoadAbort) as caught:
        _run_with_recovery_abort(
            code,
            primary_target.offset,
            recovery_target.offset,
            primary,
            recovery,
            lambda: store.ensure(
                "model.safetensors", None, None, fsdp_launch=True),
        )

    assert caught.value is primary
    pin = observed["pin"]
    assert not pin._closed
    assert os.path.exists(pin.loader_path)
    assert calls.load == 1
    assert calls.base_refs[0]() is not None
    assert store.current is None
    assert store.snapshot()["cleanup_failed_slots"] == ["cond"]
    assert slab_lifetime.cleanup_pending()
    assert slab_lifetime.retained_resource_count() == 1
    with pytest.raises(RuntimeError, match="previous model load"):
        store.ensure("model.safetensors", None, None, fsdp_launch=True)
    assert calls.load == 1

    store.unload_all()

    assert pin._closed
    assert not os.path.exists(pin.loader_path)
    assert calls.base_refs[0]() is None
    assert store.current is None
    assert store.snapshot()["cleanup_failed_slots"] == []
    assert not slab_lifetime.cleanup_pending()
    assert fsdp_checkpoint_pin._PIN_POISON_OWNERS == []


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_slot_publication_interrupt_adopts_exact_resident_for_safe_reuse(
    rig,
    monkeypatch,
    tmp_path,
):
    from dgx_monarch.actor import fsdp_checkpoint_pin, slab_lifetime

    class PublishAbort(BaseException):
        pass

    reset_slab_lifetime(monkeypatch)
    monkeypatch.setattr(fsdp_checkpoint_pin, "_PIN_POISON_OWNERS", [])
    store, calls = rig
    path = tmp_path / "model.safetensors"
    _write_one_tensor_safetensors(path, "BF16")
    monkeypatch.setattr(ms, "resolve_model_path", lambda *_args: str(path))
    monkeypatch.setattr(ms, "stock_load_preflight", lambda *_args: None)
    healthy_pin = ms.store_detect.pin_fsdp_checkpoint
    observed = {}

    def record_pin(pin_path, proof, *, handoff=None):
        observed["pin"] = healthy_pin(pin_path, proof, handoff=handoff)
        return observed["pin"]

    monkeypatch.setattr(ms.store_detect, "pin_fsdp_checkpoint", record_pin)
    code = ms.store_load.fresh_load.__code__
    target = _target_after_named_call(code, "setattr", latest_source=True)
    assert (target.opname, target.argval) == ("POP_TOP", None)
    primary = PublishAbort("slot publication interrupted")

    with pytest.raises(PublishAbort) as caught:
        _run_with_instruction_abort(
            code,
            target.offset,
            primary,
            lambda: store.ensure(
                "model.safetensors", None, None, fsdp_launch=True),
        )

    assert caught.value is primary
    pin = observed["pin"]
    assert store.current is not None
    assert store.current.fsdp_checkpoint_pin is pin
    assert not pin._closed
    assert pin._fd >= 0
    assert os.path.exists(pin.loader_path)
    assert store.snapshot()["cleanup_failed_slots"] == []
    assert not slab_lifetime.cleanup_pending()
    assert slab_lifetime.retained_resource_count() == 0
    assert calls.load == 1
    reused, transition = store.ensure(
        "model.safetensors", None, None, fsdp_launch=True)
    assert transition == TRANSITION_REUSE
    assert reused is store.current.active_patcher
    assert not pin._closed
    assert calls.load == 1

    store.unload_all()

    assert store.current is None
    assert store.snapshot()["cleanup_failed_slots"] == []
    assert not slab_lifetime.cleanup_pending()
    assert pin._closed
    assert not os.path.exists(pin.loader_path)
    assert fsdp_checkpoint_pin._PIN_POISON_OWNERS == []


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_slab_load_return_interrupt_recovers_exact_caller_handoff(
    rig,
    monkeypatch,
):
    from dgx_monarch.actor import slab_lifetime

    class LoadAbort(BaseException):
        pass

    reset_slab_lifetime(monkeypatch)
    store, calls = rig
    store.slab_weights = True

    def assert_base_released():
        assert calls.base_refs[0]() is None

    calls.close_guard = assert_base_released
    code = ms.store_load.fresh_load.__code__
    target = _target_after_named_call(code, "load_diffusion_model_slab")
    assert (target.opname, target.argval) == ("UNPACK_SEQUENCE", 3)
    primary = LoadAbort("slab caller handoff interrupted")

    with pytest.raises(LoadAbort) as caught:
        _run_with_instruction_abort(
            code,
            target.offset,
            primary,
            lambda: store.ensure("m.safetensors", None, STACK),
        )

    assert caught.value is primary
    assert len(calls.slabs) == 1
    slab = calls.slabs[0]
    assert slab.closed == 1
    assert calls.close_events == ["close"]
    assert calls.load == 1
    assert calls.unload_all == 1
    assert store.current is None
    assert store.snapshot()["cleanup_failed_slots"] == []
    assert not slab_lifetime.cleanup_pending()


def test_fsdp_pin_close_failure_blocks_reuse_until_exact_resource_retry(
    rig,
    monkeypatch,
    tmp_path,
):
    from dgx_monarch.actor import fsdp_checkpoint_pin, slab_lifetime

    reset_slab_lifetime(monkeypatch)
    store, calls = rig
    path = tmp_path / "model.safetensors"
    _write_one_tensor_safetensors(path, "BF16")
    monkeypatch.setattr(ms, "resolve_model_path", lambda *_args: str(path))
    monkeypatch.setattr(ms, "stock_load_preflight", lambda *_args: None)
    store.ensure("model.safetensors", None, None, fsdp_launch=True)
    pin = store.current.fsdp_checkpoint_pin
    alias = pin.loader_path
    healthy_unlink = fsdp_checkpoint_pin.os.unlink
    fail_close = True

    def fail_pin_alias_once(target):
        nonlocal fail_close
        if target == alias and fail_close:
            fail_close = False
            raise OSError("pin alias unlink interrupted")
        return healthy_unlink(target)

    monkeypatch.setattr(fsdp_checkpoint_pin.os, "unlink", fail_pin_alias_once)

    with pytest.raises(RuntimeError, match="failed to remove the pinned FSDP"):
        store.unload_all()

    assert store.current is None
    assert slab_lifetime.cleanup_pending()
    assert slab_lifetime.retained_resource_count() == 1
    assert os.path.exists(alias)
    os.fstat(pin._fd)
    with pytest.raises(RuntimeError, match="previous model load"):
        store.ensure("model.safetensors", None, None, fsdp_launch=True)
    assert calls.load == 1

    store.unload_all()

    assert not slab_lifetime.cleanup_pending()
    assert slab_lifetime.retained_resource_count() == 0
    assert not os.path.exists(alias)
    with pytest.raises(OSError):
        os.fstat(pin._fd)


def test_fsdp_pin_fd_close_is_never_retried_after_release_then_raise(
    rig,
    monkeypatch,
    tmp_path,
):
    from contextlib import suppress

    from dgx_monarch.actor import slab_lifetime

    reset_slab_lifetime(monkeypatch)
    store, _calls = rig
    path = tmp_path / "model.safetensors"
    _write_one_tensor_safetensors(path, "BF16")
    monkeypatch.setattr(ms, "resolve_model_path", lambda *_args: str(path))
    monkeypatch.setattr(ms, "stock_load_preflight", lambda *_args: None)
    store.ensure("model.safetensors", None, None, fsdp_launch=True)
    pin = store.current.fsdp_checkpoint_pin
    owned_fd = pin._fd
    healthy_owner = pin._file_owner
    healthy_close = os.close
    close_calls = []

    class CloseThenRaise:
        def close(self):
            close_calls.append(owned_fd)
            healthy_owner.close()
            raise OSError("descriptor close reported failure after release")

    pin._file_owner = CloseThenRaise()
    with pytest.raises(OSError, match="reported failure after release"):
        store.unload_all()

    assert not pin._closed
    assert pin._close_uncertain
    assert pin._fd == -1
    assert close_calls == [owned_fd]
    assert slab_lifetime.cleanup_pending()
    assert slab_lifetime.cleanup_poisoned()
    assert slab_lifetime.retained_resource_count() == 1

    unrelated_fd = os.open(path, os.O_RDONLY)
    try:
        if unrelated_fd != owned_fd:
            os.dup2(unrelated_fd, owned_fd)
            healthy_close(unrelated_fd)
            unrelated_fd = owned_fd
        with pytest.raises(RuntimeError, match="cleanup remains blocked"):
            store.unload_all()
        os.fstat(unrelated_fd)
        assert close_calls == [owned_fd]
        assert slab_lifetime.cleanup_pending()
        assert slab_lifetime.retained_resource_count() == 1
    finally:
        with suppress(OSError):
            healthy_close(unrelated_fd)


def test_fsdp_pin_fd_close_is_never_retried_after_fail_before_release(
    rig,
    monkeypatch,
    tmp_path,
):
    from dgx_monarch.actor import slab_lifetime

    reset_slab_lifetime(monkeypatch)
    store, _calls = rig
    path = tmp_path / "model.safetensors"
    _write_one_tensor_safetensors(path, "BF16")
    monkeypatch.setattr(ms, "resolve_model_path", lambda *_args: str(path))
    monkeypatch.setattr(ms, "stock_load_preflight", lambda *_args: None)
    store.ensure("model.safetensors", None, None, fsdp_launch=True)
    pin = store.current.fsdp_checkpoint_pin
    owned_fd = pin._fd
    healthy_owner = pin._file_owner
    close_calls = []

    class FailBeforeRelease:
        def close(self):
            close_calls.append(owned_fd)
            raise OSError("descriptor close failed before release")

    pin._file_owner = FailBeforeRelease()
    try:
        with pytest.raises(OSError, match="failed before release"):
            store.unload_all()

        assert not pin._closed
        assert pin._close_uncertain
        assert pin._fd == -1
        assert close_calls == [owned_fd]
        os.fstat(owned_fd)
        assert slab_lifetime.cleanup_pending()
        assert slab_lifetime.cleanup_poisoned()
        assert slab_lifetime.retained_resource_count() == 1

        with pytest.raises(RuntimeError, match="cleanup remains blocked"):
            store.unload_all()

        assert close_calls == [owned_fd]
        os.fstat(owned_fd)
        assert slab_lifetime.retained_resource_count() == 1
    finally:
        healthy_owner.close()


def test_ordinary_failed_load_cleanup_returns_false_and_preserves_load_primary(
    rig,
    monkeypatch,
):
    from dgx_monarch.actor import slab_lifetime

    store, calls = rig
    mm_stub = sys.modules["comfy.model_management"]
    healthy_unload = mm_stub.unload_all_models
    primary = RuntimeError("ordinary Comfy load failure")
    cleanup_failure = RuntimeError("ordinary global cleanup failure")
    cleanup_results = []

    def fail_load(_path, model_options=None):
        calls.load += 1
        calls.model_options.append(model_options)
        raise primary

    def fail_cleanup():
        raise cleanup_failure

    production_cleanup = slab_lifetime.cleanup_failed_load

    def observe_cleanup(*args, **kwargs):
        cleaned = production_cleanup(*args, **kwargs)
        cleanup_results.append(cleaned)
        return cleaned

    monkeypatch.setattr(
        sys.modules["comfy.sd"],
        "load_diffusion_model",
        fail_load,
    )
    monkeypatch.setattr(mm_stub, "unload_all_models", fail_cleanup)
    monkeypatch.setattr(slab_lifetime, "cleanup_failed_load", observe_cleanup)

    with pytest.raises(RuntimeError) as caught:
        store.ensure("m.safetensors", None, STACK)

    assert caught.value is primary
    assert cleanup_results == [False]
    assert any(
        "ordinary global cleanup failure" in note
        for note in getattr(primary, "__notes__", ())
    )
    assert calls.load == 1
    assert store.current is None
    assert store.snapshot()["cleanup_failed_slots"] == ["cond"]
    assert slab_lifetime.retained_resource_count() == 1
    assert slab_lifetime.cleanup_pending()

    with pytest.raises(RuntimeError, match="previous model load"):
        store.ensure("m.safetensors", None, STACK)
    assert calls.load == 1

    monkeypatch.setattr(mm_stub, "unload_all_models", healthy_unload)
    store.unload_all()
    assert not slab_lifetime.cleanup_pending()


def test_failed_load_cleanup_baseexception_retains_owned_slab(rig, monkeypatch):
    """Double failure keeps a durable owner and re-raises the load error."""
    import gc

    from dgx_monarch.actor import slab_lifetime

    class LoadAbort(BaseException):
        pass

    class CleanupAbort(BaseException):
        pass

    reset_slab_lifetime(monkeypatch)
    store, calls = rig
    store.slab_weights = True
    mm_stub = sys.modules["comfy.model_management"]
    healthy_unload = mm_stub.unload_all_models

    def assert_base_released():
        assert calls.base_refs[0]() is None

    calls.close_guard = assert_base_released

    def failing_unload():
        raise CleanupAbort("pool pressure interrupted cleanup")

    monkeypatch.setattr(mm_stub, "unload_all_models", failing_unload)

    def boom(base, quant, stack, precision):
        raise LoadAbort("adapter typed refusal")

    with pytest.raises(LoadAbort, match="typed refusal") as caught:
        store.ensure("m.safetensors", None, STACK, on_base_loaded=boom)
    slab_ref = calls.slab_refs[0]
    calls.slabs.clear()  # the fixture's list must not be what keeps the slab alive
    caught.value.__traceback__ = None
    del caught
    gc.collect()
    assert slab_lifetime.retained_count() == 0
    assert slab_lifetime.retained_resource_count() == 1
    assert slab_ref() is not None
    assert calls.close_events == []
    assert calls.base_refs[0]() is not None   # failed global unload still owns it
    assert store.current is None
    with pytest.raises(RuntimeError, match="previous model load"):
        store.ensure("m.safetensors", None, STACK)

    monkeypatch.setattr(mm_stub, "unload_all_models", healthy_unload)
    store.unload_all()
    gc.collect()
    assert calls.close_events == ["close"]
    assert slab_lifetime.retained_count() == 0
    assert slab_lifetime.retained_resource_count() == 0
    assert slab_ref() is None


@pytest.mark.parametrize("failure_phase", ["comfy_load", "injection"])
def test_non_slab_failed_load_cleanup_poison_requires_confirmed_unload_retry(
    rig,
    monkeypatch,
    failure_phase,
):
    """A failed non-slab load whose cleanup fails stays retained and poisoned until an unload succeeds."""
    from dgx_monarch.actor import slab_lifetime

    class LoadAbort(BaseException):
        pass

    class CleanupAbort(BaseException):
        pass

    reset_slab_lifetime(monkeypatch)
    store, calls = rig
    mm_stub = sys.modules["comfy.model_management"]
    healthy_unload = mm_stub.unload_all_models
    failure = LoadAbort(f"{failure_phase} aborted")
    cleanup_failure = CleanupAbort("global cleanup interrupted")
    cleanup_attempts = []

    if failure_phase == "comfy_load":
        def fail_load(_path, model_options=None):
            calls.load += 1
            calls.model_options.append(model_options)
            raise failure

        monkeypatch.setattr(sys.modules["comfy.sd"], "load_diffusion_model", fail_load)

    def fail_cleanup():
        cleanup_attempts.append("failed")
        raise cleanup_failure

    monkeypatch.setattr(mm_stub, "unload_all_models", fail_cleanup)

    def fail_injection(*_args):
        raise failure

    callback = fail_injection if failure_phase == "injection" else None
    with pytest.raises(LoadAbort) as caught:
        store.ensure(
            "m.safetensors",
            None,
            STACK,
            on_base_loaded=callback,
        )

    assert caught.value is failure
    assert store.current is None
    assert calls.load == 1
    assert cleanup_attempts == ["failed"]
    assert slab_lifetime.retained_count() == 0
    assert slab_lifetime.retained_resource_count() == 1
    assert slab_lifetime.cleanup_poisoned()
    assert store.snapshot()["retained_failed_load_resources"] == 1
    with pytest.raises(RuntimeError, match="previous model load"):
        store.ensure("m.safetensors", None, STACK)
    assert calls.load == 1

    with pytest.raises(CleanupAbort) as cleanup_caught:
        store.unload_all()
    assert cleanup_caught.value is cleanup_failure
    assert cleanup_attempts == ["failed", "failed"]
    assert slab_lifetime.cleanup_pending()

    monkeypatch.setattr(mm_stub, "unload_all_models", healthy_unload)
    store.unload_all()
    assert not slab_lifetime.cleanup_pending()


def test_failed_load_resource_stays_open_until_global_cleanup_is_confirmed(
    rig,
    monkeypatch,
):
    from dgx_monarch.actor import slab_lifetime

    class CleanupAbort(BaseException):
        pass

    class Resource:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    reset_slab_lifetime(monkeypatch)
    store, _calls = rig
    mm_stub = sys.modules["comfy.model_management"]
    healthy_unload = mm_stub.unload_all_models
    resource = Resource()

    def fail_cleanup():
        raise CleanupAbort("global cleanup interrupted")

    monkeypatch.setattr(mm_stub, "unload_all_models", fail_cleanup)
    with pytest.raises(CleanupAbort, match="global cleanup interrupted") as caught:
        slab_lifetime.cleanup_failed_load(
            None,
            RuntimeError("load failed"),
            "test failed load",
            resources=(resource,),
        )

    assert caught.value.__cause__ is not caught.value
    assert resource.closed == 0
    assert slab_lifetime.retained_resource_count() == 1
    assert store.snapshot()["retained_failed_load_resources"] == 1
    with pytest.raises(CleanupAbort, match="cleanup interrupted"):
        slab_lifetime.release_after_explicit_unload()
    assert resource.closed == 0
    assert slab_lifetime.cleanup_pending()

    monkeypatch.setattr(mm_stub, "unload_all_models", healthy_unload)
    slab_lifetime.release_after_explicit_unload()
    assert resource.closed == 1
    assert not slab_lifetime.cleanup_pending()
    assert store.snapshot()["retained_failed_load_resources"] == 0


def test_cleanup_failed_load_prepublishes_and_retains_the_whole_failed_batch(
    rig,
    monkeypatch,
):
    from dgx_monarch.actor import slab_lifetime

    class CloseAbort(BaseException):
        pass

    reset_slab_lifetime(monkeypatch)
    _store, _calls = rig
    primary = RuntimeError("model load failed")
    mm_stub = sys.modules["comfy.model_management"]
    healthy_unload = mm_stub.unload_all_models
    cleanup_snapshots = []
    preclose_snapshots = []

    class Resource:
        def __init__(self, name):
            self.name = name
            self.attempts = 0

        def close(self):
            self.attempts += 1
            preclose_snapshots.append((
                tuple(owned for owned, _error in (
                    slab_lifetime._RETAINED_FAILED_LOAD_RESOURCES
                )),
                tuple(owned for owned, _error in (
                    slab_lifetime._RETAINED_FAILED_LOAD_SLABS
                )),
                slab_lifetime.cleanup_pending(),
            ))
            raise CloseAbort(f"{self.name} close interrupted")

    first = Resource("first resource")
    second = Resource("second resource")
    slab = Resource("slab")

    def observe_global_unload():
        cleanup_snapshots.append((
            tuple(owned for owned, _error in (
                slab_lifetime._RETAINED_FAILED_LOAD_RESOURCES
            )),
            tuple(owned for owned, _error in (
                slab_lifetime._RETAINED_FAILED_LOAD_SLABS
            )),
            slab_lifetime.cleanup_poisoned(),
        ))
        healthy_unload()

    monkeypatch.setattr(mm_stub, "unload_all_models", observe_global_unload)

    with pytest.raises(CloseAbort) as caught:
        slab_lifetime.cleanup_failed_load(
            slab,
            primary,
            "batch cleanup regression",
            resources=(first, second),
        )

    assert caught.value.args == ("first resource close interrupted",)
    assert caught.value.__cause__ is not caught.value
    assert [resource.attempts for resource in (first, second, slab)] == [1, 1, 1]
    assert cleanup_snapshots == [((first, second), (slab,), True)]
    assert preclose_snapshots[0] == ((first, second), (slab,), True)
    assert slab_lifetime.cleanup_poisoned()
    assert slab_lifetime.cleanup_pending()
    assert slab_lifetime._RETAINED_FAILED_LOAD_RESOURCES == [
        (first, primary),
        (second, primary),
    ]
    assert slab_lifetime._RETAINED_FAILED_LOAD_SLABS == [(slab, primary)]


def test_cleanup_failed_load_publication_interruption_latches_process_poison(
    rig,
    monkeypatch,
):
    from dgx_monarch.actor import slab_lifetime

    class PublishAbort(BaseException):
        pass

    class Resource:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    reset_slab_lifetime(monkeypatch)
    _store, _calls = rig
    resources = (Resource(), Resource(), Resource())
    healthy_retain = slab_lifetime._retain_resource
    publications = 0

    def interrupt_second_publication(resource, error=None):
        nonlocal publications
        publications += 1
        if publications == 2:
            raise PublishAbort("ownership publication interrupted")
        healthy_retain(resource, error)

    monkeypatch.setattr(
        slab_lifetime,
        "_retain_resource",
        interrupt_second_publication,
    )
    with pytest.raises(PublishAbort, match="ownership publication interrupted"):
        slab_lifetime.cleanup_failed_load(
            None,
            RuntimeError("model load failed"),
            "publication interruption regression",
            resources=resources,
        )
    assert slab_lifetime.cleanup_poisoned()
    assert slab_lifetime.cleanup_pending()
    assert slab_lifetime.retained_resource_count() == 1
    assert all(resource.closed == 0 for resource in resources)


def test_retained_only_cleanup_failure_latches_poison(rig, monkeypatch):
    from dgx_monarch.actor import slab_lifetime

    resource = _FakeSlab()
    monkeypatch.setattr(slab_lifetime, "_RETAINED_FAILED_LOAD_SLABS", [])
    monkeypatch.setattr(
        slab_lifetime,
        "_RETAINED_FAILED_LOAD_RESOURCES",
        [(resource, None)],
    )
    monkeypatch.setattr(slab_lifetime, "_FAILED_LOAD_CLEANUP_POISONED", False)
    _store, _calls = rig
    mm_stub = sys.modules["comfy.model_management"]
    monkeypatch.setattr(
        mm_stub,
        "unload_all_models",
        lambda: (_ for _ in ()).throw(RuntimeError("global unload interrupted")),
    )

    with pytest.raises(RuntimeError, match="global unload interrupted"):
        slab_lifetime.release_after_explicit_unload()

    assert slab_lifetime.cleanup_poisoned()
    assert slab_lifetime.cleanup_pending()
    assert slab_lifetime.retained_resource_count() == 1
    assert resource.closed == 0
