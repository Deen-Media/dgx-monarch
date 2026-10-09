"""actor/failure.cleanup_on_failure (DESIGN.md section 5.2): a decorated failure
runs store.cleanup_active(), soft_empty_cache and gc.collect(). An ordinary
cleanup failure never masks the original error; a cancellation, in the body or
in cleanup, wins with exact identity. comfy.model_management is a ModuleType
stub in sys.modules (no real comfy in unit tests)."""
from __future__ import annotations

import sys
import types

import pytest

from dgx_monarch.actor.failure import cleanup_on_failure


class _OriginalFailure(Exception):
    pass


def _install_fake_comfy_model_management(monkeypatch, soft_empty_cache):
    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.soft_empty_cache = soft_empty_cache
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)


def test_cleanup_runs_full_sequence_and_reraises_original_exception(monkeypatch):
    calls = []
    _install_fake_comfy_model_management(
        monkeypatch, soft_empty_cache=lambda: calls.append("soft_empty_cache"))
    monkeypatch.setattr("dgx_monarch.actor.failure.gc.collect", lambda: calls.append("gc.collect"))

    class Worker:
        def __init__(self):
            self.store = types.SimpleNamespace(
                cleanup_active=lambda: calls.append("cleanup_active"))

        @cleanup_on_failure
        def boom(self):
            raise _OriginalFailure("original failure")

    with pytest.raises(_OriginalFailure, match="original failure"):
        Worker().boom()
    assert calls == ["cleanup_active", "soft_empty_cache", "gc.collect"]


def test_cleanup_store_failure_does_not_mask_original_exception(monkeypatch):
    calls = []
    _install_fake_comfy_model_management(
        monkeypatch, soft_empty_cache=lambda: calls.append("soft_empty_cache"))
    monkeypatch.setattr("dgx_monarch.actor.failure.gc.collect", lambda: calls.append("gc.collect"))

    def _wedged_cleanup():
        raise RuntimeError("store wedged mid-cleanup")

    class Worker:
        def __init__(self):
            self.store = types.SimpleNamespace(cleanup_active=_wedged_cleanup)

        @cleanup_on_failure
        def boom(self):
            raise _OriginalFailure("original failure")

    with pytest.raises(_OriginalFailure, match="original failure"):
        Worker().boom()
    # gc.collect still runs after the failed cleanup; soft_empty_cache, which
    # follows store.cleanup_active(), never runs.
    assert calls == ["gc.collect"]


def test_cleanup_soft_empty_cache_failure_does_not_mask_original_exception(monkeypatch):
    calls = []

    def _wedged_soft_empty_cache():
        calls.append("soft_empty_cache")
        raise RuntimeError("cuda allocator wedged")

    _install_fake_comfy_model_management(monkeypatch, soft_empty_cache=_wedged_soft_empty_cache)
    monkeypatch.setattr("dgx_monarch.actor.failure.gc.collect", lambda: calls.append("gc.collect"))

    class Worker:
        def __init__(self):
            self.store = types.SimpleNamespace(
                cleanup_active=lambda: calls.append("cleanup_active"))

        @cleanup_on_failure
        def boom(self):
            raise _OriginalFailure("original failure")

    with pytest.raises(_OriginalFailure, match="original failure"):
        Worker().boom()
    assert calls == ["cleanup_active", "soft_empty_cache", "gc.collect"]


def test_cleanup_skipped_gracefully_when_worker_has_no_store(monkeypatch):
    calls = []
    _install_fake_comfy_model_management(
        monkeypatch, soft_empty_cache=lambda: calls.append("soft_empty_cache"))
    monkeypatch.setattr("dgx_monarch.actor.failure.gc.collect", lambda: calls.append("gc.collect"))

    class Worker:
        @cleanup_on_failure
        def boom(self):
            raise _OriginalFailure("original failure")

    with pytest.raises(_OriginalFailure, match="original failure"):
        Worker().boom()
    assert calls == ["soft_empty_cache", "gc.collect"]


def test_success_path_never_touches_cleanup(monkeypatch):
    calls = []
    _install_fake_comfy_model_management(
        monkeypatch, soft_empty_cache=lambda: calls.append("soft_empty_cache"))
    monkeypatch.setattr("dgx_monarch.actor.failure.gc.collect", lambda: calls.append("gc.collect"))

    class Worker:
        def __init__(self):
            self.store = types.SimpleNamespace(
                cleanup_active=lambda: calls.append("cleanup_active"))

        @cleanup_on_failure
        def fine(self):
            return "ok"

    assert Worker().fine() == "ok"
    assert calls == []


def test_body_cancellation_still_runs_cleanup_and_remains_exact(monkeypatch):
    calls = []
    primary = KeyboardInterrupt("endpoint cancelled")
    _install_fake_comfy_model_management(
        monkeypatch, soft_empty_cache=lambda: calls.append("soft_empty_cache"))
    monkeypatch.setattr(
        "dgx_monarch.actor.failure.gc.collect",
        lambda: calls.append("gc.collect"),
    )

    class Worker:
        store = types.SimpleNamespace(
            cleanup_active=lambda: calls.append("cleanup_active"))

        @cleanup_on_failure
        def boom(self):
            raise primary

    with pytest.raises(KeyboardInterrupt) as caught:
        Worker().boom()

    assert caught.value is primary
    assert calls == ["cleanup_active", "soft_empty_cache", "gc.collect"]


def test_cleanup_cancellation_outranks_ordinary_body_failure(monkeypatch):
    calls = []
    primary = RuntimeError("endpoint failed")
    cancellation = KeyboardInterrupt("cleanup cancelled")
    _install_fake_comfy_model_management(
        monkeypatch, soft_empty_cache=lambda: calls.append("soft_empty_cache"))
    monkeypatch.setattr(
        "dgx_monarch.actor.failure.gc.collect",
        lambda: calls.append("gc.collect"),
    )

    def cancel_cleanup():
        calls.append("cleanup_active")
        raise cancellation

    class Worker:
        store = types.SimpleNamespace(cleanup_active=cancel_cleanup)

        @cleanup_on_failure
        def boom(self):
            raise primary

    with pytest.raises(KeyboardInterrupt) as caught:
        Worker().boom()

    assert caught.value is cancellation
    assert caught.value.__cause__ is primary
    assert calls == ["cleanup_active", "gc.collect"]


def test_ordinary_gc_failure_remains_secondary_to_body_failure(monkeypatch):
    primary = ValueError("endpoint failed")
    gc_error = RuntimeError("gc failed")
    _install_fake_comfy_model_management(monkeypatch, soft_empty_cache=lambda: None)
    monkeypatch.setattr(
        "dgx_monarch.actor.failure.gc.collect",
        lambda: (_ for _ in ()).throw(gc_error),
    )

    class Worker:
        store = types.SimpleNamespace(cleanup_active=lambda: None)

        @cleanup_on_failure
        def boom(self):
            raise primary

    with pytest.raises(ValueError) as caught:
        Worker().boom()

    assert caught.value is primary
    assert caught.value.__cause__ is gc_error


def test_reused_cancellation_never_becomes_its_own_cause(monkeypatch):
    shared = KeyboardInterrupt("shared cancellation")
    _install_fake_comfy_model_management(monkeypatch, soft_empty_cache=lambda: None)
    monkeypatch.setattr("dgx_monarch.actor.failure.gc.collect", lambda: None)

    class Worker:
        store = types.SimpleNamespace(
            cleanup_active=lambda: (_ for _ in ()).throw(shared))

        @cleanup_on_failure
        def boom(self):
            raise shared

    with pytest.raises(KeyboardInterrupt) as caught:
        Worker().boom()

    assert caught.value is shared
    assert caught.value.__cause__ is not shared
