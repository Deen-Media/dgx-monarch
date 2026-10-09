"""Production-entry regressions for actor/RDMA exception composition."""

from __future__ import annotations

import builtins
import contextlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch


class HostileDiagnostic(RuntimeError):
    def __repr__(self) -> str:
        raise KeyboardInterrupt("hostile repr executed")

    def __str__(self) -> str:
        raise KeyboardInterrupt("hostile str executed")


class DiagnosticAbort(KeyboardInterrupt):
    pass


class HostileTruthCancellation(KeyboardInterrupt):
    def __bool__(self) -> bool:
        raise AssertionError("exception truthiness must not be evaluated")


class HostileTruthOrdinary(RuntimeError):
    def __bool__(self) -> bool:
        raise AssertionError("exception truthiness must not be evaluated")


class RegistryRetryFailures(list[object]):
    def __init__(
        self, initial: list[object], errors: list[BaseException],
    ) -> None:
        super().__init__(initial)
        self.errors = errors
        self.calls = 0

    def _fail(self) -> None:
        error = self.errors[self.calls]
        self.calls += 1
        raise error

    def append(self, value: object) -> None:
        self._fail()

    def pop(self, index: int = -1) -> object:
        self._fail()


@pytest.fixture(autouse=True)
def _reset_process_roots():
    from dgx_monarch import transfer
    from dgx_monarch.actor import slab_lifetime

    transfer._RDMA_POISONED_OWNERS.clear()
    slab_lifetime._RETAINED_FAILED_LOAD_SLABS.clear()
    slab_lifetime._RETAINED_FAILED_LOAD_RESOURCES.clear()
    slab_lifetime._FAILED_LOAD_CLEANUP_POISONED = False
    slab_lifetime._PROVISIONAL_CLEAR_WHEN_EMPTY = False
    yield
    transfer._RDMA_POISONED_OWNERS.clear()
    slab_lifetime._RETAINED_FAILED_LOAD_SLABS.clear()
    slab_lifetime._RETAINED_FAILED_LOAD_RESOURCES.clear()
    slab_lifetime._FAILED_LOAD_CLEANUP_POISONED = False
    slab_lifetime._PROVISIONAL_CLEAR_WHEN_EMPTY = False


def test_transfer_error_policy_is_one_shared_exact_cancellation_selector():
    from dgx_monarch import (
        rdma_job_registry,
        rdma_receiver,
        rdma_settlement,
        transfer_utils,
    )
    from dgx_monarch.nodes import latent_outputs

    assert rdma_job_registry.prefer is transfer_utils.prefer_error
    assert rdma_receiver.prefer_error is transfer_utils.prefer_error
    assert rdma_settlement.prefer_error is transfer_utils.prefer_error
    assert latent_outputs.prefer_error is transfer_utils.prefer_error

    ordinary = RuntimeError("ordinary failure")
    cancellation = HostileTruthCancellation("operational cancellation")
    later_cancellation = SystemExit("later operational cancellation")

    selected = transfer_utils.prefer_error(
        ordinary, cancellation, "ordinary failure retained")
    assert selected is cancellation
    assert transfer_utils.prefer_error(
        selected, RuntimeError("later ordinary failure"),
        "later failure retained",
    ) is cancellation
    assert transfer_utils.prefer_error(
        selected, later_cancellation, "later cancellation retained",
    ) is cancellation


@pytest.mark.parametrize(
    ("operation", "label"),
    [
        ("register", "later RDMA scratch job registration failure"),
        ("unregister", "later RDMA scratch root retirement failure"),
    ],
)
def test_registry_retry_promotes_exact_cancellation_without_truthiness(
    monkeypatch, operation, label,
):
    from dgx_monarch import rdma_job_registry

    job = object()
    ordinary = RuntimeError("first attempt failed")
    cancellation = HostileTruthCancellation("retry cancelled")
    jobs = RegistryRetryFailures(
        [] if operation == "register" else [job],
        [ordinary, cancellation],
    )
    monkeypatch.setattr(rdma_job_registry, "_JOBS", jobs)

    with pytest.raises(HostileTruthCancellation) as caught:
        getattr(rdma_job_registry, operation)(job)

    assert caught.value is cancellation
    assert jobs.calls == 2
    assert any(
        label in note and "first attempt failed" in note
        for note in cancellation.__notes__
    )
    assert cancellation.__cause__ is not cancellation
    assert cancellation.__context__ is not cancellation


@pytest.mark.parametrize(
    ("operation", "label"),
    [
        ("register", "later RDMA scratch job registration failure"),
        ("unregister", "later RDMA scratch root retirement failure"),
    ],
)
def test_registry_retry_preserves_first_ordinary_failure_and_notes_retry(
    monkeypatch, operation, label,
):
    from dgx_monarch import rdma_job_registry

    job = object()
    first = HostileTruthOrdinary("first attempt failed")
    retry = RuntimeError("retry failed")
    jobs = RegistryRetryFailures(
        [] if operation == "register" else [job],
        [first, retry],
    )
    monkeypatch.setattr(rdma_job_registry, "_JOBS", jobs)

    with pytest.raises(HostileTruthOrdinary) as caught:
        getattr(rdma_job_registry, operation)(job)

    assert caught.value is first
    assert jobs.calls == 2
    assert any(
        label in note and "retry failed" in note
        for note in first.__notes__
    )
    assert first.__cause__ is not first
    assert first.__context__ is not first


@pytest.mark.parametrize("kind", ["resource", "slab"])
@pytest.mark.parametrize("cancellation_source", ["close", "retention"])
def test_explicit_unload_close_composes_retention_publication_exactly(
    monkeypatch, kind, cancellation_source,
):
    from dgx_monarch.actor import slab_lifetime

    close_error = (
        KeyboardInterrupt("close cancelled")
        if cancellation_source == "close"
        else RuntimeError("close failed")
    )
    retention_error = (
        RuntimeError("retention publication failed")
        if cancellation_source == "close"
        else KeyboardInterrupt("retention publication cancelled")
    )
    expected = close_error if cancellation_source == "close" else retention_error

    class Resource:
        def close(self):
            raise close_error

    resource = Resource()
    retain_name = "retain" if kind == "slab" else "_retain_resource"
    real_retain = getattr(slab_lifetime, retain_name)

    def retain_then_fail(*args):
        real_retain(*args)
        if len(args) == 2:
            raise retention_error

    monkeypatch.setattr(slab_lifetime, retain_name, retain_then_fail)
    close = (
        slab_lifetime.close_slab_or_retain_after_explicit_unload
        if kind == "slab"
        else slab_lifetime.close_or_retain_after_explicit_unload
    )

    with pytest.raises(BaseException) as caught:
        close(resource, "regression probe")

    assert caught.value is expected
    assert expected.__cause__ is not expected
    assert slab_lifetime.cleanup_pending()


def test_explicit_unload_release_attempts_every_owner_and_promotes_cancellation(
    monkeypatch,
):
    from dgx_monarch.actor import slab_lifetime

    ordinary = RuntimeError("first LIFO close failed")
    cancellation = KeyboardInterrupt("later close cancelled")

    class Resource:
        def __init__(self, error):
            self.error = error
            self.calls = 0

        def close(self):
            self.calls += 1
            raise self.error

    later = Resource(cancellation)
    first = Resource(ordinary)
    slab_lifetime._retain_resource(later)
    slab_lifetime._retain_resource(first)

    mm = types.ModuleType("comfy.model_management")
    mm.unload_all_models = lambda: None
    mm.soft_empty_cache = lambda: None
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setattr(slab_lifetime.gc, "collect", lambda: 0)

    with pytest.raises(KeyboardInterrupt) as caught:
        slab_lifetime.release_after_explicit_unload()

    assert caught.value is cancellation
    assert cancellation.__cause__ is ordinary
    assert (first.calls, later.calls) == (1, 1)
    assert slab_lifetime.cleanup_pending()


@pytest.mark.parametrize("stage", ["vram", "host", "memory"])
def test_worker_status_hostile_exception_diagnostics_are_total(
    monkeypatch, stage,
):
    from dgx_monarch import telemetry
    from dgx_monarch.actor import worker_status

    failure = HostileDiagnostic(f"{stage} failed")
    monkeypatch.setattr(worker_status, "source_manifest_sha256", lambda: "source")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1, 2))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    monkeypatch.setattr(telemetry, "host_stats", lambda: {"ok": True})

    mm = types.ModuleType("comfy.model_management")
    mm.MAX_PINNED_MEMORY = -1
    mm.NUM_STREAMS = 0
    mm.EXTRA_RESERVED_VRAM = 0
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

    worker = SimpleNamespace(
        rank=0,
        world=1,
        topology={},
        store=SimpleNamespace(snapshot=lambda: {}),
        _setup_key=object(),
        _memory_detail=lambda: {"ok": True},
    )
    if stage == "vram":
        monkeypatch.setattr(
            torch.cuda, "mem_get_info",
            lambda: (_ for _ in ()).throw(failure))
    elif stage == "host":
        monkeypatch.setattr(
            telemetry, "host_stats",
            lambda: (_ for _ in ()).throw(failure))
    else:
        worker._memory_detail = lambda: (_ for _ in ()).throw(failure)

    status = worker_status.status_impl(worker)

    assert status[f"{stage}_error"] == "<HostileDiagnostic>"


@pytest.mark.parametrize("hostile_repr", [False, True])
def test_family_memo_best_effort_survives_hostile_diagnostics(
    monkeypatch, hostile_repr,
):
    from dgx_monarch.actor import store_family

    class MemoError(OSError):
        def __repr__(self):
            if hostile_repr:
                raise DiagnosticAbort("memo repr aborted")
            return "MemoError()"

    class Logger:
        def warning(self, *_args):
            if not hostile_repr:
                raise DiagnosticAbort("memo logger aborted")

    store_family.memoize_family(
        "/unused", "flux",
        identity=lambda _path: (_ for _ in ()).throw(MemoError()),
        memo_path="/unused/family.json", lock=object(), limit=2,
        logger=Logger(),
    )


@pytest.mark.parametrize("phase", ["registration", "descriptor"])
def test_sender_drop_cancellation_outranks_ordinary_primary(
    monkeypatch, phase,
):
    from dgx_monarch import transfer
    from dgx_monarch.transfer import LatentReturn

    cancellation = KeyboardInterrupt(f"{phase} drop cancelled")
    created = []

    class DropFuture:
        def get(self, timeout=None):
            raise cancellation

    class Buffer:
        def __init__(self, tensor):
            self.tensor = tensor
            created.append(self)

        def drop(self):
            return DropFuture()

    def register(tensor):
        if phase == "registration" and created:
            raise RuntimeError("registration failed")
        return Buffer(tensor)

    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "RDMABuffer", register)
    monkeypatch.setattr(transfer, "_qp_split", lambda: 2)
    if phase == "descriptor":
        monkeypatch.setattr(
            LatentReturn, "_retain",
            lambda *_args: (_ for _ in ()).throw(
                RuntimeError("descriptor publication failed")))

    with pytest.raises(KeyboardInterrupt) as caught:
        LatentReturn("rdma", min_bytes=1).pack(
            "samples", torch.arange(8, dtype=torch.uint8))

    assert caught.value is cancellation
    assert cancellation.__cause__ is not cancellation
    assert transfer._RDMA_POISONED_OWNERS


@pytest.mark.parametrize(
    "ordering", ["ordinary-cancellation", "cancellation-ordinary", "same-cancellation"],
)
def test_drop_outcome_error_composes_with_durable_native_failure(
    monkeypatch, ordering,
):
    from dgx_monarch import transfer

    ordinary = RuntimeError("native buffer drop failed")
    cancellation = KeyboardInterrupt("drop outcome publication cancelled")
    native_error, drop_error = {
        "ordinary-cancellation": (ordinary, cancellation),
        "cancellation-ordinary": (cancellation, ordinary),
        "same-cancellation": (cancellation, cancellation),
    }[ordering]
    def release(parts, owner):
        failure = transfer._drop_failure(parts[0], native_error)
        owner["drop_outcome"] = ([failure], drop_error)
        return [failure]

    monkeypatch.setattr(transfer, "_release_rdma_parts_guarded", release)
    monkeypatch.setattr(transfer, "_read_parts_concurrent", lambda *_args: None)

    class Buffer:
        def read_into(self, *_args, **_kwargs):
            return SimpleNamespace(get=lambda **_kwargs: None)

    part = {"buffer": Buffer(), "offset": 0, "nbytes": 1}
    descriptor = {
        "kind": "rdma", "dtype": "uint8", "shape": [1], "parts": [part],
    }

    with pytest.raises(KeyboardInterrupt) as caught:
        transfer.read_latent_result(descriptor)

    assert caught.value is cancellation
    if ordering == "ordinary-cancellation":
        assert cancellation.__cause__ is ordinary
    else:
        assert isinstance(cancellation.__cause__, RuntimeError)
        assert "read completed but 1 buffer" in str(cancellation.__cause__)
    assert cancellation.__cause__ is not cancellation
    assert cancellation.__context__ is not cancellation
    assert transfer._RDMA_POISONED_OWNERS[0].parts == (part,)


def test_abort_handoff_promotes_drop_cancellation(monkeypatch):
    from dgx_monarch import transfer
    from dgx_monarch.transfer import LatentReturn

    primary = RuntimeError("result publication failed")
    cancellation = KeyboardInterrupt("handoff release cancelled")
    failure = SimpleNamespace(error=cancellation, summary="release cancelled")
    monkeypatch.setattr(transfer, "_fail_closed", lambda *_args: ([failure], None))
    handoff = {"parts": [{}], "keepalive": object(), "state": "registered"}

    with pytest.raises(KeyboardInterrupt) as caught:
        LatentReturn("message").abort_handoff(handoff, primary)

    assert caught.value is cancellation
    assert cancellation.__cause__ is primary
    assert handoff["state"] == "settled"


def test_poison_retry_preserves_first_cancellation(monkeypatch):
    from dgx_monarch import transfer

    cancellation = KeyboardInterrupt("first poison publication cancelled")
    calls = 0

    def fail_twice(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise cancellation
        raise RuntimeError("poison publication retry failed")

    failures = [object()]
    monkeypatch.setattr(transfer, "_poison_failed_drops", fail_twice)
    monkeypatch.setattr(transfer, "_unowned_failures", lambda value: value)

    assert transfer._poison_with_retry(failures, None, "probe") is cancellation
    assert calls == 2


def test_settlement_final_retry_preserves_cancellation(monkeypatch):
    from dgx_monarch import transfer

    cancellation = KeyboardInterrupt("settlement cancelled")
    calls = 0

    def fail_closed(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError(f"recovery {calls} failed")

    monkeypatch.setattr(
        transfer, "_settle_parts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(cancellation))
    monkeypatch.setattr(transfer, "_fail_closed", fail_closed)

    with pytest.raises(KeyboardInterrupt) as caught:
        transfer._settle_or_fail_closed([], None, "probe")

    assert caught.value is cancellation
    assert cancellation.__cause__ is not cancellation
    assert calls == 2


def test_best_effort_aimdo_rewire_survives_exception_and_logger_diagnostics(
    monkeypatch,
):
    from dgx_monarch.actor import comfy_dynamic

    real_import = builtins.__import__

    def hostile_import(name, *args, **kwargs):
        if name.startswith("comfy_aimdo."):
            raise HostileDiagnostic("rewire failed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", hostile_import)
    monkeypatch.setattr(
        comfy_dynamic.log, "warning",
        lambda *_args: (_ for _ in ()).throw(DiagnosticAbort("logger failed")))

    comfy_dynamic._rewire_submodule_lib(SimpleNamespace(lib=object()))


def test_prompt_server_fail_open_survives_exception_and_logger_diagnostics(
    monkeypatch,
):
    from dgx_monarch.actor import server_stub

    real_import = builtins.__import__

    def hostile_import(name, *args, **kwargs):
        if name == "server":
            raise HostileDiagnostic("server import failed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", hostile_import)
    monkeypatch.setattr(
        server_stub.log, "warning",
        lambda *_args: (_ for _ in ()).throw(DiagnosticAbort("logger failed")))

    assert server_stub.ensure_prompt_server_stub() is False


def test_optional_attention_fallback_survives_hostile_diagnostics(monkeypatch):
    from dgx_monarch import adapters
    from dgx_monarch.actor import attention_dispatch

    dispatch = attention_dispatch._AttentionDispatch()
    def implementation(*_args, **_kwargs):
        return None

    monkeypatch.setattr(adapters, "make_usp_attention", lambda *_args: implementation)
    dispatch.configure("TORCH_FLASH", True)
    monkeypatch.setattr(
        adapters, "make_usp_attention",
        lambda *_args: (_ for _ in ()).throw(HostileDiagnostic("kernel failed")))
    monkeypatch.setattr(
        attention_dispatch.log, "warning",
        lambda *_args: (_ for _ in ()).throw(DiagnosticAbort("logger failed")))

    dispatch.configure("SAGE", True)

    assert dispatch._impl is implementation
    assert dispatch._kernel == "TORCH_FLASH"


def test_unbake_capture_fallback_survives_hostile_diagnostics(monkeypatch):
    from dgx_monarch.actor import store_bake, unbake

    mm = types.ModuleType("comfy.model_management")
    loads = []
    mm.load_models_gpu = lambda models, **kwargs: loads.append((models, kwargs))
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setattr(
        unbake, "capture_unbake_record",
        lambda *_args: (_ for _ in ()).throw(HostileDiagnostic("capture failed")))
    monkeypatch.setattr(
        store_bake.log, "warning",
        lambda *_args: (_ for _ in ()).throw(DiagnosticAbort("logger failed")))
    active = SimpleNamespace(
        model=object(), patches={}, backup={}, backup_buffers={})

    assert store_bake._merge_and_free(active, "/unused") is None
    assert len(loads) == 1


def test_checkpoint_detection_fallback_survives_hostile_diagnostics(monkeypatch):
    from dgx_monarch.actor import store_detect
    from dgx_monarch.adapters import detect

    class HostileSniffError(detect.CheckpointSniffError):
        def __repr__(self):
            raise DiagnosticAbort("sniff repr failed")

    monkeypatch.setattr(
        detect, "sniff_checkpoint",
        lambda *_args: (_ for _ in ()).throw(HostileSniffError("bad header")))
    monkeypatch.setattr(
        store_detect.log, "warning",
        lambda *_args: (_ for _ in ()).throw(DiagnosticAbort("logger failed")))

    assert store_detect._detect_checkpoint_kind("model.safetensors", "bf16") == "unknown"


def test_residency_never_raise_diagnostics_remain_total(monkeypatch):
    from dgx_monarch.actor import store_residency
    from dgx_monarch.capacity_fit import StockFit

    class HostileTypeError(TypeError):
        def __repr__(self):
            raise DiagnosticAbort("measurement repr failed")

    fit = StockFit(False, True, 10, 5)
    monkeypatch.setattr(
        store_residency, "measured_tag",
        lambda *_args: (_ for _ in ()).throw(HostileTypeError("bad measurement")))
    monkeypatch.setattr(
        store_residency.log, "warning",
        lambda *_args: (_ for _ in ()).throw(DiagnosticAbort("logger failed")))

    assert store_residency._wire_tail(fit, None) == ""

    monkeypatch.setattr(
        store_residency, "memo_context",
        lambda *_args: (_ for _ in ()).throw(HostileDiagnostic("memo failed")))
    descriptor = store_residency._descriptor(
        unet_name="model.safetensors", path="/unused", fit=fit,
        model_options={}, file_identity=lambda _path: "identity",
        family_hint=None,
    )
    assert descriptor.memo_context == {}


def test_progress_diagnostics_never_kill_render(monkeypatch):
    from dgx_monarch.actor import sampling

    class Port:
        def send(self, _message):
            raise HostileDiagnostic("progress send failed")

    monkeypatch.setattr(
        sampling.log, "debug",
        lambda *_args: (_ for _ in ()).throw(DiagnosticAbort("logger failed")))
    callback, finish = sampling.make_progress_callback(Port(), 2)

    callback(0, None, None, 2)
    finish()


def test_public_pack_conditioning_preserves_conds_keyword():
    from dgx_monarch import transfer

    assert transfer.pack_conditioning(conds=[]) == []


def test_slab_dtype_fallback_survives_hostile_diagnostics(monkeypatch):
    from dgx_monarch import mesh_safety
    from dgx_monarch.actor import comfy_bridge
    from dgx_monarch.safetensors_header import UnsupportedSafetensorsDtypeError

    class HostileDtypeError(UnsupportedSafetensorsDtypeError):
        def __repr__(self):
            raise DiagnosticAbort("dtype repr failed")

    @contextlib.contextmanager
    def unsupported(*_args, **_kwargs):
        raise HostileDtypeError("unsupported dtype")
        yield

    sentinel = object()
    monkeypatch.setattr(comfy_bridge, "slab_load", unsupported)
    monkeypatch.setattr(mesh_safety, "stock_load_preflight", lambda *_args: None)
    monkeypatch.setattr(
        comfy_bridge.log, "warning",
        lambda *_args: (_ for _ in ()).throw(DiagnosticAbort("logger failed")))

    assert comfy_bridge.load_diffusion_model_slab(
        "/unused/model.safetensors", {}, lambda *_args, **_kwargs: sentinel,
    ) == (sentinel, None, True)
