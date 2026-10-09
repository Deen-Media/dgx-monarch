"""The concurrent-endpoint contract (F1, docs/DESIGN.md section 5.2): every
GPUWorker endpoint is async.

torchmonarch refuses an actor that mixes sync and async endpoints:
`ActorMesh.__init__` raises ValueError at spawn (monarch/_src/actor/actor_mesh.py,
torchmonarch 0.6.0). These tests repeat that check, so a missed conversion
fails in pytest, not at cluster bring-up.
"""
import asyncio
import inspect
import sys
import threading
import types
from types import SimpleNamespace

import pytest
from monarch._src.actor.endpoint import EndpointProperty

from dgx_monarch.actor.worker import GPUWorker


def _body(prop):
    """The user-written endpoint body.

    @concurrent_endpoint, on every long-running GPUWorker endpoint, wraps the
    method in an explicit-response-port wrapper and keeps the original under
    __wrapped__ (functools.wraps). These tests exercise the body (locking,
    ordering, validation), which the wrapper calls unchanged;
    tests/test_dispatch_contract.py covers the wrapper's dispatch end to end.
    """
    method = prop._method
    return getattr(method, "__wrapped__", method)


def _endpoints():
    out = []
    for name in dir(GPUWorker):
        attr = getattr(GPUWorker, name, None)
        if isinstance(attr, EndpointProperty):
            out.append((name, getattr(attr._method, "__wrapped__", attr._method)))
    return out


def test_worker_exposes_endpoints():
    names = {n for n, _ in _endpoints()}
    # Endpoints the driver dispatches; the test fails if one disappears.
    assert {
        "setup", "load_model", "load_uncond_model", "unload", "sample",
        "compute_sigmas", "status", "provenance_baseline", "clear_vram",
        "teardown_group", "ack_latent_handoff",
    } <= names


def test_all_endpoints_are_async():
    sync = [n for n, m in _endpoints() if not inspect.iscoroutinefunction(m)]
    assert not sync, (
        f"these endpoints are still sync; the sync/async mixing ban will "
        f"reject GPUWorker at spawn: {sync}"
    )


def test_blocking_impls_stay_sync():
    # The _impl bodies run on the GPU executor thread through run_in_executor,
    # which cannot await a coroutine, so they must stay plain functions.
    for name in ("_setup_impl", "_sample_impl", "_load_model_impl",
                 "_compute_sigmas_impl", "_provenance_baseline_impl",
                 "_clear_vram_impl", "_teardown_group_impl"):
        fn = getattr(GPUWorker, name)
        assert not inspect.iscoroutinefunction(fn), f"{name} must be a plain sync def"


def test_queued_render_consumes_prior_cancellation():
    worker = GPUWorker.__new__(GPUWorker)
    worker.rank = 0
    worker._gpu_lock = asyncio.Lock()
    worker._cancel_lock = threading.Lock()
    worker._active_cancel = None
    worker._pending_cancels = {}

    async def on_gpu(fn, *args):
        return fn(*args)

    worker._on_gpu = on_gpu
    worker._sample_impl = lambda request, port, event: event.is_set()

    async def exercise():
        result = await _body(GPUWorker.cancel_sample)(worker, "queued-id")
        assert result["cancelled"] is True
        # A cancel recorded before the sample starts, as for a render queued
        # on _gpu_lock, must reach that sample.
        assert await _body(GPUWorker.sample)(
            worker, {
                "_dgxm_render_id": "queued-id",
                "model": {"unet_name": "model.safetensors"},
            }) is True
        assert "queued-id" not in worker._pending_cancels

    asyncio.run(exercise())


@pytest.mark.parametrize(("endpoint_name", "args", "error", "match"), [
    (
        "load_model",
        ("model.safetensors", {"weight_dtype": "invalid"}),
        ValueError,
        "weight_dtype",
    ),
    (
        "load_uncond_model",
        ("model.safetensors", {"weight_dtype": "invalid"}),
        ValueError,
        "weight_dtype",
    ),
    (
        "gate_swap_cycle",
        ("model.safetensors", {"weight_dtype": "invalid"}),
        ValueError,
        "weight_dtype",
    ),
    (
        "gate_fsdp_reload_cycle",
        ("model.safetensors", {"weight_dtype": "invalid"}),
        ValueError,
        "weight_dtype",
    ),
    ("sample", ({
        "model": {
            "unet_name": "model.safetensors",
            "options": {"weight_dtype": "invalid"},
        },
    },), ValueError, "weight_dtype"),
    ("sample", ({
        "model": {"unet_name": "model.safetensors"},
        "uncond_model": {
            "unet_name": "uncond.safetensors",
            "options": {"weight_dtype": "invalid"},
        },
    },), ValueError, "weight_dtype"),
    *[
        (
            "sample",
            ({
                "model": {"unet_name": "model.safetensors"},
                "uncond_model": uncond_spec,
            },),
            TypeError,
            "sample uncond_model spec must be a mapping",
        )
        for uncond_spec in (0, False, [])
    ],
    ("compute_sigmas", ({
        "unet_name": "model.safetensors",
        "options": {"weight_dtype": "invalid"},
    }, "normal", 1), ValueError, "weight_dtype"),
])
def test_invalid_loader_input_refuses_before_gpu_dispatch_or_failure_cleanup(
    monkeypatch,
    endpoint_name,
    args,
    error,
    match,
):
    worker = GPUWorker.__new__(GPUWorker)
    worker._gpu_lock = asyncio.Lock()
    worker._cancel_lock = threading.Lock()
    worker._active_cancel = None
    worker._pending_cancels = {}
    resident = object()
    effects = []
    worker.store = SimpleNamespace(
        current=resident,
        cleanup_active=lambda: effects.append("cleanup_active"),
    )
    for name in (
        "_load_model_impl",
        "_load_uncond_model_impl",
        "_gate_swap_cycle_impl",
        "_gate_fsdp_reload_cycle_impl",
        "_sample_impl",
        "_compute_sigmas_impl",
    ):
        setattr(worker, name, lambda *_args: effects.append("impl"))

    async def on_gpu(*_args):
        effects.append("gpu_dispatch")

    worker._on_gpu = on_gpu
    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.soft_empty_cache = lambda: effects.append("soft_empty_cache")
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

    async def exercise():
        endpoint = _body(getattr(GPUWorker, endpoint_name))
        with pytest.raises(error, match=match):
            await endpoint(worker, *args)

    asyncio.run(exercise())

    assert effects == []
    assert worker.store.current is resident


def test_provenance_baseline_orders_after_gpu_work_while_status_stays_concurrent():
    worker = GPUWorker.__new__(GPUWorker)
    worker._gpu_lock = asyncio.Lock()
    entered = asyncio.Event()
    release = asyncio.Event()
    order = []

    def load_impl(*_args):
        order.append("load")
        return {"loaded": True}

    def baseline_impl(generation):
        order.append("baseline")
        return {"setup_generation": generation, "event_seq": 1}

    async def on_gpu(fn, *args):
        if fn is load_impl:
            entered.set()
            await release.wait()
        return fn(*args)

    worker._on_gpu = on_gpu
    worker._load_model_impl = load_impl
    worker._provenance_baseline_impl = baseline_impl
    worker._status_impl = lambda: {"diagnostic": "responsive"}

    async def exercise():
        load = asyncio.create_task(
            _body(GPUWorker.load_model)(worker, "model.sft"))
        await entered.wait()
        baseline = asyncio.create_task(
            _body(GPUWorker.provenance_baseline)(worker, 7))
        await asyncio.sleep(0)

        assert not baseline.done()
        assert await asyncio.wait_for(
            _body(GPUWorker.status)(worker), timeout=0.1
        ) == {"diagnostic": "responsive"}

        release.set()
        assert await load == {"loaded": True}
        assert await baseline == {"setup_generation": 7, "event_seq": 1}

    asyncio.run(exercise())
    assert order == ["load", "baseline"]
