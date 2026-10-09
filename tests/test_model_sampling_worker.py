"""Worker-side model-sampling patches are render-local and fail closed. The file
also covers sample cancellation, the per-cond cfg equalization skip and the RDMA
latent handoff registry."""
from __future__ import annotations

import dis
import pickle
import sys
import threading
import types

import pytest
import torch

import dgx_monarch.transfer as transfer
from dgx_monarch.actor import sampling
from dgx_monarch.actor import worker as worker_mod
from dgx_monarch.rdma_ownership import (
    READY,
    REGISTERING,
    RELEASED,
    RETIRING,
    HandoffRegistry,
)
from dgx_monarch.transfer import LatentReturn, read_latent_result


class NestedTensor:
    """Current-Comfy-shaped direct modality wrapper for the wire boundary."""

    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True

    def unbind(self):
        return self.tensors


class _SharedModel:
    def __init__(self, sampling_object):
        self.active = sampling_object
        self.backup = None


class _Patcher:
    def __init__(self, shared: _SharedModel, *, parent=None):
        self.shared = shared
        self.parent = parent
        self.object_patches = {}
        self.model_options = {}
        self.clone_count = 0
        # The sample path reads model.diffusion_model on a sharded topology.
        self.model = types.SimpleNamespace(diffusion_model=types.SimpleNamespace())

    def get_model_object(self, name):
        assert name == "model_sampling"
        if name in self.object_patches:
            return self.object_patches[name]
        if self.shared.backup is not None:
            return self.shared.backup
        return self.shared.active

    def clone(self):
        self.clone_count += 1
        clone = _Patcher(self.shared, parent=self)
        clone.object_patches = dict(self.object_patches)
        clone.model_options = dict(self.model_options)
        return clone

    def add_object_patch(self, name, value):
        self.object_patches[name] = value


def _install_sd3_stub(monkeypatch, calls):
    advanced = types.ModuleType("comfy_extras.nodes_model_advanced")

    class ModelSamplingSD3:
        def patch(self, patcher, shift):
            calls.append((patcher, shift))
            shifted = patcher.clone()
            shifted.add_object_patch("model_sampling", ("sd3", shift))
            return (shifted,)

    advanced.ModelSamplingSD3 = ModelSamplingSD3
    package = types.ModuleType("comfy_extras")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "comfy_extras", package)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_model_advanced", advanced)


def test_shift_then_default_explicitly_restores_stock_sampling(monkeypatch):
    stock = object()
    shared = _SharedModel(stock)
    resident = _Patcher(shared)
    calls = []
    _install_sd3_stub(monkeypatch, calls)

    shifted = sampling.model_sampling_render_clone(
        resident, {"kind": "sd3", "shift": 5})
    assert shifted.get_model_object("model_sampling") == ("sd3", 5.0)
    assert calls[0][0].get_model_object("model_sampling") is stock

    # Mirror Comfy leaving the render clone's object patch active on the shared
    # model while retaining the stock object in the shared patch backup.
    shared.active = shifted.get_model_object("model_sampling")
    shared.backup = stock
    default = sampling.model_sampling_render_clone(resident, None)

    assert default is not resident
    assert default.object_patches["model_sampling"] is stock
    assert resident.object_patches == {}
    assert shared.active == ("sd3", 5.0)


def test_render_clone_rejects_invalid_sampling_before_touching_patcher():
    resident = _Patcher(_SharedModel(object()))
    with pytest.raises(ValueError, match="finite number"):
        sampling.model_sampling_render_clone(
            resident, {"kind": "sd3", "shift": float("nan")})
    assert resident.clone_count == 0
    assert resident.object_patches == {}


class _Store:
    def __init__(self):
        self.patchers = {
            "cond": _Patcher(_SharedModel(object())),
            "uncond": _Patcher(_SharedModel(object())),
        }
        self.calls = []
        self.consents = []

    def ensure(self, unet_name, options, loras, *, slot, on_base_loaded,
               rescue_consent=None):
        self.calls.append((unet_name, options, loras, slot, on_base_loaded))
        self.consents.append(rescue_consent)
        return self.patchers[slot], "reuse"


def _worker(monkeypatch):
    worker = worker_mod.GPUWorker.__new__(worker_mod.GPUWorker)
    worker._setup_key = ("ready",)
    worker._setup_generation = 17
    worker.topology = {"ulysses": 1, "ring": 1, "cfg": 1}
    worker.rank = 0
    worker.store = _Store()
    worker._init_rdma_handoffs()
    worker._check_uma_reserve = lambda: None

    monkeypatch.setattr(
        worker_mod.worker_env,
        "verify_sample_artifact_authorization",
        lambda _worker, request, _fingerprint: [
            {"digest": "cond"},
            *([{"digest": "uncond"}] if request.get("uncond_model") else []),
        ],
    )
    monkeypatch.setattr(
        worker_mod.worker_env, "assert_resident_artifact_identity", lambda *_args: None)
    monkeypatch.setattr(worker_mod, "_latent_signature", lambda _samples: {"digest": "latent"})
    monkeypatch.setattr("dgx_monarch.actor.sampling._dp_info", lambda: (0, 1))

    from dgx_monarch.actor import comfy_bridge

    monkeypatch.setattr(comfy_bridge, "gpu_load_seconds_reset", lambda: 0.0)
    return worker


def _model(name, sampling_value):
    return {
        "unet_name": name,
        "options": {},
        "loras": [],
        "model_sampling": sampling_value,
    }


@pytest.mark.parametrize(
    ("cancel_slot", "with_uncond", "expected_slots"),
    [
        ("cond", False, ["cond"]),
        ("reserve", True, ["cond"]),
        ("uncond", True, ["cond", "uncond"]),
    ],
)
def test_sample_observes_cancellation_at_model_load_boundaries(
    monkeypatch, cancel_slot, with_uncond, expected_slots,
):
    worker = _worker(monkeypatch)
    cancel_event = threading.Event()
    original_ensure = worker.store.ensure

    def ensure(*args, **kwargs):
        result = original_ensure(*args, **kwargs)
        if kwargs["slot"] == cancel_slot:
            cancel_event.set()
        return result

    worker.store.ensure = ensure
    if cancel_slot == "reserve":
        worker._check_uma_reserve = cancel_event.set
    from dgx_monarch import adoption_evidence

    monkeypatch.setattr(
        adoption_evidence,
        "build_worker_evidence",
        lambda *_args, **_kwargs: pytest.fail(
            "cancelled load reached pre-denoise evidence construction"),
    )
    request = {"kind": "ksampler", "model": _model("primary", None)}
    if with_uncond:
        request["uncond_model"] = _model("negative", None)

    with pytest.raises(sampling.RenderCancelledError) as caught:
        worker_mod.GPUWorker._sample_impl.__wrapped__(
            worker, request, cancel_event=cancel_event)

    assert "before denoising" in str(caught.value)
    assert [call[3] for call in worker.store.calls] == expected_slots


def test_sample_observes_cancellation_after_conditioning_preprocess(monkeypatch):
    worker = _worker(monkeypatch)
    worker.topology["cfg"] = 2
    worker.store.patchers["cond"].model = types.SimpleNamespace(
        diffusion_model=types.SimpleNamespace())
    cancel_event = threading.Event()
    from dgx_monarch import adapters, adoption_evidence

    monkeypatch.setattr(
        adoption_evidence, "build_worker_evidence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(adapters, "get_adapter", lambda _model: object())

    def equalize(_adapter, positive, negative, _latent):
        cancel_event.set()
        return positive, negative

    monkeypatch.setattr(worker_mod, "equalize_cond_lengths", equalize)
    monkeypatch.setattr(
        worker_mod,
        "run_ksampler",
        lambda *_args, **_kwargs: pytest.fail(
            "cancelled preprocessing reached the denoise loop"),
    )
    request = {
        "kind": "ksampler",
        "model": _model("primary", None),
        "latent": {"samples": object()},
        "positive": object(),
        "negative": object(),
    }

    with pytest.raises(sampling.RenderCancelledError) as caught:
        worker_mod.GPUWorker._sample_impl.__wrapped__(
            worker, request, cancel_event=cancel_event)

    assert "before denoising" in str(caught.value)


def _equalization_calls(monkeypatch, constant):
    """Run one cfg2 request and report whether the cond equalizer was called."""
    worker = _worker(monkeypatch)
    worker.topology["cfg"] = 2
    worker.store.patchers["cond"].model = types.SimpleNamespace(
        diffusion_model=types.SimpleNamespace())
    from dgx_monarch import adapters, adoption_evidence

    monkeypatch.setattr(
        adoption_evidence, "build_worker_evidence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(adapters, "get_adapter", lambda _model: types.SimpleNamespace(
        family="toy", cfg_cond_padding="pad", cfg_parallel_supported=True,
        cfg_batch_constant=constant))

    calls = []

    def equalize(_adapter, positive, negative, _latent):
        calls.append((positive, negative))
        return positive, negative

    monkeypatch.setattr(worker_mod, "equalize_cond_lengths", equalize)

    def stop(*_args, **_kwargs):
        raise RuntimeError("the denoise loop is out of scope here")

    monkeypatch.setattr(worker_mod, "run_ksampler", stop)
    request = {
        "kind": "ksampler",
        "model": _model("primary", None),
        "latent": {"samples": object()},
        "positive": object(),
        "negative": object(),
    }
    with pytest.raises(RuntimeError, match="out of scope"):
        worker_mod.GPUWorker._sample_impl.__wrapped__(
            worker, request, cancel_event=threading.Event())
    return calls


def test_a_per_cond_dispatch_family_reaches_the_sampler_unpadded(monkeypatch):
    """Padding never made a per-cond constant fold, and under per-cond dispatch
    it would hand exactly one rank a padded cond: that rank's forward can refuse
    where the other rank's returns, leaving the survivor alone in an
    all-gather. The skip reads a static family attribute, so every rank takes
    it and the two ranks still agree."""
    assert _equalization_calls(monkeypatch, "num_tokens") == []
    # A family without the constant still needs its two conds to fold, so it
    # keeps the pad rule it was measured on.
    assert len(_equalization_calls(monkeypatch, None)) == 1


def test_ksampler_and_custom_receive_primary_and_uncond_render_clones(monkeypatch):
    worker = _worker(monkeypatch)
    clones = []
    sampler_calls = []

    def render_clone(patcher, value):
        clone = types.SimpleNamespace(source=patcher, model_sampling=value)
        clones.append(clone)
        return clone

    def run_ksampler(patcher, request, progress_port, *, cancel_event, dp_cond_exempt_keys=frozenset()):
        sampler_calls.append(("ksampler", patcher, None))
        return torch.zeros(1), None

    def run_custom(
        patcher, request, uncond_patcher, progress_port, *, cancel_event,
        dp_cond_exempt_keys=frozenset(),
    ):
        sampler_calls.append(("custom", patcher, uncond_patcher))
        return torch.zeros(1), None

    monkeypatch.setattr(worker_mod, "model_sampling_render_clone", render_clone)
    monkeypatch.setattr(worker_mod, "run_ksampler", run_ksampler)
    monkeypatch.setattr(worker_mod, "run_custom", run_custom)

    ksampler_request = {
        "kind": "ksampler",
        "model": _model("primary", {"kind": "sd3", "shift": 3}),
    }
    worker_mod.GPUWorker._sample_impl.__wrapped__(worker, ksampler_request)

    custom_request = {
        "kind": "custom",
        "model": _model("primary", {"kind": "sd3", "shift": 5}),
        "uncond_model": _model("negative", None),
    }
    worker_mod.GPUWorker._sample_impl.__wrapped__(worker, custom_request)

    assert sampler_calls[0] == ("ksampler", clones[0], None)
    assert sampler_calls[1] == ("custom", clones[1], clones[2])
    assert clones[0].model_sampling == {"kind": "sd3", "shift": 3.0}
    assert clones[1].model_sampling == {"kind": "sd3", "shift": 5.0}
    assert clones[2].model_sampling is None


def test_worker_keeps_packed_primary_and_denoised_on_message_transport(monkeypatch):
    worker = _worker(monkeypatch)
    worker._latent_return = LatentReturn("rdma", min_bytes=1)
    samples = NestedTensor((
        torch.arange(4, dtype=torch.float32).reshape(1, 2, 2),
        torch.arange(3, dtype=torch.float32).reshape(1, 3),
    ))
    denoised = NestedTensor(tuple(part + 10 for part in samples.unbind()))

    monkeypatch.setattr(
        worker_mod, "model_sampling_render_clone", lambda patcher, _value: patcher
    )
    monkeypatch.setattr(
        worker_mod,
        "run_custom",
        lambda *_args, **_kwargs: (
            samples,
            {"samples": samples, "denoised": denoised},
        ),
    )
    request = {
        "kind": "custom",
        "model": _model("primary", None),
        "render_seq": 7,
        "pipeline_depth": 3,
    }

    result = worker_mod.GPUWorker._sample_impl.__wrapped__(worker, request)
    # Trusted fixture: simulate only the Monarch actor pickle boundary.
    wire_result = pickle.loads(pickle.dumps(result))  # noqa: S301
    restored = read_latent_result(wire_result["latent"])

    assert result["latent"]["kind"] == "message"
    assert type(restored) is NestedTensor
    assert [tuple(part.shape) for part in restored.unbind()] == [(1, 2, 2), (1, 3)]
    wire_denoised = wire_result["latent_extra"]["denoised"]
    assert type(wire_denoised) is NestedTensor
    assert [tuple(part.shape) for part in wire_denoised.unbind()] == [(1, 2, 2), (1, 3)]
    assert worker._latent_return._keepalive == {}


class _Buffer:
    def __init__(self, data):
        self.data_ptr = data.data_ptr()
        self.dropped = 0

    def drop(self):
        self.dropped += 1
        raise AssertionError("worker ACK must never invoke native drop")


def _rdma_worker(monkeypatch, *, depth=1):
    worker = _worker(monkeypatch)
    worker._latent_return = LatentReturn("rdma", min_bytes=1)
    samples = torch.arange(24, dtype=torch.uint8)
    created = []

    def register(data):
        # The durable attempt must precede native construction.
        assert any(
            entry["_registry_state"] == REGISTERING
            for entry in worker._latent_handoffs._entries.values()
        )
        buffer = _Buffer(data)
        created.append(buffer)
        return buffer

    monkeypatch.setattr(
        worker_mod, "model_sampling_render_clone", lambda patcher, _value: patcher)
    monkeypatch.setattr(
        worker_mod, "run_custom",
        lambda *_args, **_kwargs: (samples, {"samples": samples}))
    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "RDMABuffer", register)
    monkeypatch.setattr(transfer, "_RDMA_POISONED_OWNERS", [])
    monkeypatch.setenv("DGXM_RDMA_QP_SPLIT", "1")
    request = {
        "kind": "custom",
        "model": _model("primary", None),
        "render_seq": 7,
        "pipeline_depth": depth,
        "_dgxm_render_id": "duplicate-render-id",
    }
    return worker, samples, created, request


def _call_after(code, attribute, *, last=False):
    instructions = list(dis.get_instructions(code))
    loads = [
        index for index, instruction in enumerate(instructions)
        if instruction.opname == "LOAD_ATTR" and instruction.argval == attribute
    ]
    start = loads[-1 if last else 0]
    call = next(
        index for index in range(start, len(instructions))
        if instructions[index].opname == "CALL"
    )
    return instructions[call + 1].offset


def _interrupt_instruction(code, target, operation, primary):
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6)
                   if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target:
            monitoring.set_local_events(tool_id, code, 0)
            raise primary

    monitoring.use_tool_id(tool_id, "dgxm-worker-handoff-test")
    monitoring.register_callback(tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(tool_id, code, monitoring.events.INSTRUCTION)
    caught = None
    try:
        operation()
    except BaseException as exc:
        caught = exc
    finally:
        monitoring.set_local_events(tool_id, code, 0)
        monitoring.register_callback(tool_id, monitoring.events.INSTRUCTION, None)
        monitoring.free_tool_id(tool_id)
    assert caught is primary


def test_worker_rdma_ack_is_generation_bound_and_idempotent(monkeypatch):
    worker, samples, created, request = _rdma_worker(monkeypatch)
    result = worker_mod.GPUWorker._sample_impl.__wrapped__(worker, request)
    descriptor = result["latent"]
    token = descriptor["owner_token"]

    assert descriptor["kind"] == "rdma"
    assert descriptor["setup_generation"] == 17
    assert len(token) == 32
    int(token, 16)
    handoff = worker._latent_handoffs.get(17, token)
    assert handoff is not None and handoff["_registry_state"] == READY
    assert handoff["parts"] is descriptor["parts"]
    assert handoff["parts"][0]["buffer"] is created[0]
    assert handoff["keepalive"].data_ptr() == samples.data_ptr()
    assert worker._latent_return._keepalive == {}

    ack = worker._ack_latent_handoff_impl
    assert ack(18, token)["status"] == "unknown"
    assert ack(17, "0" * 32)["status"] == "unknown"
    assert handoff["_registry_state"] == READY

    # Setup may have installed a different LatentReturn before a delayed ACK.
    worker._setup_generation = 18
    worker._latent_return = types.SimpleNamespace(
        acknowledge_handoff=lambda _handoff: pytest.fail("used current return"))
    parts = handoff["parts"]
    assert ack(17, token)["status"] == "released"
    assert handoff["_registry_state"] == RELEASED
    assert parts == [] and handoff["keepalive"] is None
    assert created[0].dropped == 0
    assert ack(17, token)["status"] == "already_released"
    assert created[0].dropped == 0


def test_duplicate_render_id_gets_independent_random_tokens(monkeypatch):
    worker, _samples, created, request = _rdma_worker(monkeypatch, depth=2)
    first = worker_mod.GPUWorker._sample_impl.__wrapped__(worker, dict(request))
    second = worker_mod.GPUWorker._sample_impl.__wrapped__(worker, dict(request))
    first_token = first["latent"]["owner_token"]
    second_token = second["latent"]["owner_token"]

    assert first_token != second_token
    assert worker._latent_handoffs.live_count == 2
    assert worker._latent_handoffs.get(17, first_token) is not None
    assert worker._latent_handoffs.get(17, second_token) is not None
    assert len(created) == 2


def test_full_handoff_registry_falls_back_without_evicting(monkeypatch):
    worker, _samples, created, request = _rdma_worker(monkeypatch, depth=1)
    first = worker_mod.GPUWorker._sample_impl.__wrapped__(worker, dict(request))
    token = first["latent"]["owner_token"]
    owner = worker._latent_handoffs.get(17, token)

    second = worker_mod.GPUWorker._sample_impl.__wrapped__(worker, dict(request))
    assert second["latent"]["kind"] == "message"
    assert worker._latent_handoffs.live_count == 1
    assert worker._latent_handoffs.get(17, token) is owner
    assert owner["parts"][0]["buffer"] is created[0]
    assert len(created) == 1


def test_lost_result_retains_backing_until_actor_recycle(monkeypatch):
    worker, samples, _created, request = _rdma_worker(monkeypatch)
    result = worker_mod.GPUWorker._sample_impl.__wrapped__(worker, request)
    token = result["latent"]["owner_token"]
    del result

    handoff = worker._latent_handoffs.get(17, token)
    assert handoff is not None and handoff["_registry_state"] == READY
    assert handoff["keepalive"].data_ptr() == samples.data_ptr()
    assert worker._latent_handoffs.live_count == 1


def test_ambiguous_registration_refuses_and_remains_owned_until_recycle(monkeypatch):
    worker, samples, _created, request = _rdma_worker(monkeypatch)
    monkeypatch.setattr(
        transfer, "RDMABuffer",
        lambda _data: (_ for _ in ()).throw(RuntimeError("registration refused")),
    )

    with pytest.raises(RuntimeError, match="Attached mesh reset required"):
        worker_mod.GPUWorker._sample_impl.__wrapped__(worker, request)
    assert worker._latent_handoffs.live_count == 1
    handoff = next(iter(worker._latent_handoffs._entries.values()))
    assert handoff["state"] == "settled"
    assert handoff["parts"] and handoff["keepalive"].data_ptr() == samples.data_ptr()
    assert worker._latent_handoffs.reconcile(handoff) is False
    assert transfer._RDMA_POISONED_OWNERS[0].phase == "RDMA buffer construction"


def test_handoff_registry_rejects_double_publish_without_corrupting_the_owner():
    registry = HandoffRegistry()
    handoff = {}
    registry.publish(handoff, 9, 1)
    token = handoff["token"]
    part = {"buffer": object()}
    keepalive = object()
    handoff.update(parts=[part], keepalive=keepalive)
    registry.mark_ready(handoff)

    with pytest.raises(RuntimeError, match="already registry-owned"):
        registry.publish(handoff, 10, 2)

    assert registry.get(9, token) is handoff
    assert handoff["setup_generation"] == 9 and handoff["token"] == token
    assert handoff["_registry_state"] == READY
    assert handoff["parts"] == [part] and handoff["keepalive"] is keepalive
    assert registry.live_count == 1


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_publish_call_return_boundary_is_registry_owned():
    class StopNow(BaseException):
        pass

    registry = HandoffRegistry()
    handoff = {}

    def publish_call():
        outcome = registry.publish(handoff, 9, 1)
        return outcome

    code = publish_call.__code__
    target = _call_after(code, "publish")
    primary = StopNow("publish return interrupted")
    _interrupt_instruction(code, target, publish_call, primary)

    assert registry.owns(handoff)
    assert handoff["_registry_state"] == REGISTERING
    assert not handoff["parts"] and handoff["keepalive"] is None
    assert registry.reconcile(handoff)
    assert registry.live_count == 0


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize("boundary", ["pack return", "worker return", "decorator return"])
def test_worker_return_boundaries_retain_durable_ownership(monkeypatch, boundary):
    class StopNow(BaseException):
        pass

    worker, samples, created, request = _rdma_worker(monkeypatch)
    if boundary == "pack return":
        code = worker_mod.GPUWorker._pack_latent_result.__code__
        target = _call_after(code, "pack", last=True)
        expected_state = REGISTERING
    elif boundary == "worker return":
        code = worker_mod.sample_protocol.run_sample.__code__
        target = _call_after(code, "_pack_latent_result")
        expected_state = READY
    else:
        code = worker_mod.GPUWorker._sample_impl.__code__
        target = next(
            instruction.offset for instruction in dis.get_instructions(code)
            if instruction.opname == "RETURN_VALUE")
        expected_state = READY
    primary = StopNow(f"{boundary} interrupted")
    _interrupt_instruction(code, target, lambda: worker._sample_impl(request), primary)

    assert len(created) == 1 and created[0].dropped == 0
    assert transfer._RDMA_POISONED_OWNERS == []
    assert worker._latent_handoffs.live_count == 1
    handoff = next(iter(worker._latent_handoffs._entries.values()))
    assert handoff["_registry_state"] == expected_state
    assert handoff["parts"][0]["buffer"] is created[0]
    assert handoff["keepalive"].data_ptr() == samples.data_ptr()


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_handoff_reconcile_failure_never_masks_publication_interruption(monkeypatch):
    class StopNow(BaseException):
        pass

    worker, _samples, created, request = _rdma_worker(monkeypatch)
    registry = worker._latent_handoffs
    original_reconcile = registry.reconcile

    def fail_reconcile(handoff):
        assert original_reconcile(handoff) is False
        raise RuntimeError("reconcile interrupted")

    monkeypatch.setattr(registry, "reconcile", fail_reconcile)
    code = worker_mod.GPUWorker._pack_latent_result.__code__
    target = _call_after(code, "pack", last=True)
    primary = StopNow("descriptor publication interrupted")
    _interrupt_instruction(code, target, lambda: worker._sample_impl(request), primary)

    handoff = next(iter(registry._entries.values()))
    assert registry.live_count == 1
    assert handoff["parts"][0]["buffer"] is created[0]
    assert any("reconciliation interrupted" in note for note in primary.__notes__)


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize("boundary", ["record return", "released store", "reply return"])
def test_ack_instruction_boundaries_remain_retryable(boundary):
    class StopNow(BaseException):
        pass

    class CountingList(list):
        clears = 0

        def clear(self):
            self.clears += 1
            super().clear()

    registry = HandoffRegistry()
    handoff = {}
    registry.publish(handoff, 3, 1)
    token = handoff["token"]
    handoff["parts"] = CountingList([object()])
    handoff["keepalive"] = object()
    registry.mark_ready(handoff)
    code = HandoffRegistry.acknowledge.__code__
    instructions = list(dis.get_instructions(code))
    if boundary == "record return":
        target = _call_after(code, "_record_tombstone_locked", last=True)
    elif boundary == "released store":
        released = [
            index for index, instruction in enumerate(instructions)
            if instruction.opname == "LOAD_GLOBAL" and instruction.argval == RELEASED
        ][-1]
        store = next(
            index for index in range(released, len(instructions))
            if instructions[index].opname == "STORE_SUBSCR"
        )
        target = instructions[store + 1].offset
    else:
        target = next(
            instruction.offset for instruction in instructions
            if instruction.opname == "RETURN_CONST" and instruction.argval == "released"
        )
    primary = StopNow(f"ACK {boundary} interrupted")
    _interrupt_instruction(
        code, target, lambda: registry.acknowledge(3, token), primary)

    retry = registry.acknowledge(3, token)
    expected = "released" if boundary == "record return" else "already_released"
    assert retry == expected
    assert handoff["_registry_state"] == RELEASED
    assert handoff["parts"] == [] and handoff["keepalive"] is None
    assert handoff["parts"].clears == 1
    assert registry.tombstone_count == 1


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize("operation", ["pop", "popleft", "append"])
def test_tombstone_bound_repairs_interrupted_pruning(operation):
    class StopNow(BaseException):
        pass

    registry = HandoffRegistry(max_tombstones=1)
    old = {}
    registry.publish(old, 1, 1)
    registry.mark_ready(old)
    assert registry.acknowledge(1, old["token"]) == "released"

    current = {}
    registry.publish(current, 1, 1)
    registry.mark_ready(current)
    code = HandoffRegistry._record_tombstone_locked.__code__
    target = _call_after(code, operation)
    primary = StopNow(f"tombstone {operation} interrupted")
    _interrupt_instruction(
        code,
        target,
        lambda: registry.acknowledge(1, current["token"]),
        primary,
    )

    assert current["_registry_state"] == RETIRING
    assert registry.acknowledge(1, current["token"]) == "released"
    assert current["_registry_state"] == RELEASED
    assert registry.tombstone_count <= 1
    assert len(registry) <= 1


def test_sample_revalidates_before_model_residency(monkeypatch):
    worker = _worker(monkeypatch)
    request = {
        "kind": "ksampler",
        "model": _model("primary", {"kind": "sd3", "shift": True}),
    }
    with pytest.raises(ValueError, match="finite number"):
        worker_mod.GPUWorker._sample_impl.__wrapped__(worker, request)
    assert worker.store.calls == []


def test_compute_sigmas_uses_render_clone_and_revalidates_first(monkeypatch):
    worker = _worker(monkeypatch)
    observed = {}
    sampling_object = object()

    def render_clone(patcher, value):
        observed["patcher"] = patcher
        observed["value"] = value
        return types.SimpleNamespace(
            get_model_object=lambda name: sampling_object if name == "model_sampling" else None)

    samplers = types.ModuleType("comfy.samplers")

    def calculate_sigmas(model_sampling, scheduler, steps):
        observed["calculate"] = (model_sampling, scheduler, steps)
        return torch.arange(steps + 1, dtype=torch.float32)

    samplers.calculate_sigmas = calculate_sigmas
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy.samplers = samplers
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.samplers", samplers)
    monkeypatch.setattr(worker_mod, "model_sampling_render_clone", render_clone)

    spec = _model("primary", {"kind": "sd3", "shift": 5})
    result = worker_mod.GPUWorker._compute_sigmas_impl.__wrapped__(
        worker, spec, "normal", 2, 0.5)
    assert observed["patcher"] is worker.store.patchers["cond"]
    assert observed["value"] == {"kind": "sd3", "shift": 5.0}
    assert observed["calculate"] == (sampling_object, "normal", 4)
    assert torch.equal(result, torch.tensor([2.0, 3.0, 4.0]))

    worker.store.calls.clear()
    spec["model_sampling"] = {"kind": "sd3", "shift": 101}
    with pytest.raises(ValueError, match="finite number"):
        worker_mod.GPUWorker._compute_sigmas_impl.__wrapped__(
            worker, spec, "normal", 2)
    assert worker.store.calls == []
