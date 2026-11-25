"""Reuse routing, request identity and the residency flips ensure() honours."""
import logging
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.actor import model_store as ms
from dgx_monarch.actor.store_detect import LivePrecisionEvidence
from dgx_monarch.constants import TRANSITION_LOAD, TRANSITION_REUSE
from store_ensure_helpers import STACK
from store_ensure_helpers import rig as rig  # a fixture tests ask for by name.


def test_full_load_without_slab(rig):
    store, calls = rig
    _active, transition = store.ensure("m.safetensors", None, STACK)
    assert transition == TRANSITION_LOAD
    assert calls.load == 1
    assert calls.slabs == []                 # slab mode off: no slab built
    assert store.current.slab is None
    assert set(store.snapshot()) == {
        "cond",
        "uncond",
        "cleanup_failed_slots",
        "failed_load_cleanup_pending",
        "retained_failed_load_slabs",
        "retained_failed_load_resources",
    }


def test_explicit_bf16_reaches_comfy_and_keeps_distinct_store_identity(rig):
    store, calls = rig
    options = {"weight_dtype": "bf16"}
    detected = []

    _active, transition = store.ensure(
        "m.safetensors",
        options,
        STACK,
        on_base_loaded=lambda _base, kind, _stack, _precision: detected.append(kind),
    )

    assert transition == TRANSITION_LOAD
    assert calls.model_options == [{"dtype": torch.bfloat16}]
    assert store.current.base_key == (
        "m.safetensors",
        (("weight_dtype", "bf16"),),
    )
    assert store.current.request_key[-1] == "bf16"
    assert detected == ["bf16"]

    _active, transition = store.ensure("m.safetensors", options, STACK)
    assert transition == TRANSITION_REUSE
    assert calls.load == 1


def test_qwen_cache_policy_is_postload_not_a_comfy_loader_kwarg(rig, monkeypatch):
    """Comfy's loader does not carry transformer options into its patcher (nor does
    the fake), so the Qwen cache policy is set after load, never as a loader kwarg."""
    store, calls = rig
    monkeypatch.setattr(ms, "_detect_family", lambda _base: "qwen_image21")
    observed = []
    options = {"qwen_image21_cache": {"device": "cpu", "dtype": "int8"}}

    store.ensure(
        "m.safetensors", options, STACK,
        on_base_loaded=lambda base, *_args: observed.append(base.model_options),
    )

    assert calls.model_options == [{}]
    assert observed == [{"transformer_options": {
        "qwen_image21_cache": {"device": "cpu", "dtype": "int8"}}}]
    assert store.current.base_patcher.model_options == observed[0]


def test_qwen_cache_defaults_off_only_after_qwen_load(rig, monkeypatch):
    store, _calls = rig
    monkeypatch.setattr(ms, "_detect_family", lambda _base: "qwen_image21")
    store.ensure("m.safetensors", None, STACK)
    assert store.current.base_patcher.model_options["transformer_options"] == {
        "qwen_image21_cache": {"device": "off", "dtype": "default"}}


def test_qwen_postload_policy_preserves_other_patcher_options():
    from dgx_monarch.actor.qwen_image21_cache import apply_loaded_cache_policy

    patcher = SimpleNamespace(model_options={
        "loader_only": "kept",
        "transformer_options": {"existing": "kept"},
    })
    apply_loaded_cache_policy(
        patcher, "qwen_image21", {"qwen_image21_cache": {"device": "cpu", "dtype": "int4"}})
    assert patcher.model_options == {
        "loader_only": "kept",
        "transformer_options": {
            "existing": "kept",
            "qwen_image21_cache": {"device": "cpu", "dtype": "int4"},
        },
    }
    untouched = SimpleNamespace(model_options={"transformer_options": {"existing": "kept"}})
    apply_loaded_cache_policy(untouched, "flux2", None)
    assert untouched.model_options == {"transformer_options": {"existing": "kept"}}


@pytest.mark.parametrize(("value", "error"), [
    ("fp16", ValueError),
    (None, TypeError),
    (1, TypeError),
    (True, TypeError),
])
def test_invalid_weight_dtype_fails_before_worker_reads_or_loads(
    rig,
    monkeypatch,
    value,
    error,
):
    store, calls = rig
    callbacks = []
    monkeypatch.setattr(
        ms,
        "request_artifact_identity",
        lambda *_args, **_kwargs: pytest.fail("invalid dtype reached artifact reads"),
    )

    with pytest.raises(error, match="weight_dtype"):
        store.ensure(
            "m.safetensors",
            {"weight_dtype": value},
            STACK,
            on_base_loaded=lambda *_args: callbacks.append("entered"),
        )

    assert calls.load == 0
    assert callbacks == []


def test_fp16_full_load_reuses_and_hotswaps_without_quant_invalidation(
        rig, monkeypatch):
    store, _calls = rig
    store.lora_low_rss = False
    # Even if Comfy casts the loaded parameters to BF16, an FP16 artifact
    # remains FP16 for request identity and the FSDP callback.
    monkeypatch.setattr(ms, "_detect_checkpoint_kind", lambda _path, _kind: "fp16")
    observed_callback_kinds = []
    invalidated = []
    monkeypatch.setattr(
        store, "_invalidate_gpu_weights", lambda patcher: invalidated.append(patcher))

    _active, transition = store.ensure(
        "m.safetensors", None, STACK,
            on_base_loaded=lambda _base, kind, _stack, _precision: (
                observed_callback_kinds.append(kind)
            ),
    )
    assert transition == TRANSITION_LOAD
    assert store.current.quant_kind == "fp16"
    assert store.current.request_key[-1] == "fp16"
    assert observed_callback_kinds == ["fp16"]

    _active, transition = store.ensure("m.safetensors", None, STACK)
    assert transition == TRANSITION_REUSE

    replacement = [{"name": "b.safetensors", "strength": 0.5}]
    _active, transition = store.ensure("m.safetensors", None, replacement)
    assert transition == "hot-swap"
    assert store.current.request_key[-1] == "fp16"
    assert invalidated == []


def test_a_quantized_swap_between_stacks_that_patch_nothing_keeps_the_gpu_weights(
        rig, monkeypatch):
    """No weight changes, so nothing is unloaded: the model on the GPU is already right.

    A LoRA whose keys match no weight of the model leaves the base pristine.
    A strength change on it unloaded every model, copying the whole quantized
    model to the host and back (93 s on the pair for a change that alters no
    weight). A change to a stack that does patch weights still invalidates.
    """
    store, _calls = rig
    store.lora_low_rss = False
    monkeypatch.setattr(ms, "_detect_checkpoint_kind", lambda _path, _kind: "fp8")
    monkeypatch.setattr(
        ms, "_detect_live_precision",
        lambda _patcher, _kind: LivePrecisionEvidence("fp8", "uniform_or_quantized_fp8"))
    invalidated = []
    monkeypatch.setattr(
        store, "_invalidate_gpu_weights", lambda patcher: invalidated.append(patcher))
    patched = {"k.weight": ["delta"]}
    monkeypatch.setattr(
        ms.ModelStore, "_build_active",
        lambda self, base, stack: SimpleNamespace(
            tag="ACTIVE", model=base.model,
            patches=dict(patched) if stack and stack[0]["name"] == "real.safetensors" else {}))
    no_delta = [{"name": "pins.safetensors", "strength": 1.0}]

    _active, transition = store.ensure("m.safetensors", None, no_delta)
    assert transition == TRANSITION_LOAD and store.current.quant_kind == "fp8"
    _active, transition = store.ensure("m.safetensors", None, [{**no_delta[0], "strength": 1.05}])
    assert transition == "hot-swap"
    assert invalidated == []
    old = store.current.active_patcher
    _active, transition = store.ensure("m.safetensors", None, [{"name": "real.safetensors", "strength": 1.0}])
    assert transition == "hot-swap"
    assert invalidated == [old]


def test_a_quantized_change_on_a_whole_stock_resident_unpatches_without_a_host_copy(
        rig, monkeypatch):
    """On unified memory the strength change moves only the patched weights.

    The global unload copied the whole quantized model to the host and back to
    change the keys the stack patches (83 to 91 s per Worker on the pair for 40
    keys of a 12 GiB model). Now this model's clones point their offload device
    at their load device for ComfyUI's own unload, so the unpatch restores the
    backed-up keys in place; the offload devices come back, the other slot is
    not touched, and ComfyUI's next load applies the new stack.
    """
    store, _calls = rig
    swap = _classic_fp8(store, monkeypatch)
    unrelated = SimpleNamespace(model=SimpleNamespace(model="ANOTHER_MODEL"))
    swap.mm.current_loaded_models.insert(0, unrelated)
    assert _change_strength(store, swap) == "hot-swap"
    assert swap.invalidated == []
    assert swap.unloads == [("cuda:0", "cuda:0")]         # ComfyUI's unload ran with nothing to copy
    assert (swap.base.offload_device, swap.old.offload_device) == ("cpu", "cpu")
    assert swap.mm.current_loaded_models == [unrelated]
    assert "uncond" not in store._gpu_evicted


def _classic_fp8(store, monkeypatch):
    """A quantized stock resident in classic mode on unified memory, held whole by a ComfyUI entry."""
    store.lora_low_rss = False
    monkeypatch.setattr("dgx_monarch.mesh_safety.gpu_is_integrated", lambda: True)
    monkeypatch.setattr(ms, "_detect_checkpoint_kind", lambda _path, _kind: "fp8")
    monkeypatch.setattr(
        ms, "_detect_live_precision",
        lambda _patcher, _kind: LivePrecisionEvidence("fp8", "uniform_or_quantized_fp8"))
    invalidated = []
    monkeypatch.setattr(
        store, "_invalidate_gpu_weights", lambda patcher: invalidated.append(patcher))
    monkeypatch.setattr(
        ms.ModelStore, "_build_active",
        lambda self, base, stack: SimpleNamespace(
            tag="ACTIVE", model=base.model, load_device="cuda:0", offload_device="cpu",
            patches={"k.weight": ["delta"]}))
    stack = [{"name": "real.safetensors", "strength": 1.0}]
    _active, transition = store.ensure("m.safetensors", None, stack)
    assert transition == TRANSITION_LOAD
    stored = store.current
    base, old = stored.base_patcher, stored.active_patcher
    base.load_device = "cuda:0"
    unloads = []

    class _Entry:
        model = old

        def model_unload(self):
            unloads.append((old.offload_device, base.offload_device))
            return True

    mm = sys.modules["comfy.model_management"]
    mm.current_loaded_models[:] = [_Entry()]
    return SimpleNamespace(stack=stack, stored=stored, base=base, old=old, unloads=unloads,
                           invalidated=invalidated, mm=mm)


def _change_strength(store, swap) -> str:
    _active, transition = store.ensure("m.safetensors", None, [{**swap.stack[0], "strength": 0.6}])
    return transition


def _held_whole(model_name: str) -> SimpleNamespace:
    """Another slot's stock resident that ComfyUI holds whole."""
    model = SimpleNamespace(diffusion_model=model_name)
    return SimpleNamespace(
        base_patcher=SimpleNamespace(model=model, load_device="cuda:0", offload_device="cpu"),
        active_patcher=SimpleNamespace(model=model, load_device="cuda:0", offload_device="cpu"),
        slab=None, residency_rung="")


@pytest.mark.parametrize("gate", ["partly loaded", "not stock", "model shared with the other slot"])
def test_a_quantized_change_outside_the_in_place_gates_keeps_the_global_unload(rig, monkeypatch, gate):
    """Each gate of the in-place unpatch holds on its own.

    A partly loaded model is the memory one: with its offload device pinned,
    ComfyUI's unpatch would move its host part onto the device. A
    comfy-managed resident is ComfyUI's to move, and a model the other slot
    shares would be unpatched under that slot too.
    """
    from dgx_monarch.actor import store_residency

    store, _calls = rig
    swap = _classic_fp8(store, monkeypatch)
    if gate == "partly loaded":
        swap.base.model.model_lowvram = True
    elif gate == "not stock":
        swap.stored.residency_rung = store_residency.RUNG_COMFY_MANAGED
    else:
        store.uncond = SimpleNamespace(base_patcher=SimpleNamespace(model=swap.base.model),
                                       active_patcher=None, slab=None, residency_rung="")
    assert _change_strength(store, swap) == "hot-swap"
    assert swap.invalidated == [swap.old]
    assert swap.unloads == []                             # no unload ran with the offload pinned
    assert (swap.base.offload_device, swap.old.offload_device) == ("cpu", "cpu")


def test_the_in_place_unpatch_leaves_the_other_slot_alone(rig, monkeypatch):
    """The swap unloads only its own model: the other slot's patchers and records are untouched.

    It makes no claim past the swap: with smart memory off, ComfyUI's next load
    that does not use that slot still unloads it.
    """
    store, _calls = rig
    swap = _classic_fp8(store, monkeypatch)
    store.uncond = other = _held_whole("UNCOND_DIT")
    kept = SimpleNamespace(model=other.active_patcher)
    swap.mm.current_loaded_models.append(kept)
    assert _change_strength(store, swap) == "hot-swap"
    assert swap.invalidated == []
    assert kept in swap.mm.current_loaded_models
    assert (other.base_patcher.offload_device, other.active_patcher.offload_device) == ("cpu", "cpu")
    assert "uncond" not in store._gpu_evicted


def test_a_quantized_change_off_unified_memory_keeps_the_global_unload(rig, monkeypatch):
    store, _calls = rig
    store.lora_low_rss = False
    monkeypatch.setattr("dgx_monarch.mesh_safety.gpu_is_integrated", lambda: False)
    monkeypatch.setattr(ms, "_detect_checkpoint_kind", lambda _path, _kind: "fp8")
    monkeypatch.setattr(
        ms, "_detect_live_precision",
        lambda _patcher, _kind: LivePrecisionEvidence("fp8", "uniform_or_quantized_fp8"))
    invalidated = []
    monkeypatch.setattr(
        store, "_invalidate_gpu_weights", lambda patcher: invalidated.append(patcher))
    monkeypatch.setattr(
        ms.ModelStore, "_build_active",
        lambda self, base, stack: SimpleNamespace(tag="ACTIVE", model=base.model,
                                                  patches={"k.weight": ["delta"]}))
    stack = [{"name": "real.safetensors", "strength": 1.0}]
    store.ensure("m.safetensors", None, stack)
    old = store.current.active_patcher
    _active, transition = store.ensure("m.safetensors", None, [{**stack[0], "strength": 0.6}])
    assert transition == "hot-swap"
    assert invalidated == [old]


def test_reuse_routes_without_reload(rig):
    store, calls = rig
    store.ensure("m.safetensors", None, STACK)
    _active, transition = store.ensure("m.safetensors", None, STACK)
    assert transition == TRANSITION_REUSE
    assert calls.load == 1                   # no second disk load


def test_parallel_teardown_reloads_real_store_and_attention_dispatch(
        rig, monkeypatch):
    """A new process-group generation reloads both model and USP attention."""
    from dgx_monarch.actor import worker_env

    store, calls = rig
    generation = {"value": 1}
    built = []

    class BoundAttention:
        def __init__(self):
            self.generation = generation["value"]

        def __call__(self):
            if self.generation != generation["value"]:
                raise ValueError("ProcessGroup is not registered")
            return self.generation

    def make_attention(_kernel, _sync):
        impl = BoundAttention()
        built.append(impl)
        return impl

    monkeypatch.setattr("dgx_monarch.adapters.make_usp_attention", make_attention)
    dispatch = worker_env._AttentionDispatch()
    dispatch.configure("TORCH_FLASH", True)
    injected = []
    first_active, first_transition = store.ensure(
        "m.safetensors", None, STACK,
        on_base_loaded=lambda *_args: injected.append(dispatch._impl),
    )
    first_base = store.current.base_patcher

    parallel_state = types.ModuleType("xfuser.core.distributed.parallel_state")
    parallel_state.destroy_model_parallel = lambda: None
    parallel_state.destroy_distributed_environment = lambda: None
    distributed = types.ModuleType("xfuser.core.distributed")
    distributed.parallel_state = parallel_state
    core = types.ModuleType("xfuser.core")
    core.distributed = distributed
    xfuser = types.ModuleType("xfuser")
    xfuser.core = core
    monkeypatch.setitem(sys.modules, "xfuser", xfuser)
    monkeypatch.setitem(sys.modules, "xfuser.core", core)
    monkeypatch.setitem(sys.modules, "xfuser.core.distributed", distributed)
    monkeypatch.setitem(
        sys.modules, "xfuser.core.distributed.parallel_state", parallel_state)

    worker_env.teardown_parallel_state(SimpleNamespace(store=store, _attn=dispatch))
    assert store.current is None
    generation["value"] = 2
    dispatch.configure("TORCH_FLASH", True)
    second_active, second_transition = store.ensure(
        "m.safetensors", None, STACK,
        on_base_loaded=lambda *_args: injected.append(dispatch._impl),
    )

    assert (first_transition, second_transition) == (TRANSITION_LOAD, TRANSITION_LOAD)
    assert calls.load == 2
    assert first_active is not second_active
    assert first_base is not store.current.base_patcher
    assert [impl.generation for impl in injected] == [1, 2]
    with pytest.raises(ValueError, match="ProcessGroup is not registered"):
        built[0]()
    assert dispatch() == 2


def test_same_name_checkpoint_replacement_forces_reload(rig, monkeypatch):
    store, calls = rig
    signature = {"value": "model-a"}

    def identity(unet_name, lora_stack):
        return {
            "digest": signature["value"], "comfy": "commit",
            "artifacts": [{"kind": "diffusion_models", "name": unet_name,
                           "signature": signature["value"]}],
        }

    monkeypatch.setattr(ms, "request_artifact_identity", identity)
    store.ensure("m.safetensors", None, STACK)
    signature["value"] = "model-b"
    _active, transition = store.ensure("m.safetensors", None, STACK)

    assert transition == TRANSITION_LOAD
    assert calls.load == 2
    assert store.current.artifact_identity["digest"] == "model-b"


def test_same_name_lora_replacement_cannot_reuse_stale_active(rig, monkeypatch):
    store, calls = rig
    lora_signature_value = {"value": "lora-a"}

    def identity(unet_name, lora_stack):
        artifacts = [{"kind": "diffusion_models", "name": unet_name,
                      "signature": "model"}]
        artifacts.extend(
            {"kind": "loras", "name": entry["name"],
             "signature": lora_signature_value["value"]}
            for entry in lora_stack or []
        )
        return {"digest": lora_signature_value["value"], "comfy": "commit",
                "artifacts": artifacts}

    monkeypatch.setattr(ms, "request_artifact_identity", identity)
    store.ensure("m.safetensors", None, STACK)
    lora_signature_value["value"] = "lora-b"
    _active, transition = store.ensure("m.safetensors", None, STACK)

    assert transition == TRANSITION_LOAD
    assert calls.load == 2
    assert store.current.artifact_identity["digest"] == "lora-b"


def test_artifact_identity_drift_blocks_base_callback(rig, monkeypatch):
    from dgx_monarch.mesh_safety import ArtifactBindingError

    store, calls = rig
    identities = iter(("before", "after"))
    callbacks = []

    def identity(unet_name, _lora_stack):
        value = next(identities)
        return {
            "digest": value, "comfy": "commit",
            "artifacts": [{"kind": "diffusion_models", "name": unet_name,
                           "signature": value}],
        }

    monkeypatch.setattr(ms, "request_artifact_identity", identity)
    with pytest.raises(
        ArtifactBindingError,
        match="identity changed while loading base before adapter injection",
    ):
        store.ensure(
            "m.safetensors", None, STACK,
            on_base_loaded=lambda *_args: callbacks.append("entered"),
        )

    assert callbacks == []
    assert store.current is None
    assert calls.unload_all == 1


def test_artifact_identity_must_stay_stable_through_full_load(rig, monkeypatch):
    from dgx_monarch.mesh_safety import ArtifactBindingError

    store, calls = rig
    identities = iter(("before", "before", "after"))
    callbacks = []

    def identity(unet_name, _lora_stack):
        value = next(identities)
        return {
            "digest": value, "comfy": "commit",
            "artifacts": [{"kind": "diffusion_models", "name": unet_name,
                           "signature": value}],
        }

    monkeypatch.setattr(ms, "request_artifact_identity", identity)
    with pytest.raises(ArtifactBindingError, match="identity changed while loading"):
        store.ensure(
            "m.safetensors", None, STACK,
            on_base_loaded=lambda *_args: callbacks.append("entered"),
        )

    assert callbacks == ["entered"]
    assert store.current is None
    assert calls.unload_all == 1


def test_slab_mode_flip_drops_resident_models(rig):
    store, calls = rig
    store.slab_weights = True
    store.ensure("m.safetensors", None, STACK)
    slab = calls.slabs[0]
    store.set_slab_mode(False)               # flip: storage classes differ
    assert store.current is None
    assert slab.closed == 1                  # _drop closed the slab
    store.set_slab_mode(False)               # no-op flip: nothing to drop
    assert slab.closed == 1


def test_a_stock_policy_respelling_adopts_the_resident_instead_of_reloading(
    rig, caplog,
):
    """Two setup and bind cycles in one worker: graph prep resolves auto to stock,
    then the dispatch fallback respells the same residency as explicit stock.
    Exactly one load may happen; store_residency.policy_keeps_resident says why."""
    store, calls = rig
    store.set_slab_mode("auto")
    store.ensure("m.safetensors", None, STACK)
    assert calls.load == 1
    assert store.current.residency_rung == "stock_fits"
    assert store.current.slab_auto_retry is True    # the vouched-auto promotion

    with caplog.at_level(logging.INFO):
        store.set_lora_mode(False)
        store.set_slab_mode(False, reason="the normal render forced stock residency")
        _active, transition = store.ensure("m.safetensors", None, STACK)

    assert calls.load == 1
    assert transition == TRANSITION_REUSE
    # set_slab_mode clears the vouched-auto promotion; its comment says why.
    assert store.current.slab_auto_retry is False
    assert caplog.text.count("adopt resident") == 2


def test_a_slab_resident_is_still_dropped_when_the_policy_forces_stock(rig):
    store, calls = rig
    store.slab_weights = True
    store.ensure("m.safetensors", None, STACK)
    assert store.current.slab is not None
    store.set_slab_mode("auto")                     # auto can resolve to slab
    assert store.current is not None
    store.set_slab_mode(False)
    assert store.current is None                    # storage classes differ
    assert calls.slabs[0].closed == 1


def test_an_unbaked_lora_resident_is_dropped_when_low_rss_turns_on(rig):
    """A resident with an unbaked stack also holds comfy's full weight backup, the
    copy low-RSS mode removes, so turning low-RSS on drops it."""
    store, _calls = rig
    store.lora_low_rss = False
    store.ensure("m.safetensors", None, STACK)
    assert store.current.base_baked is False
    store.set_lora_mode(True)
    assert store.current is None


def test_a_baked_base_reloads_rather_than_cloning_after_a_lora_policy_flip(rig):
    store, calls = rig                              # the rig starts in low_rss
    store.ensure("m.safetensors", None, STACK)
    assert store.current.base_baked is True
    store.set_lora_mode(False)
    assert store.current is not None                # same weights either mode
    _active, transition = store.ensure(
        "m.safetensors", None, [{"name": "a.safetensors", "strength": 0.5}])
    assert transition == TRANSITION_LOAD            # never a clone of a baked base
    assert calls.load == 2
