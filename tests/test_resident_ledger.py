"""The resident-weights ledger: what it declares to comfy, what it leaves alone,
and how a drop unloads without a host copy.

The stand-in patcher below reproduces the one comfy behaviour the ledger relies
on, ``ModelPatcher.partially_load``'s early return when the model is already
fully loaded. The resident ledger seam of tests/canary/comfy_seam_contracts.py
checks that behaviour against the installed comfy.
"""
import logging

import pytest

from dgx_monarch.actor import resident_ledger
from dgx_monarch.actor.model_store import ModelStore
from store_ensure_helpers import STACK
from store_ensure_helpers import rig as rig  # a fixture tests ask for by name.

SIZE = 32 * 2**30


class StandInModel:
    """The four fields comfy's loader decides a load by."""

    def __init__(self):
        self.model_loaded_weight_memory = 0
        self.model_lowvram = False
        self.device = "cpu"
        self.current_weight_patches_uuid = None


class StandInPatcher:
    """A ModelPatcher stand-in that reproduces partially_load's early return."""

    def __init__(self, size=SIZE):
        self.model = StandInModel()
        self.load_device = "cuda:0"
        self.offload_device = "cuda:0"
        self.patches = {}
        self.weight_wrapper_patches = {}
        self.patches_uuid = "stack-0"
        self.size = size
        self.loads = []

    def model_size(self):
        return self.size

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        self.loads.append((device_to, extra_memory))
        model = self.model
        if not model.model_lowvram and model.model_loaded_weight_memory > 0:
            return 0
        before = model.model_loaded_weight_memory
        model.model_loaded_weight_memory = self.size
        model.model_lowvram = False
        model.device = device_to
        model.current_weight_patches_uuid = self.patches_uuid
        return self.size - before


def test_declare_full_loads_the_patcher_and_writes_comfys_ledger():
    patcher = StandInPatcher()

    assert resident_ledger.declare(patcher, "slab") is None

    assert patcher.loads == [("cuda:0", resident_ledger.FULL_LOAD_EXTRA)]
    assert patcher.model.model_loaded_weight_memory == SIZE
    assert patcher.model.model_lowvram is False
    assert patcher.model.device == "cuda:0"
    assert patcher.model.current_weight_patches_uuid == "stack-0"


class _Recorder(logging.Handler):
    """Collect this module's own journal lines in the order they are written."""

    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    def emit(self, record):
        self.sink.append(record.getMessage())


def test_the_ledger_line_comes_after_comfys_own_load_line():
    """The journal order an operator reads at load time.

    Comfy writes `loaded completely` from inside the load, so comfy's line
    lands first and this pack's declared line lands second.
    docs/TROUBLESHOOTING.md #79 tells an operator to read them in that order.
    """
    journal: list[str] = []

    class LoggingPatcher(StandInPatcher):
        def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
            journal.append("loaded completely; 33280.00 MB loaded, full load: True")
            return super().partially_load(device_to, extra_memory, force_patch_weights)

    handler = _Recorder(journal)
    resident_ledger.log.addHandler(handler)
    try:
        assert resident_ledger.declare(LoggingPatcher(), "slab") is None
    finally:
        resident_ledger.log.removeHandler(handler)

    assert len(journal) == 2
    assert journal[0].startswith("loaded completely;")
    assert journal[1].startswith("resident ledger: declared 32.00 GiB of slab weights")


def test_a_declared_model_is_not_declared_twice():
    patcher = StandInPatcher()
    resident_ledger.declare(patcher, "slab")

    assert resident_ledger.declare(patcher, "slab") == (
        "comfy's ledger already counts these weights")
    assert len(patcher.loads) == 1


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda p: setattr(p, "model", None), "the patcher holds no model"),
        (lambda p: setattr(p, "load_device", None),
         "the patcher names no load device"),
        (lambda p: p.patches.update({"diffusion_model.w": ["patch"]}),
         "weight patches are still pending"),
        (lambda p: p.weight_wrapper_patches.update({"diffusion_model.w": ["fn"]}),
         "weight wrapper patches are still pending"),
        (lambda p: setattr(p.model, "model_lowvram", True),
         "comfy already holds part of these weights"),
        (lambda p: setattr(p.model, "model_loaded_weight_memory", 1 << 30),
         "comfy's ledger already counts these weights"),
    ],
)
def test_declare_leaves_a_patcher_alone_and_says_why(mutate, reason):
    patcher = StandInPatcher()
    mutate(patcher)

    assert resident_ledger.declare(patcher, "slab") == reason
    assert patcher.loads == []


def test_a_pending_lora_stack_keeps_comfys_own_sample_time_load():
    """The bake's model-sized backup stays out of the build's peak."""
    patcher = StandInPatcher()
    patcher.patches["diffusion_model.blocks.0.w"] = ["one patch"]

    assert resident_ledger.declare(patcher, "slab") is not None
    assert patcher.model.model_loaded_weight_memory == 0
    assert patcher.model.device == "cpu"


def _standin_active(monkeypatch):
    patcher = StandInPatcher()
    monkeypatch.setattr(
        ModelStore, "_build_active", lambda self, base, stack: patcher)
    return patcher


def test_a_slab_load_declares_its_residency(rig, monkeypatch):
    store, _calls = rig
    store.slab_weights = True
    active = _standin_active(monkeypatch)

    store.ensure("m.safetensors", None, STACK)

    assert active.loads == [("cuda:0", resident_ledger.FULL_LOAD_EXTRA)]
    assert active.model.model_loaded_weight_memory == SIZE


def test_a_sharded_load_declares_its_residency(rig, monkeypatch):
    store, _calls = rig
    store.slab_weights = False
    active = _standin_active(monkeypatch)

    def shard(base, quant_kind, lora_stack, precision_evidence):
        base._dgxm_fsdp = True

    store.ensure("m.safetensors", None, None, on_base_loaded=shard)

    assert active.loads == [("cuda:0", resident_ledger.FULL_LOAD_EXTRA)]


def test_a_stock_load_declares_its_residency(rig, monkeypatch):
    """A stock resident is placed at load time, as slab and FSDP residents are,
    so comfy's sample-time load prices only the working set. Above world 1, a
    rank that partial-loads at sample time meets the partial-load guard
    (2026-10-01)."""
    store, _calls = rig
    store.slab_weights = False
    active = _standin_active(monkeypatch)

    store.ensure("m.safetensors", None, STACK)

    assert active.loads == [("cuda:0", resident_ledger.FULL_LOAD_EXTRA)]
    assert active.model.model_loaded_weight_memory == SIZE


def test_only_a_stock_load_declares_its_residency_not_comfy_managed(rig, monkeypatch):
    """The same rule, pinned at the call site rather than the side effect above:
    a fresh stock load declares residency "stock", and a comfy-managed decision
    declares nothing because ComfyUI's own DynamicVRAM already prices and
    places that load. ``gpu_is_integrated`` is forced False only for the second
    load, so the managed wall's capacity check takes its discrete-GPU early
    return instead of reaching ``comfy_dynamic.pinned_staging_active``, which
    this rig's fake ``comfy`` module cannot answer."""
    from dgx_monarch import mesh_safety, residency_mode

    recorded = []
    monkeypatch.setattr(
        resident_ledger, "declare",
        lambda _active, residency: recorded.append(residency))

    store, calls = rig
    store.slab_weights = False
    store.ensure("m.safetensors", None, STACK)
    assert recorded == ["stock"]

    recorded.clear()
    monkeypatch.setenv(residency_mode.ENV_ACTIVE, "1")
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: False)
    store.ensure("other.safetensors", None, STACK, slot="uncond")
    assert calls.load == 2
    assert recorded == []


def test_a_discarded_model_is_unloaded_without_a_host_copy():
    """Every loaded clone of the dropped model points at its load device before
    comfy's unload detaches it, so the detach moves nothing; a different model
    keeps its own offload device."""
    from types import SimpleNamespace

    from dgx_monarch.actor import resident_ledger

    dropped, kept = object(), object()
    base = SimpleNamespace(model=dropped, load_device="cuda:0", offload_device="cpu")
    clone = SimpleNamespace(model=dropped, load_device="cuda:0", offload_device="cpu")
    other = SimpleNamespace(model=kept, load_device="cuda:0", offload_device="cpu")
    seen = []
    mm = SimpleNamespace(
        current_loaded_models=[SimpleNamespace(model=clone), SimpleNamespace(model=other)],
        unload_all_models=lambda: seen.append((base.offload_device, clone.offload_device, other.offload_device)),
        soft_empty_cache=lambda: seen.append("emptied"))

    resident_ledger.unload_without_offload(mm, (base, None))

    assert seen == [("cuda:0", "cuda:0", "cpu"), "emptied"]


def test_a_store_wide_unload_pins_every_slot_before_the_first_drop(monkeypatch):
    """comfy's unload is global, so dropping one slot detaches the other too.
    Pinned one slot at a time, the first drop copied the second model to host
    RAM: 66 to 70 s per recycle for an Ideogram4 pair (hardware, 2026-10-06).
    Both slots must point at their load device before any unload runs."""
    import sys
    from types import ModuleType, SimpleNamespace

    from dgx_monarch.actor import resident_ledger

    def slot(model):
        base = SimpleNamespace(model=model, load_device="cuda:0", offload_device="cpu")
        active = SimpleNamespace(model=model, load_device="cuda:0", offload_device="cpu")
        return SimpleNamespace(base_patcher=base, active_patcher=active)

    cond, uncond = slot(object()), slot(object())
    stranger = SimpleNamespace(model=object(), load_device="cuda:0", offload_device="cpu")
    mm = ModuleType("comfy.model_management")
    mm.current_loaded_models = [SimpleNamespace(model=cond.active_patcher),
                                SimpleNamespace(model=uncond.active_patcher),
                                SimpleNamespace(model=stranger)]
    comfy = ModuleType("comfy")
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

    resident_ledger.pin_store_for_unload(cond, None, uncond)

    assert {p.offload_device for s in (cond, uncond) for p in (s.base_patcher, s.active_patcher)} == {"cuda:0"}
    assert stranger.offload_device == "cpu"      # a model the store does not own keeps comfy's default
    resident_ledger.pin_store_for_unload(None, None)  # an empty store imports nothing and moves nothing


def test_unload_all_pins_both_slots_before_it_drops_either(rig, monkeypatch):
    """The pin has to run while both slots are still loaded: the first drop
    already detaches the second model."""
    store, _calls = rig
    store.slab_weights = False
    store.ensure("m.safetensors", None, STACK)
    store.ensure("other.safetensors", None, STACK, slot="uncond")
    order = []
    monkeypatch.setattr(
        resident_ledger, "pin_store_for_unload",
        lambda *slots: order.append(("pin", [slot is not None for slot in slots])))
    drop = store._drop
    monkeypatch.setattr(store, "_drop", lambda slot: (order.append(slot), drop(slot))[1])

    store.unload_all()

    assert order == [("pin", [True, True]), "uncond", "cond"]
    assert store.current is None and store.uncond is None


def test_a_partially_loaded_model_keeps_comfys_offload_on_drop():
    """Some of its weights are on the host already; pinning would copy them to
    the device just to discard them."""
    from types import SimpleNamespace

    from dgx_monarch.actor import resident_ledger

    partial = SimpleNamespace(model_lowvram=True)
    base = SimpleNamespace(model=partial, load_device="cuda:0", offload_device="cpu")
    seen = []
    mm = SimpleNamespace(current_loaded_models=[SimpleNamespace(model=base)],
                         unload_all_models=lambda: seen.append(base.offload_device),
                         soft_empty_cache=lambda: None)

    resident_ledger.unload_without_offload(mm, (base,))

    assert seen == ["cpu"]


def test_a_drop_flushes_the_cache_after_its_last_reference_goes(rig, monkeypatch):
    """The weights stay on the device through the unload, so the flush that
    returns their blocks to the host must run once nothing holds the base;
    otherwise the next capacity quote reads the host as still holding the model
    (flux2 fp8mixed with compile_dit, 2026-09-30)."""
    import sys

    store, calls = rig
    store.ensure("m.safetensors", None, STACK)
    released_at_flush = []
    monkeypatch.setattr(sys.modules["comfy.model_management"], "soft_empty_cache",
                        lambda: released_at_flush.append(calls.base_refs[0]() is None))

    store.unload_all()

    assert released_at_flush[-1] is True


def test_a_failed_flush_after_the_drop_still_completes_it(rig, monkeypatch):
    """The flush runs once the slot is released and its resources closed, so a
    device error there leaves nothing half dropped."""
    import sys

    store, calls = rig
    store.ensure("m.safetensors", None, STACK)

    def flush():
        if calls.base_refs[0]() is None:
            raise RuntimeError("CUDA error: unspecified launch failure")

    monkeypatch.setattr(sys.modules["comfy.model_management"], "soft_empty_cache", flush)

    store.unload_all()

    assert store.current is None
    assert store.snapshot()["cleanup_failed_slots"] == []
