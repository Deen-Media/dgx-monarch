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
    from dgx_monarch import mesh_safety

    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 100 << 30)
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
    monkeypatch.setattr(store, "_drop", lambda slot, *rest: (order.append(slot), drop(slot, *rest))[1])

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


# The copy-free drop (``unload_dropped``): only the dropped model's records leave
# comfy's loaded list, and no weight moves to the host. The stand-ins below have
# the shape of comfy's LoadedModel and ModelPatcher; the copy-free drop seam of
# tests/canary/comfy_seam_contracts.py runs the same drop on the real classes.

class _Record:
    def __init__(self, patcher, log):
        self.model, self.log, self.real_model = patcher, log, object()
        self.model_finalizer = _Namespace(detach=lambda: log.append(("finalizer", patcher.name)))

    def model_unload(self):
        self.log.append(("model_unload", self.model.name, self.model.offload_device))
        return True


def _Namespace(**fields):
    from types import SimpleNamespace

    return SimpleNamespace(**fields)


def _slot(name, log, *, lowvram=False, model=None, **fields):
    model = model or _Namespace(model_lowvram=lowvram)

    def patcher(kind):
        label = f"{name}-{kind}"
        return _Namespace(name=label, model=model, load_device="cuda:0", offload_device="cpu", backup={"w": 1},
                          detach=lambda unpatch_all=True: log.append(("detach", label, unpatch_all)),
                          unpin_all_weights=lambda: log.append(("unpin", label)))

    return _Namespace(**{"base_patcher": patcher("base"), "active_patcher": patcher("active"), "slab": None,
                         "fsdp_checkpoint_pin": None, "residency_rung": "stock_fits", **fields})


def _comfy(log, *records):
    return _Namespace(current_loaded_models=list(records), unload_all_models=lambda: log.append("unload_all"),
                      soft_empty_cache=lambda: log.append("flush"))


@pytest.fixture
def unified(monkeypatch):
    from dgx_monarch import mesh_safety

    monkeypatch.delenv(resident_ledger.COPY_FREE_DROP_DISABLE_ENV, raising=False)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)


def test_a_drop_unloads_only_its_own_model_and_leaves_the_other_slot_loaded(unified):
    """comfy's global unload would detach the other slot's model to the host too, a copy made
    before anything is freed. The drop takes its own records out of the list and no other."""
    log: list = []
    dropped, kept = _slot("old", log), _slot("stays", log)
    mine, clone, theirs = (_Record(dropped.active_patcher, log), _Record(dropped.base_patcher, log),
                           _Record(kept.active_patcher, log))
    mm = _comfy(log, mine, theirs, clone)
    assert resident_ledger.unload_dropped(mm, dropped, kept) is True
    assert mm.current_loaded_models == [theirs]
    # Each of its records unloads pinned to the load device, so the detach moves nothing.
    assert log == [("model_unload", "old-base", "cuda:0"), ("model_unload", "old-active", "cuda:0"), "flush"]
    # The slot that stays is pinned too, for the unload comfy runs at a later load.
    assert {kept.base_patcher.offload_device, kept.active_patcher.offload_device} == {"cuda:0"}


def test_a_partly_loaded_model_is_detached_in_place_on_drop(unified):
    """Pinned, comfy's detach would copy its host part to the device; unpinned, its device part to
    the host. Both are waste for a model being thrown away, so it takes the step comfy takes
    when it replaces a clone, which moves no weight, and what the skipped unpatch would
    have released is released here."""
    log: list = []
    dropped = _slot("old", log, lowvram=True)
    mine = _Record(dropped.active_patcher, log)
    mm = _comfy(log, mine)
    assert resident_ledger.unload_dropped(mm, dropped, None) is True
    assert mm.current_loaded_models == []
    assert log == [("detach", "old-active", False), ("unpin", "old-active"), ("finalizer", "old-active"), "flush"]
    assert dropped.active_patcher.backup == {} and mine.model_finalizer is None and mine.real_model is None
    assert dropped.active_patcher.offload_device == "cuda:0", "its option patches are not moved either"


def test_a_model_both_slots_share_stays_loaded(unified):
    log: list = []
    kept = _slot("stays", log)
    dropped = _slot("old", log, model=kept.base_patcher.model)
    shared = _Record(kept.active_patcher, log)
    mm = _comfy(log, shared, _Record(dropped.active_patcher, log))
    assert resident_ledger.unload_dropped(mm, dropped, kept) is True
    assert len(mm.current_loaded_models) == 2 and log == ["flush"]


@pytest.mark.parametrize(("change", "pinned", "why"), [
    ({}, True, "a whole stock resident on unified memory"),
    ({"slab": object()}, True, "a slab resident, pinned at load already"),
    ({"lowvram": True}, False, "a partly loaded model: the pin would copy its host part to the device"),
    ({"residency_rung": "comfy_managed"}, False, "comfy-managed weights are ComfyUI's to move"),
])
def test_the_slot_that_stays_is_pinned_only_where_no_later_unload_can_copy_it(unified, change, pinned, why):
    from dgx_monarch.actor import capacity_quote

    assert capacity_quote.RUNG_COMFY_MANAGED == "comfy_managed"
    log: list = []
    dropped, kept = _slot("old", log), _slot("stays", log, **change)
    assert resident_ledger.stays_on_device(kept) is pinned, why
    assert resident_ledger.stays_on_device(None) is True, "an empty slot has nothing to copy"
    assert resident_ledger.drop_copies_nothing(dropped, kept) is pinned, why
    resident_ledger.unload_dropped(_comfy(log, _Record(dropped.active_patcher, log)), dropped, kept)
    assert (kept.active_patcher.offload_device == "cuda:0") is pinned, why


@pytest.mark.parametrize(("change", "switch", "integrated", "why"), [
    ({"slab": object()}, "", True, "a slab closes only after every model is detached"),
    ({"fsdp_checkpoint_pin": object()}, "", True, "so does a shard build's checkpoint pin"),
    ({}, "1", True, "DGXM_DISABLE_COPY_FREE_DROP=1"),
    ({}, "", False, "a discrete GPU: an offload there frees device memory the next load may need"),
])
def test_a_slab_a_shard_build_a_discrete_gpu_and_the_switch_keep_the_global_unload(
        monkeypatch, unified, change, switch, integrated, why):
    from dgx_monarch import mesh_safety

    monkeypatch.setenv(resident_ledger.COPY_FREE_DROP_DISABLE_ENV, switch)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: integrated)
    assert resident_ledger.drop_copies_nothing(_slot("old", [], **change), None) is False, (
        "nothing is promised, even beside an empty slot")
    log: list = []
    dropped, kept = _slot("old", log, **change), _slot("stays", log)
    mm = _comfy(log, _Record(dropped.active_patcher, log), _Record(kept.active_patcher, log))
    assert resident_ledger.detaches_alone(dropped) is False, why
    assert resident_ledger.drop_copies_nothing(dropped, kept) is False, "and the price is told so"
    assert resident_ledger.unload_dropped(mm, dropped, kept) is False
    assert log == ["unload_all", "flush"], why
    assert kept.active_patcher.offload_device == "cpu", "the earlier drop pinned only what it dropped"


def test_a_shard_base_keeps_the_global_unload_where_a_plain_stock_resident_does_not(unified):
    plain, shards = _slot("old", []), _slot("old", [])
    shards.base_patcher._dgxm_fsdp = True
    assert resident_ledger.detaches_alone(plain) is True
    assert resident_ledger.detaches_alone(shards) is False


def test_the_store_leaves_the_other_slot_loaded_and_unmarked_after_a_drop(rig, monkeypatch):
    """Both slots held, the cond slot dropped: comfy's list keeps the uncond model, and the store
    does not mark it as unloaded, since nothing reloads. Under the switch the earlier drop stands."""
    import sys

    store, calls = rig
    store.slab_weights = False
    monkeypatch.delenv(resident_ledger.COPY_FREE_DROP_DISABLE_ENV, raising=False)
    store.ensure("m.safetensors", None, STACK)
    store.ensure("other.safetensors", None, STACK, slot="uncond")
    from dgx_monarch import mesh_safety

    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    mm = sys.modules["comfy.model_management"]
    kept = store.uncond.base_patcher
    kept.load_device = "cuda:0"
    assert len(mm.current_loaded_models) == 2 and kept.offload_device == "cpu"
    store._drop("cond")
    assert store.current is None and store._gpu_evicted == set()
    assert [record.model for record in mm.current_loaded_models] == [kept]
    assert (calls.own_unloads, calls.unload_all) == (1, 1), "one unload step, the dropped model's own"
    assert kept.offload_device == "cuda:0", "the store hands the drop the slot that stays, and it is pinned"

    store.ensure("m.safetensors", None, STACK)
    monkeypatch.setenv(resident_ledger.COPY_FREE_DROP_DISABLE_ENV, "1")
    store._drop("cond")
    assert store._gpu_evicted == {"uncond"}, "the global unload took the other slot's GPU weights"


def test_a_store_wide_unload_stays_global(rig, monkeypatch):
    """``unload_all`` is the clear: ComfyUI must hold no model after it, the store's or any
    other, so it keeps the global unload even where a single drop would detach alone."""
    from dgx_monarch import mesh_safety

    store, calls = rig
    store.slab_weights = False
    monkeypatch.delenv(resident_ledger.COPY_FREE_DROP_DISABLE_ENV, raising=False)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    store.ensure("m.safetensors", None, STACK)
    store.ensure("other.safetensors", None, STACK, slot="uncond")
    assert resident_ledger.detaches_alone(store.current), "a single drop of this slot would detach alone"
    store.unload_all()
    assert store.current is None and store.uncond is None
    assert (calls.own_unloads, calls.unload_all) == (0, 2), "two drops, each through the global unload"


def test_the_drops_switch_reaches_a_worker_through_its_reviewed_environment():
    """The worker loop deletes every name outside its reviewed list before the actors start, so
    a switch set on the Worker's unit is only real if its name is on that list."""
    from dgx_monarch.cli import worker_process_env

    environment = {"PATH": "/usr/bin", resident_ledger.COPY_FREE_DROP_DISABLE_ENV: "1"}
    assert worker_process_env.worker_environment_key_allowed(resident_ledger.COPY_FREE_DROP_DISABLE_ENV)
    assert worker_process_env.filter_worker_environment(environment) == 0
    assert environment[resident_ledger.COPY_FREE_DROP_DISABLE_ENV] == "1"


def test_mark_after_drop_says_what_the_drop_did_to_the_other_slot():
    store = ModelStore()
    store.current, store.uncond = object(), object()
    store._gpu_evicted.add("cond")
    resident_ledger.mark_after_drop(store, "cond", True)
    assert store._gpu_evicted == set()
    resident_ledger.mark_after_drop(store, "cond", False)
    assert store._gpu_evicted == {"uncond"}


def test_a_store_wide_drop_ends_in_one_host_trim_and_says_what_came_back(monkeypatch, caplog):
    """Memory a user cleared must reach the box: the flush after a store-wide drop trims the C library's arenas.

    ``flush_dropped`` is the last step of every drop; the hard clear's is store-wide. The drop of one slot
    before a fresh load is not trimmed (one shipped template drops once a render). The trim is
    replaced here by a recorder, so the test needs no glibc.
    """
    from dgx_monarch.actor import host_trim

    order: list[str] = []
    returned = {"bytes": 10 * 2**30}

    def trim():
        order.append("trim")
        return True, 0.31, 20 * 2**30, 20 * 2**30 + returned["bytes"]

    monkeypatch.setattr(host_trim, "trim", trim)
    mm = type("Mm", (), {"soft_empty_cache": staticmethod(lambda: order.append("flush"))})
    with caplog.at_level(logging.INFO):
        resident_ledger.flush_dropped(mm, "cond", True)
    assert order == ["flush", "trim"]
    assert "the host trim after the drop returned 10.0 GiB to the box" in caplog.text
    assert "(MemAvailable 20.0 to 30.0 GiB, 0.31 s)" in caplog.text
    # A trim that returned little says nothing, and a cache flush that raised is still followed by the trim.
    caplog.clear()
    order.clear()
    returned["bytes"] = 64 * 2**20

    def broken():
        order.append("flush")
        raise RuntimeError("CUDA is gone")

    with caplog.at_level(logging.INFO):
        resident_ledger.flush_dropped(type("Mm", (), {"soft_empty_cache": staticmethod(broken)}), "cond", True)
    assert order == ["flush", "trim"] and "host trim" not in caplog.text
    # A C library with no trim: nothing is said here (host_trim says so once itself).
    monkeypatch.setattr(host_trim, "trim", lambda: (False, 0.0, 20 * 2**30, 20 * 2**30))
    caplog.clear()
    with caplog.at_level(logging.INFO):
        resident_ledger.flush_dropped(mm, "cond", True)
    assert "host trim" not in caplog.text
    # The drop of one slot before a fresh load: the cache is flushed, nothing is trimmed.
    order.clear()
    monkeypatch.setattr(host_trim, "trim", lambda: order.append("trim") or (True, 0.1, 0, 2**34))
    resident_ledger.flush_dropped(mm, "cond")
    assert order == ["flush"]
