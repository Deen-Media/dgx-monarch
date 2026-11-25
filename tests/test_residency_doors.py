"""Every door through which the Workers' models go makes the driver's residency memos true.

The render-site checks credit a model as already in MemAvailable from the
driver's memos (``nodes/pending_load_relief.dit_resident``), and the driver
footprint (#52) and MiniMax H3 checks credit an explicit preset's DiT
because its loader node placed it. This file holds, through each door's real
code:

- an operator's drop (a Clear VRAM node at hard, a configuration change, the
  capacity rescue's slot drop) ends the explicit preset's credit until the
  loader node runs again or a load completes (``recycle_drain.preset_placed``), so a cached loader
  after a hard clear is charged; a recycle, a deferral and the gate's own
  unloads keep it, as before;
- a configuration change drains every memo, the loader's included;
- a process-group teardown, its rollback, a worker-policy change that unloads
  every resident, a fleet marked defunct and a new fleet in place of a retired
  one drain the render memos and mark the load still to come, and leave the
  loader's memo and the preset credit;
- a worker-policy change marks the same exactly where a Worker's reply names a
  slot it dropped (an adopted resident marks nothing), through
  ``apply_worker_args``, a render's own setup and the cross-mode leg's
  restore; a reply without that key leaves it to the driver's inference,
  which reads the Worker's own mode chain;
- the loader node writes its own placement, in each slot, so a dual-model
  graph's second loader does not hide the first's.

CPU only, on the driver-unload rig of ``test_driver_unload`` and the mesh
rigs of ``test_core_reliability`` and ``topology_transition_helpers``.
"""
from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest
import torch

import test_driver_unload as rig
from dgx_monarch import mesh_safety, mesh_setup
from dgx_monarch.mesh_stop import PROC_STOP_DEFAULT_S
from dgx_monarch.nodes import (
    common,
    loader_preflight,
    ops,
    pending_load_relief,
    recycle_drain,
    render_preflight,
    samplers,
)
from dgx_monarch.nodes.common import MeshSpec, ModelSpec
from test_core_reliability import _replaceable_fleet
from test_driver_unload import DIT_BYTES, UNET, _comfy, _h3_latent, _h3_need, _h3_request, _h3_rig
from topology_transition_helpers import _acks, _Endpoint, _Future, _handle, _ValueMesh, _Workers

_isolated, box, journal = rig._isolated, rig.box, rig.journal

_GIB = 2 ** 30
LTX = "ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors"
LTX_BYTES = 22 * _GIB


class _Checked(Exception):
    """Raised where run_render leaves the render-site checks behind it."""


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    pending_load_relief.reset()
    monkeypatch.setattr(common, "_claim_adoption_context",
                        lambda: (_ for _ in ()).throw(_Checked()))
    monkeypatch.setattr(pending_load_relief, "relieve", lambda *_a, **_k: None)
    for memo in ("_LAST_RENDERED_UNET", "_LAST_SUBMITTED_UNET", "_LAST_DRIVER_CHARGED_UNET"):
        monkeypatch.setattr(render_preflight, memo, None)
    monkeypatch.setattr(loader_preflight, "_CLEARED_UNETS", {})
    yield
    pending_load_relief.reset()


def _cond() -> list:
    return [[torch.zeros(1, _h3_request()["positive"][0][0].shape[1], 8), {}]]


def _render(model, latent) -> None:
    """One queue press through the KSampler node, up to the end of the checks."""
    with pytest.raises(_Checked):
        samplers.DGXMonarchKSampler().sample(
            model, 1, 8, 1.0, "euler", "simple", _cond(), _cond(), latent)


def _refused(model, latent) -> str:
    with pytest.raises((mesh_safety.StockLoadCapacityError,
                        render_preflight.driver_footprint.DriverFootprintCapacityError)) as refused:
        samplers.DGXMonarchKSampler().sample(
            model, 1, 8, 1.0, "euler", "simple", _cond(), _cond(), latent)
    return str(refused.value)


def _hard_clear(monkeypatch, model) -> None:
    """Clear VRAM (DGX Monarch) at hard: every Worker unloads its models."""
    handle = model.mesh.handle
    monkeypatch.setattr(ops, "ensure_live", lambda _handle: handle)
    handle.call_all = lambda method, *_a, **_k: [{"rank": 0, "method": method}]
    response = ops.DGXMonarchClearVRAM().clear(model.mesh, "hard", include_driver=False)
    assert json.loads(response["result"][0])["workers"] == [{"rank": 0, "method": "clear_vram"}]


@pytest.fixture
def h3(monkeypatch, tmp_path, box):
    """H3 on the pair under an explicit uly2: the box reads room for the
    activations alone, short of the DiT beside them, and ComfyUI holds nothing."""
    rig_ = _h3_rig(monkeypatch, tmp_path, preset="uly2")
    box.mem = _h3_need(weights=0) + 4 * _GIB
    assert box.mem < _h3_need(weights=DIT_BYTES)
    _comfy(monkeypatch, box, models=())
    return ModelSpec(mesh=rig_.model.mesh, unet_name=UNET)


@pytest.fixture
def ltx(monkeypatch, tmp_path, box):
    """LTX 2.3 under an explicit uly2: the driver footprint (#52) charges a
    pending 22 GiB DiT; the box reads 8 GiB, the DiT already inside it."""
    path = tmp_path / LTX
    path.write_bytes(b"x")
    fake = types.ModuleType("folder_paths")
    fake.get_full_path = lambda _kind, _name: str(path)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    monkeypatch.setattr(render_preflight, "sniff_checkpoint", lambda _path: ("ltx", "fp8"))
    monkeypatch.setattr("dgx_monarch.driver_footprint.file_size_bytes", lambda _path: LTX_BYTES)
    monkeypatch.setattr("dgx_monarch.render_memory_price.preflight_render_memory",
                        lambda *_a, **_k: None)
    box.mem = 8 * _GIB
    _comfy(monkeypatch, box, models=())
    handle = rig._leased(owns_hosts=False, world=2, call_all=lambda *a, **k: [],
                         effective_worker_args=lambda requested=None: dict(requested or {}))
    mesh = SimpleNamespace(topology_preset="uly2", worker_args={}, world=2, handle=handle)
    return ModelSpec(mesh=mesh, unet_name=LTX)


LTX_LATENT = {"samples": torch.zeros(1, 128, 4, 8, 8)}


# --- A. an operator's drop ends an explicit preset's credit -------------------------------


def test_h3_under_an_explicit_preset_charges_the_dit_a_hard_clear_dropped(monkeypatch, h3):
    render_preflight.note_successful_render(UNET)  # a render of it succeeded
    loader_preflight._CLEARED_UNETS[UNET] = None   # and the loader's price cleared it
    _render(h3, _h3_latent())                      # credited: the loader placed it
    _hard_clear(monkeypatch, h3)                   # ComfyUI's cached loader does not run again
    assert "Charged: DiT weights 19.5 GiB (not yet loaded)" in _refused(h3, _h3_latent())
    assert loader_preflight._CLEARED_UNETS == {UNET: None}, "the loader's memo is not a render memo"


def test_the_footprint_check_under_an_explicit_preset_charges_the_dit_a_hard_clear_dropped(
        monkeypatch, ltx):
    render_preflight.note_successful_render(LTX)
    _render(ltx, LTX_LATENT)
    _hard_clear(monkeypatch, ltx)
    assert "driver-side footprint preflight" in _refused(ltx, LTX_LATENT)


def test_the_loader_running_again_after_the_drop_restores_the_credit(monkeypatch, h3):
    _hard_clear(monkeypatch, h3)
    pending_load_relief.note_deferred(UNET)        # the loader ran, and left it to the ceremony
    _render(h3, _h3_latent())


def test_the_capacity_rescues_slot_drop_is_an_operators_drop(monkeypatch, h3):
    from dgx_monarch.nodes import consent_rescue

    calls: list = []
    handle = SimpleNamespace(call_all=lambda method, *_a, **_k: calls.append(method))
    consent_rescue._drop_uncertified_slot(SimpleNamespace(unet_name=UNET), handle)
    assert calls == ["unload"]
    _refused(h3, _h3_latent())


@pytest.mark.parametrize("door", ["recycle", "a ceremony's unload", "a deferral", "none"])
@pytest.mark.parametrize("dropped_before", [False, True])
def test_an_explicit_presets_credit_stands_where_no_operator_dropped(
        monkeypatch, h3, door, dropped_before):
    """Controls, as before this change: a recycle (ComfyUI's cached loader does
    not place again, and the credit stands there as it always did), the gate's
    own unloads, a deferral, and an eager load with nothing after it. With an
    operator's drop before the loader ran, the credit rests on the loader's
    run alone, so none of these doors may take it."""
    if dropped_before:
        _hard_clear(monkeypatch, h3)
    if door == "a deferral":
        pending_load_relief.note_deferred(UNET)
    else:
        pending_load_relief.note_placed(UNET)
    if door == "recycle":
        recycle_drain.drop_residency_memos()
    elif door == "a ceremony's unload":
        pending_load_relief.note_unloaded()
    _render(h3, _h3_latent())


# --- B. the loader writes its own placement, in each slot ---------------------------------


@pytest.fixture
def loader(monkeypatch, tmp_path):
    """The UNET loaders' eager load under an explicit preset, down to the Worker call."""
    import contextlib

    from dgx_monarch import capacity_agreement
    from dgx_monarch.nodes import loaders, render_session

    calls: list = []
    handle = SimpleNamespace(
        world=2, defunct=False, sample_leases={}, abandoned_sample_leases={},
        call_all=lambda endpoint, *_a, **_k: calls.append(endpoint) or [{"host": "h", "transition": "load"}])
    monkeypatch.setattr(loaders, "ensure_live", lambda value, **_k: value)
    monkeypatch.setattr(mesh_setup, "ensure_request_setup", lambda *_a, **_k: object())
    monkeypatch.setattr(render_session, "mutation_render_session",
                        lambda _handle: contextlib.nullcontext())
    monkeypatch.setattr(render_preflight, "preflight_sol_sequence_parallel", lambda *_a: None)
    monkeypatch.setattr(loader_preflight, "preflight_loader_footprint", lambda *_a, **_k: object())
    monkeypatch.setattr(loader_preflight, "record_rescue_row", lambda *_a, **_k: None)
    monkeypatch.setattr(capacity_agreement, "agree_load", lambda *_a, **_k: None)
    mesh = MeshSpec(handle=handle, topology_preset="uly2", attention="TORCH_FLASH",
                    sync_ulysses=True)
    return SimpleNamespace(loaders=loaders, mesh=mesh, calls=calls)


def test_the_loader_writes_its_placement_and_the_second_slot_keeps_the_first(loader):
    loader.loaders.DGXMonarchUNETLoader().load(loader.mesh, "cond.safetensors")
    loader.loaders.DGXMonarchUncondUNETLoader().load(loader.mesh, "uncond.safetensors")
    assert loader.calls == ["load_model", "load_uncond_model"]
    assert pending_load_relief.dit_resident("cond.safetensors")
    assert pending_load_relief.dit_resident("uncond.safetensors")
    model = SimpleNamespace(unet_name="cond.safetensors", mesh=loader.mesh)
    assert not pending_load_relief.load_pending(model, None)


def test_the_loaders_run_after_a_hard_clear_credits_the_preset_again(monkeypatch, h3, loader):
    """The cached loader is charged after the hard clear; once the loader node
    runs again and places the model, the H3 check credits it as before."""
    _hard_clear(monkeypatch, h3)
    _refused(h3, _h3_latent())
    h3.mesh.handle.call_all = lambda endpoint, *_a, **_k: [{"host": "h", "transition": "load"}]
    h3.mesh.attention, h3.mesh.sync_ulysses = "TORCH_FLASH", True
    loader.loaders.DGXMonarchUNETLoader().load(h3.mesh, UNET)
    _render(h3, _h3_latent())


# --- C. the mesh layer's doors -------------------------------------------------------------


def _stamp(monkeypatch, name: str = UNET) -> None:
    """A render, a pipelined push, the loader's memo and its placement, all for one
    model, after a hard Clear VRAM, so the preset credit rests on the loader's run.
    Each door below then reads that credit through the H3 check under uly2 (``h3``)."""
    cleared = SimpleNamespace(call_all=lambda method, *_a, **_k: [{"rank": 0, "method": method}])
    monkeypatch.setattr(ops, "ensure_live", lambda _handle: cleared)
    ops.DGXMonarchClearVRAM().clear(SimpleNamespace(handle=cleared), "hard", include_driver=False)
    pending_load_relief.note_placed(name)
    render_preflight.note_successful_render(name)
    render_preflight.note_submitted_render(name)
    loader_preflight._CLEARED_UNETS[name] = None


def _render_memos_drained(name: str = UNET) -> bool:
    return (not pending_load_relief.dit_resident(name, charged=True)
            and pending_load_relief.load_pending(SimpleNamespace(unet_name=name), True))


def test_a_configuration_change_drains_every_memo_and_ends_the_preset_credit(
        monkeypatch, isolated_environ, h3):
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch import monarch_launch

    _stamp(monkeypatch)
    events: list = []
    config = types.SimpleNamespace(
        source="cluster.toml", hosts=(types.SimpleNamespace(gpus=1),), comfy_dir="/comfy")
    _replaceable_fleet(monkeypatch, config, events)
    monkeypatch.setattr(monarch_launch, "prepare_driver", lambda _config: None)
    with pytest.raises(RuntimeError, match="replacement reached"):
        mesh_mod.get_mesh(mode="cluster")
    assert events == [("shutdown", 30, PROC_STOP_DEFAULT_S), "spawn"]
    assert _render_memos_drained()
    assert loader_preflight._CLEARED_UNETS == {}, "the loader node runs again and stamps its memo"
    assert "Charged: DiT weights 19.5 GiB (not yet loaded)" in _refused(h3, _h3_latent())


@pytest.mark.parametrize("already", [False, True])
def test_a_fleet_marked_defunct_drains_before_the_next_renders_checks(
        monkeypatch, tmp_path, box, already):
    """run_render prices before ensure_live respawns a dead fleet, so the mark
    comes where the fleet goes defunct (``mesh_helpers._publish_defunct``):
    the next render's checks charge the DiT its replacement lacks. A handle
    already defunct (a late fault on a retired fleet) marks nothing, since a
    newer fleet may hold the model."""
    from monarch._rust_bindings.monarch_hyperactor.supervision import SupervisionError

    from dgx_monarch import mesh_helpers

    rig_ = _h3_rig(monkeypatch, tmp_path, preset="auto")
    box.mem = _h3_need(weights=0) + 4 * _GIB
    _comfy(monkeypatch, box, models=())
    model = ModelSpec(mesh=rig_.model.mesh, unet_name=UNET)
    render_preflight.note_successful_render(UNET)
    _render(model, _h3_latent())
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    dead = SimpleNamespace(defunct=already, teardown_complete=False, replacement_blocked=None)
    marker = mesh_safety._TEARDOWN_REASON_MARKERS[0]
    mesh_helpers.mark_defunct_on_supervision_failure(
        dead, SupervisionError(f"proc has status: stopped: {marker}"))
    assert dead.defunct is True
    if already:
        _render(model, _h3_latent())
    else:
        assert "Charged: DiT weights 19.5 GiB (not yet loaded)" in _refused(model, _h3_latent())


@pytest.mark.parametrize("latch_fails", [True, False])
def test_an_fsdp_gate_cleanup_that_retires_the_fleet_drains_before_the_next_renders_checks(
        monkeypatch, tmp_path, box, latch_fails):
    """The FSDP Gate's cleanup after an aborted proof: where the all-rank unload
    and the DIRTY latch both fail, it marks the handle defunct itself
    (``gate_fsdp._latch_or_retire_failed_cleanup``), so the next render's checks
    charge the DiT the replacement lacks. A latched handle stays as it was."""
    from dgx_monarch.nodes import gate_fsdp

    rig_ = _h3_rig(monkeypatch, tmp_path, preset="auto")
    box.mem = _h3_need(weights=0) + 4 * _GIB
    _comfy(monkeypatch, box, models=())
    model = ModelSpec(mesh=rig_.model.mesh, unet_name=UNET)
    render_preflight.note_successful_render(UNET)
    _render(model, _h3_latent())
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    latches: list = []

    def latch(*args):
        latches.append(args)
        if latch_fails:
            raise RuntimeError("the DIRTY latch was not published")

    def unload(*_a, **_k):
        raise TimeoutError("the all-rank unload was not confirmed")

    handle = SimpleNamespace(world=2, defunct=False, teardown_complete=False,
                             replacement_blocked=None, setup_cleanup_state=None,
                             call_all=unload, _latch_ambiguous_mutation=latch)
    logger = SimpleNamespace(error=lambda *_a: None, info=lambda *_a: None)
    assert gate_fsdp.cleanup_aborted_fsdp_proof(
        handle, RuntimeError("proof aborted"), logger=logger, state={}) is False
    assert (len(latches), handle.defunct) == ((2, True) if latch_fails else (1, False))
    if latch_fails:
        assert "Charged: DiT weights 19.5 GiB (not yet loaded)" in _refused(model, _h3_latent())


def test_a_new_fleet_in_place_of_a_retired_one_marks_the_load_still_to_come(
        monkeypatch, isolated_environ, h3):
    """A fleet marked defunct outside result collection: ensure_live respawns it."""
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch import monarch_launch

    _stamp(monkeypatch)
    events: list = []
    config = types.SimpleNamespace(
        source="cluster.toml", hosts=(types.SimpleNamespace(gpus=1),), comfy_dir="/comfy")
    _key, retired = _replaceable_fleet(monkeypatch, config, events)
    retired.defunct, retired.teardown_complete = True, True
    retired.config, retired.gpus_per_host = config, 1
    monkeypatch.setattr(monarch_launch, "prepare_driver", lambda _config: None)
    with pytest.raises(RuntimeError, match="replacement reached"):
        mesh_mod.ensure_live(retired)
    assert events == ["spawn"]
    assert _render_memos_drained()
    assert loader_preflight._CLEARED_UNETS == {UNET: None}
    _render(h3, _h3_latent())  # not an operator's drop: the preset credit stands


def test_the_init_node_after_a_recycle_starts_a_fleet_and_keeps_the_preset_credit(
        monkeypatch, isolated_environ, h3):
    """The release runner's path: a recycle retires the cached fleet, the
    configuration is unchanged, and the Init node's get_mesh starts a new one.
    That is no operator's drop, so a cached loader keeps the preset credit."""
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch import monarch_launch

    _stamp(monkeypatch)
    events: list = []
    config = types.SimpleNamespace(
        source="cluster.toml", hosts=(types.SimpleNamespace(gpus=1),), comfy_dir="/comfy")
    _key, retired = _replaceable_fleet(monkeypatch, config, events)
    retired.defunct, retired.teardown_complete, retired.config_fingerprint = True, True, "same"
    monkeypatch.setattr(mesh_mod, "config_fingerprint", lambda _path: "same")
    monkeypatch.setattr(monarch_launch, "prepare_driver", lambda _config: None)
    with pytest.raises(RuntimeError, match="replacement reached"):
        mesh_mod.get_mesh(mode="cluster")
    assert events == ["spawn"]
    _render(h3, _h3_latent())


def test_a_live_cached_fleet_marks_nothing(monkeypatch, isolated_environ):
    """The Init node's every run reaches get_mesh; a fleet it reuses holds what it held."""
    from dgx_monarch import mesh as mesh_mod

    _stamp(monkeypatch)
    events: list = []
    config = types.SimpleNamespace(
        source="cluster.toml", hosts=(types.SimpleNamespace(gpus=1),), comfy_dir="/comfy")
    _key, live = _replaceable_fleet(monkeypatch, config, events)
    live.config_fingerprint = "same"
    monkeypatch.setattr(mesh_mod, "config_fingerprint", lambda _path: "same")
    assert mesh_mod.get_mesh(mode="cluster") is live
    assert pending_load_relief.dit_resident(UNET, charged=True)


def test_a_topology_change_drains_the_render_memos(monkeypatch, h3):
    _stamp(monkeypatch)
    workers = _Workers([_Future(_acks())])
    handle, _old, new = _handle(workers)
    handle.ensure_setup(new, "TORCH_FLASH")
    assert len(workers.teardown_group.calls) == 1
    assert _render_memos_drained()
    assert loader_preflight._CLEARED_UNETS == {UNET: None}
    _render(h3, _h3_latent())  # not an operator's drop: the preset credit stands


def test_a_setup_rollback_drains_the_render_memos(monkeypatch, h3):
    """A first setup, so no topology teardown runs before it: only the rollback's."""
    _stamp(monkeypatch)
    workers = _Workers([_Future(_acks())],
                       setup_futures=[_Future({"rank": 0}), RuntimeError("dispatch failed")])
    handle, _old, new = _handle(workers)
    handle.setup_key = None
    with pytest.raises(RuntimeError, match="dispatch failed"):
        handle.ensure_setup(new, "TORCH_FLASH")
    assert len(workers.teardown_group.calls) == 1
    assert _render_memos_drained()
    assert loader_preflight._CLEARED_UNETS == {UNET: None}
    _render(h3, _h3_latent())  # not an operator's drop: the preset credit stands


_LAST = {"slab_weights": True, "lora_low_rss": True}


@pytest.mark.parametrize(("last", "new", "drops"), [
    (_LAST, {**_LAST, "compile_dit": True}, True),
    (_LAST, {**_LAST, "family_override": "krea2"}, True),
    (_LAST, {**_LAST, "fp8_matrix_mult": "on"}, True),
    ({**_LAST, "shared_act_scale": False}, _LAST, True),  # dropped: back to on
    (_LAST, {**_LAST, "shared_act_scale": True}, False),  # added: absent already read on
    (None, {"slab_weights": True}, True),                 # the last apply did not publish
    (_LAST, {**_LAST, "slab_weights": False}, True),      # the store may drop a slab resident
    (_LAST, {**_LAST, "lora_low_rss": False}, True),      # or one with unbaked LoRA patches
    (_LAST, {**_LAST, "mmap_fallback": True}, False),
    ({**_LAST, "fp8_matrix_mult": "auto"}, _LAST, False),  # absent is auto
    ({**_LAST, "shared_act_scale": True}, _LAST, False),   # dropped: still on
    ({"compile_dit": False}, {"compile_dit": False, "mmap_fallback": True}, False),
])
def test_a_reply_without_the_drop_key_leaves_the_inference_to_mark(
        monkeypatch, h3, last, new, drops):
    """A Worker of an older build names no drops in its reply (``{}`` here), so
    the driver's inference answers (``drops_residents``)."""
    _stamp(monkeypatch)
    workers = _Workers([])
    values = _ValueMesh([{}, {}])
    workers.apply_worker_args = _Endpoint(_Future(values))
    handle, _old, _new = _handle(workers)
    handle.active_worker_args = dict(last) if last is not None else {}
    if last is None:
        handle.worker_args_key = None
    handle.apply_worker_args(new, merge_config=False)
    assert len(workers.apply_worker_args.calls) == 1
    assert _render_memos_drained() is drops
    assert loader_preflight._CLEARED_UNETS == {UNET: None}
    _render(h3, _h3_latent())  # not an operator's drop: the preset credit stands


# --- D. a residency change the model store acts on ----------------------------------------


_POLICIES = [
    {},                                                  # the Init node's auto widgets
    {"slab_weights": False, "lora_low_rss": False},      # stock residency, as the gate forces it
    {"slab_weights": True, "lora_low_rss": True},
    {"slab_weights": "auto", "lora_low_rss": True},
    {"slab_weights": True, "lora_low_rss": False},
    {"lora_low_rss": False},
    {"slab_weights": False},
    {"comfy_managed": True, "slab_weights": False, "lora_low_rss": False},
]


def _worker_modes(monkeypatch, args: dict, integrated: bool, fsdp: bool) -> tuple:
    """What a Worker's apply resolves: its defaults (``_uma_memory_defaults``),
    then the slab rule (``slab_mode_effective``), as ``apply_worker_args_impl`` reads them."""
    from dgx_monarch import residency_mode
    from dgx_monarch.actor import comfy_bridge, comfy_dynamic, worker_env

    monkeypatch.setattr(comfy_dynamic, "_REQUESTED", None, raising=False)  # the process latch
    wa = comfy_bridge._uma_memory_defaults(dict(args), integrated=integrated)
    return (worker_env.slab_mode_effective(wa, {"fsdp": fsdp}), bool(wa.get("lora_low_rss")),
            residency_mode.requested(wa))


@pytest.mark.parametrize("fsdp", [False, True])
@pytest.mark.parametrize("integrated", [True, False])
@pytest.mark.parametrize("args", _POLICIES)
def test_the_driver_reads_the_workers_residency_modes(monkeypatch, args, integrated, fsdp):
    from dgx_monarch import mesh_setup_transition

    driver = mesh_setup_transition.residency_modes(args, integrated=integrated, fsdp=fsdp)
    assert driver == _worker_modes(monkeypatch, args, integrated, fsdp)


@pytest.mark.parametrize("integrated", [True, False])
@pytest.mark.parametrize("new", _POLICIES)
@pytest.mark.parametrize("last", _POLICIES)
def test_no_drop_of_the_model_store_goes_unmarked(monkeypatch, last, new, integrated):
    """The real store's setters, in the Worker's order, over every resident a
    policy can hold: whatever they drop, the driver has marked."""
    from dgx_monarch import mesh_setup_transition
    from dgx_monarch.actor import model_store, store_residency

    if (last.get("comfy_managed") is True) is not (new.get("comfy_managed") is True):
        pytest.skip("comfy_managed is a bootstrap policy: a live Worker refuses the change")
    marked = mesh_setup_transition.drops_residents(last, new)
    old_slab, old_lora, managed = _worker_modes(monkeypatch, last, integrated, False)
    new_slab, new_lora, _ = _worker_modes(monkeypatch, new, integrated, False)
    kinds = ({"comfy-managed": {"residency_rung": store_residency.RUNG_COMFY_MANAGED}} if managed
             else {"slab": {"slab": SimpleNamespace(certificate=None)}, "stock": {},
                   "stock with unbaked LoRA patches": {"lora_sig": "lora"},
                   "slab with unbaked LoRA patches": {"slab": SimpleNamespace(certificate=None),
                                                      "lora_sig": "lora"}})
    for kind, fields in kinds.items():
        store = model_store.ModelStore()
        store.slab_weights, store.lora_low_rss = old_slab, old_lora
        store.current = SimpleNamespace(**{
            "slab": None, "residency_rung": "", "lora_sig": "", "base_baked": False,
            "base_key": ("model.safetensors",), "slab_auto_retry": True, **fields})
        dropped: list = []
        monkeypatch.setattr(store, "_drop", lambda slot, everything=False, d=dropped: d.append(slot))
        store.set_lora_mode(new_lora)
        store.set_slab_mode(new_slab)
        assert marked or not dropped, f"{kind}: dropped on {last} to {new}, and nothing marked it"


@pytest.mark.parametrize(("last", "new", "fsdp", "drops"), [
    ({}, {"slab_weights": False, "lora_low_rss": False}, False, True),  # the gate's stock policy
    ({"slab_weights": False, "lora_low_rss": False}, {}, False, True),  # and back
    ({"slab_weights": True, "lora_low_rss": True}, {"slab_weights": "auto", "lora_low_rss": True},
     False, False),                                                     # auto keeps either class
    ({"slab_weights": True, "lora_low_rss": True}, {"slab_weights": False, "lora_low_rss": True},
     True, False),                                                      # under FSDP slab is off
    ({"slab_weights": False, "lora_low_rss": False},
     {"slab_weights": False, "lora_low_rss": False, "mmap_fallback": True}, False, False),
])
def test_without_the_drop_key_a_residency_change_the_store_may_act_on_marks(
        monkeypatch, h3, last, new, fsdp, drops):
    """The inference's residency rows, for a reply without the key (``{}`` here)."""
    _stamp(monkeypatch)
    workers = _Workers([])
    workers.apply_worker_args = _Endpoint(_Future(_ValueMesh([{}, {}])))
    handle, _old, _new = _handle(workers)
    handle.active_worker_args = dict(last)
    if fsdp:
        handle.topology = SimpleNamespace(fsdp=True)
    handle.apply_worker_args(new, merge_config=False)
    assert len(workers.apply_worker_args.calls) == 1
    assert _render_memos_drained() is drops
    assert loader_preflight._CLEARED_UNETS == {UNET: None}
    _render(h3, _h3_latent())  # not an operator's drop: the preset credit stands


DROPPED = "dropped_slots"  # the reply key (residency_doors.DROPPED, pinned in test_worker_drop_report)


def _reply(*dropped: str) -> dict:
    """A Worker's reply that names the slots its apply dropped."""
    return {"lora_low_rss": False, "slab_weights": False, DROPPED: list(dropped)}


GATE_STOCK = {"slab_weights": False, "lora_low_rss": False}


@pytest.mark.parametrize(("replies", "new", "drops"), [
    pytest.param([_reply("cond"), _reply("cond")], {"mmap_fallback": True}, True,
                 id="reported, where the inference sees no drop"),
    pytest.param([_reply("cond"), _reply()], {"mmap_fallback": True}, True,
                 id="one rank's drop: the load is pending on that rank"),
    pytest.param([_reply(), _reply()], GATE_STOCK, False,
                 id="adopted, where the inference would mark"),
    pytest.param([{}, _reply()], GATE_STOCK, True, id="one reply without the key: the inference"),
    pytest.param([{}, {}], {"mmap_fallback": True}, False, id="no key, no inferred drop"),
])
def test_the_driver_marks_exactly_where_a_rank_reports_a_drop(monkeypatch, h3, replies, new, drops):
    _stamp(monkeypatch)
    workers = _Workers([])
    workers.apply_worker_args = _Endpoint(_Future(_ValueMesh(replies)))
    handle, _old, _new = _handle(workers)
    applied = handle.apply_worker_args(new, merge_config=False)
    assert applied == replies                     # the reply reaches every reader unchanged
    assert _render_memos_drained() is drops
    assert loader_preflight._CLEARED_UNETS == {UNET: None}
    _render(h3, _h3_latent())  # not an operator's drop: the preset credit stands


@pytest.mark.parametrize("inferred", [True, False])
def test_an_apply_whose_reply_is_lost_marks_where_the_inference_says(monkeypatch, h3, inferred):
    _stamp(monkeypatch)
    workers = _Workers([])
    workers.apply_worker_args = _Endpoint(_Future(error=TimeoutError("the reply did not come")))
    handle, _old, _new = _handle(workers)
    with pytest.raises(TimeoutError):
        handle.apply_worker_args(GATE_STOCK if inferred else {"mmap_fallback": True}, merge_config=False)
    assert _render_memos_drained() is inferred


@pytest.mark.parametrize(("restore", "drops"), [((_reply("cond"), _reply("cond")), True),
                                                ((_reply(), _reply()), False)])
def test_the_cross_mode_references_restore_marks_what_the_workers_dropped(
        monkeypatch, h3, restore, drops):
    """The identity gate's cross-mode leg runs its stock reference under
    ``gate_identity.temporary_worker_policy`` (slab off) and restores slab on
    leaving it. Where the Workers drop the stock reference's resident on the
    restore, the next render's checks charge it; where they keep it, they do not.
    ``gate_cross_mode``'s still-slab check reads the entry's reply as before."""
    from dgx_monarch.nodes import gate_identity

    _stamp(monkeypatch)
    workers = _Workers([])
    workers.apply_worker_args = _Endpoint(_Future(_ValueMesh([_reply(), _reply()])),
                                          _Future(_ValueMesh(list(restore))))
    handle, _old, _new = _handle(workers)
    asked = {"lora_low_rss": True, "slab_weights": True}
    handle.active_worker_args = dict(asked)
    with gate_identity.temporary_worker_policy(handle, asked, {"slab_weights": False}) as applied:
        assert not [row for row in applied if row and row.get("slab_weights")]
        assert not _render_memos_drained()           # the entry dropped nothing
        render_preflight.note_successful_render(UNET)  # the stock reference rendered
    sent = [args[0]["slab_weights"] for args, _kwargs in workers.apply_worker_args.calls]
    assert sent == [False, True]
    assert _render_memos_drained() is drops
    assert loader_preflight._CLEARED_UNETS == {UNET: None}


def test_a_renders_own_policy_through_setup_marks_a_reported_drop(monkeypatch, h3):
    """The route every render takes: ``ensure_setup`` on the live topology sends
    the request's policy where its key differs, through ``apply_worker_args``."""
    _stamp(monkeypatch)
    workers = _Workers([])
    workers.apply_worker_args = _Endpoint(_Future(_ValueMesh([_reply("cond"), _reply("cond")])))
    handle, old, _new = _handle(workers)
    handle.ensure_setup(old, "TORCH_FLASH", True, GATE_STOCK)
    assert len(workers.apply_worker_args.calls) == 1
    assert _render_memos_drained()
    assert loader_preflight._CLEARED_UNETS == {UNET: None}
    _render(h3, _h3_latent())  # not an operator's drop: the preset credit stands


def test_a_stock_resident_the_workers_adopt_keeps_the_no_render_loads_credit(
        monkeypatch, tmp_path, box):
    """The path the release test runs on every first use without a PASS: the
    no-render path loads the model under the Init node's auto widgets, the
    user's render goes out under the gate's stock policy (``ensure_setup``),
    the Workers adopt their stock resident and say so, nothing is marked, and a
    retry after that render failed still credits the DiT under ``auto``."""
    rig_ = _h3_rig(monkeypatch, tmp_path, preset="auto")
    box.mem = _h3_need(weights=0) + 4 * _GIB
    _comfy(monkeypatch, box, models=())
    model = ModelSpec(mesh=rig_.model.mesh, unet_name=UNET)
    pending_load_relief.note_unloaded()     # the no-render path: every rank unloads
    pending_load_relief.note_placed(UNET)   # and loads the model
    workers = _Workers([])
    workers.apply_worker_args = _Endpoint(_Future(_ValueMesh([_reply(), _reply()])))
    handle, old, _new = _handle(workers)
    handle.ensure_setup(old, "TORCH_FLASH", True, GATE_STOCK)
    assert len(workers.apply_worker_args.calls) == 1
    _render(model, _h3_latent())            # the retry: credited, so it admits
