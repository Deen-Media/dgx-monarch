"""A Worker's apply_worker_args reply names the model slots that apply really dropped.

The driver marks the load still to come exactly where a rank reports a drop
(``mesh_setup_transition.reported_drop``), so the reply must name every slot
the store discarded and none it kept. This drives the real
``actor/worker_env.apply_worker_args_impl`` over a real ``ModelStore``: its
residency setters, its ``unload_all`` and its keep rule run as shipped; only
``_drop``'s ComfyUI cleanup is stubbed, to what a successful drop leaves (an
empty slot), and the unified-memory defaults are read as a GB10 reads them.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from dgx_monarch import fp8_compute
from dgx_monarch.adapters import quant_activation_scale

# The reply's key, spelt out so the wire name is pinned where the driver and the
# Workers meet (one build ships both).
DROPPED = "dropped_slots"


def test_the_reply_key_is_the_one_the_driver_reads():
    from dgx_monarch import residency_doors

    assert residency_doors.DROPPED == DROPPED


def _resident(**fields) -> SimpleNamespace:
    return SimpleNamespace(**{
        "slab": None, "residency_rung": "", "lora_sig": "", "base_baked": False,
        "base_key": ("model.safetensors",), "slab_auto_retry": True,
        "base_patcher": None, "active_patcher": None, **fields})


SLAB = {"slab": SimpleNamespace(certificate=None)}
STOCK: dict = {}
UNBAKED_LORA = {"lora_sig": "lora"}
BAKED_LORA = {"lora_sig": "lora", "base_baked": True}


@pytest.fixture
def worker(monkeypatch):
    """A Worker as apply_worker_args_impl reads it, after an apply of ``last``."""
    from dgx_monarch.actor import comfy_bridge, comfy_dynamic, model_store, worker_env

    monkeypatch.setattr(comfy_bridge, "_CUSTOM_NODES_DISABLED", None, raising=False)
    monkeypatch.setattr(comfy_bridge, "_apply_worker_args", lambda _wa: None)
    defaults = comfy_bridge._uma_memory_defaults
    monkeypatch.setattr(comfy_bridge, "_uma_memory_defaults", lambda wa: defaults(wa, integrated=True))
    monkeypatch.setattr(comfy_dynamic, "_REQUESTED", None, raising=False)
    for name in ("DGXM_COMPILE_DIT", "DGXM_LOAD_PROFILE", quant_activation_scale.ENABLE_ENV,
                 quant_activation_scale.LOG_ENV):
        monkeypatch.setenv(name, "placeholder")  # so the test restores what the apply writes
        monkeypatch.delenv(name)
    monkeypatch.setattr(fp8_compute._STATE, "mode", fp8_compute.DEFAULT)

    def settle(last: dict, resident: dict) -> SimpleNamespace:
        """One Worker with its own store, its resident loaded under ``last``."""
        store = model_store.ModelStore()
        monkeypatch.setattr(store, "_drop", lambda slot, everything=False: setattr(
            store, "uncond" if slot == "uncond" else "current", None))
        monkeypatch.setattr(store, "release_retained_cleanup", lambda: None)
        rig = SimpleNamespace(store=store, topology={"ulysses": 2, "fsdp": False}, uma_reserve_gb=0.0,
                              _slab_mode_effective=worker_env.slab_mode_effective,
                              _active_worker_args={})
        worker_env.apply_worker_args_impl(rig, last)   # the policy the resident loaded under
        rig._active_worker_args = dict(last)
        store.current = _resident(**resident)
        return rig

    return settle


def _apply(rig, new: dict) -> list:
    from dgx_monarch.actor import worker_env

    reply = worker_env.apply_worker_args_impl(rig, new)
    assert set(reply) >= {"lora_low_rss", "slab_weights"}  # the keys every reader reads stay
    return reply.get(DROPPED)


AUTO: dict = {}                                            # the Init node's auto widgets
GATE_STOCK = {"slab_weights": False, "lora_low_rss": False}  # the gate's stock policy
SLAB_ON = {"slab_weights": True, "lora_low_rss": True}


@pytest.mark.parametrize(("last", "resident", "new"), [
    pytest.param(SLAB_ON, SLAB, {**SLAB_ON, "slab_weights": False}, id="slab resident under slab off"),
    pytest.param(GATE_STOCK, STOCK, SLAB_ON, id="stock resident under slab on"),
    pytest.param({"slab_weights": False, "lora_low_rss": True}, UNBAKED_LORA, GATE_STOCK,
                 id="unbaked LoRA patches on a LoRA mode flip"),
    pytest.param(AUTO, STOCK, {"compile_dit": True}, id="compile_dit"),
    pytest.param(AUTO, STOCK, {"family_override": "wan"}, id="the forced family"),
    pytest.param(AUTO, STOCK, {"fp8_matrix_mult": "on"}, id="the FP8 matmul setting"),
    pytest.param(AUTO, STOCK, {"shared_act_scale": False}, id="the shared activation scale"),
])
def test_the_reply_names_the_slot_the_apply_dropped(worker, last, resident, new):
    rig = worker(last, resident)
    assert _apply(rig, new) == ["cond"]
    assert rig.store.current is None


@pytest.mark.parametrize(("last", "resident", "new"), [
    pytest.param(AUTO, STOCK, GATE_STOCK,
                 id="a stock resident adopted under the gate's stock policy"),
    pytest.param(AUTO, SLAB, SLAB_ON, id="a slab resident under slab on"),
    pytest.param({"slab_weights": False, "lora_low_rss": True}, BAKED_LORA, GATE_STOCK,
                 id="a baked LoRA resident on a LoRA mode flip"),
    pytest.param(AUTO, STOCK, {"mmap_fallback": True}, id="a change no setter acts on"),
])
def test_the_reply_names_no_slot_the_worker_kept(worker, last, resident, new):
    rig = worker(last, resident)
    held = rig.store.current
    assert _apply(rig, new) == []
    assert rig.store.current is held


def test_an_empty_store_reports_no_drop(worker):
    rig = worker(AUTO, STOCK)
    rig.store.current = None
    assert _apply(rig, {"compile_dit": True}) == []


@pytest.mark.parametrize(("last", "resident", "new", "marked"), [
    pytest.param(AUTO, STOCK, GATE_STOCK, False, id="the Workers adopt their stock resident"),
    pytest.param(SLAB_ON, SLAB, {**SLAB_ON, "slab_weights": False}, True,
                 id="the Workers drop their slab resident"),
])
def test_the_driver_marks_what_the_real_worker_apply_reports(worker, last, resident, new, marked):
    """Both halves: the driver's real ``MeshHandle.apply_worker_args`` sends the
    policy, and its endpoint runs the real ``apply_worker_args_impl`` on two
    Workers for exactly the args the driver sent."""
    from dgx_monarch.actor import worker_env
    from dgx_monarch.nodes import pending_load_relief, render_preflight
    from topology_transition_helpers import _Future, _handle, _ValueMesh, _Workers

    ranks = [worker(last, resident), worker(last, resident)]
    sent: list = []

    class _RealWorkers:
        def call(self, effective):
            sent.append(effective)
            return _Future(_ValueMesh([worker_env.apply_worker_args_impl(rig, dict(effective))
                                       for rig in ranks]))

    workers = _Workers([])
    workers.apply_worker_args = _RealWorkers()
    handle, _old, _new = _handle(workers)
    handle.active_worker_args = dict(last)
    pending_load_relief.note_placed("model.safetensors")
    render_preflight.note_successful_render("model.safetensors")
    replies = handle.apply_worker_args(new, merge_config=False)
    assert sent == [new]
    assert [reply[DROPPED] for reply in replies] == ([["cond"]] * 2 if marked else [[]] * 2)
    assert pending_load_relief.dit_resident("model.safetensors", charged=True) is not marked
