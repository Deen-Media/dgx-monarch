"""GPUWorker setup generations, the fabric environment, and live policy changes."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from dgx_monarch.actor import worker_env
from dgx_monarch.config_schema import ClusterConfigError

# setup_impl and apply_worker_args_impl write a worker's own environment: the
# CUDA device, the fabric variables, the NCCL launch order, and the compile,
# load-profile and activation-scale switches. The tests here run both in this
# process, so each one hands back the environment it found.
pytestmark = pytest.mark.usefixtures("isolated_environ")


def _replacement_env(**overrides) -> dict:
    env = {
        "setup_generation": 2,
        "world": 2,
        "rank": 0,
        "gpus_per_host": 1,
        "local_gpu_index": 0,
        "master_addr": "192.0.2.1",
        "master_port": 29777,
        "topology": {"dp": 1, "ulysses": 2, "ring": 1, "cfg": 1},
        "fabric_env": {},
        "comfy_dir": "/unused",
        "worker_args": {},
    }
    env.update(overrides)
    return env


def test_setup_key_change_retires_old_generation_before_bootstrap(monkeypatch):
    events: list[str] = []
    worker = SimpleNamespace(
        _setup_key=(1, 2, "192.0.2.1", 29777, (("ulysses", 2),)),
        _setup_generation=1,
        _setup_provenance=(1, {}),
        _setup_cleanup_failed=False,
        _active_worker_args={"slab_weights": True},
        rank=0,
        world=2,
        topology={"ulysses": 2},
        _teardown_parallel_state=lambda: events.append("teardown"),
    )

    def fail_bootstrap(*_args, **_kwargs):
        events.append("ensure_comfy")
        raise RuntimeError("bootstrap refused")

    monkeypatch.setattr(worker_env, "ensure_comfy", fail_bootstrap)

    with pytest.raises(RuntimeError, match="bootstrap refused"):
        worker_env.setup_impl(worker, _replacement_env())

    assert events == ["teardown", "ensure_comfy"]
    assert worker._setup_key is None
    assert worker._setup_generation is None
    assert worker._setup_provenance is None
    assert worker.rank is None and worker.world is None and worker.topology == {}
    assert worker._setup_cleanup_failed is True


def test_setup_key_change_settles_old_teardown_before_new_environment(monkeypatch):
    worker = SimpleNamespace(
        _setup_key=(1, "ready"),
        _setup_generation=1,
        _setup_provenance=(1, {}),
        _setup_cleanup_failed=False,
        rank=0,
        world=2,
        topology={"ulysses": 2},
        _teardown_parallel_state=lambda: (_ for _ in ()).throw(
            RuntimeError("old teardown failed")
        ),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "old-device")
    monkeypatch.setenv("NCCL_DEBUG", "old-debug")
    monkeypatch.setattr(
        worker_env,
        "ensure_comfy",
        lambda *_args, **_kwargs: pytest.fail("bootstrap must follow settled teardown"),
    )

    with pytest.raises(RuntimeError, match="old teardown failed"):
        worker_env.setup_impl(
            worker,
            _replacement_env(fabric_env={"NCCL_DEBUG": "INFO"}),
        )

    assert worker_env.os.environ["CUDA_VISIBLE_DEVICES"] == "old-device"
    assert worker_env.os.environ["NCCL_DEBUG"] == "old-debug"
    assert worker._setup_key is None
    assert worker._setup_cleanup_failed is True


def test_actor_setup_refuses_nccl_proto_before_environment_mutation(monkeypatch):
    teardown_calls: list[str] = []
    worker = SimpleNamespace(
        _setup_key=(1, "ready"),
        _setup_cleanup_failed=False,
        _teardown_parallel_state=lambda: teardown_calls.append("teardown"),
    )
    monkeypatch.delenv("NCCL_PROTO", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "old-device")

    with pytest.raises(ClusterConfigError, match=r"NCCL_PROTO.*unset"):
        worker_env.setup_impl(
            worker,
            _replacement_env(world=1, rank=0, fabric_env={"NCCL_PROTO": "LL"}),
        )

    assert "NCCL_PROTO" not in worker_env.os.environ
    assert worker_env.os.environ["CUDA_VISIBLE_DEVICES"] == "old-device"
    assert teardown_calls == []
    assert worker._setup_key == (1, "ready")
    assert worker._setup_cleanup_failed is False


def test_same_generation_attention_change_retires_the_old_dispatcher(monkeypatch):
    """Attention and sync state close over the current process groups.

    A retry that changes either policy cannot return early because its
    generation and topology still match; it must retire the dispatcher before
    any new bootstrap mutation.
    """
    events: list[str] = []
    original = _replacement_env()
    worker = SimpleNamespace(
        _setup_key=worker_env._setup_key(original, {}),
        _setup_generation=2,
        _setup_provenance=None,
        _setup_cleanup_failed=False,
        _active_worker_args={},
        rank=0,
        world=2,
        topology=dict(original["topology"]),
        _teardown_parallel_state=lambda: events.append("teardown"),
    )

    def stop_after_retirement(*_args, **_kwargs):
        events.append("bootstrap")
        raise RuntimeError("stop after old dispatcher retirement")

    monkeypatch.setattr(worker_env, "ensure_comfy", stop_after_retirement)
    with pytest.raises(RuntimeError, match="old dispatcher retirement"):
        worker_env.setup_impl(worker, _replacement_env(attention="SAGE"))

    assert events == ["teardown", "bootstrap"]
    assert worker._setup_key is None


def test_exact_setup_retry_reuses_dispatcher_but_not_changed_runtime_state():
    env = _replacement_env(
        attention="TORCH_FLASH", sync_ulysses=True,
        rdma_latent_return=False, pipeline_depth=1,
    )
    assert worker_env._setup_key(env, {}) == worker_env._setup_key(dict(env), {})
    assert worker_env._setup_key(env, {}) != worker_env._setup_key(
        _replacement_env(sync_ulysses=False), {})
    assert worker_env._setup_key(env, {}) != worker_env._setup_key(
        _replacement_env(rank=1), {})
    assert worker_env._setup_key(env, {}) != worker_env._setup_key(
        _replacement_env(rdma_latent_return=True), {})
    assert worker_env._setup_key(env, {}) != worker_env._setup_key(
        env, {"NCCL_SOCKET_IFNAME": "ib0"})
    # Worker policy must stay out of the group and dispatcher key:
    # apply_worker_args_impl applies it live, without a new setup.
    assert worker_env._setup_key(env, {}) == worker_env._setup_key(
        _replacement_env(worker_args={"slab_weights": True}), {})


def _policy_worker(store) -> SimpleNamespace:
    return SimpleNamespace(
        store=store,
        topology={"ulysses": 2, "ring": 1, "cfg": 1, "dp": 1, "fsdp": False},
        uma_reserve_gb=0.0,
        _slab_mode_effective=worker_env.slab_mode_effective,
        _active_worker_args={},
    )


class _PolicyStore:
    """The narrow model-store surface a live policy application touches."""

    def __init__(self) -> None:
        self.lora_low_rss = True
        self.slab_weights: bool | str = "auto"
        self.swap_verify = 2
        self.family_override: str | None = None
        self.events: list[str] = []

    def set_lora_mode(self, low_rss: bool) -> None:
        self.lora_low_rss = bool(low_rss)

    def set_slab_mode(self, slab_weights, reason: str = "") -> None:
        self.slab_weights = slab_weights

    def unload_all(self) -> None:
        self.events.append("unload_all")

    def release_retained_cleanup(self) -> None:
        self.events.append("release")


def test_a_forced_stock_policy_leaves_resident_eviction_to_the_setters(monkeypatch):
    """The dispatch fallback re-applies policy inside a live render. A blanket
    unload there reloads the checkpoint the graph-prep bind already made
    resident; on 2026-08-12 each box loaded the same 39.1 GiB LTX 2.5
    transformer twice this way, unified memory did not return the first copy
    in time, and the sibling worker died (docs/VALIDATION.md, LTX 2.5 promotion
    evidence)."""
    from dgx_monarch.actor import comfy_bridge

    monkeypatch.setattr(comfy_bridge, "_CUSTOM_NODES_DISABLED", None, raising=False)
    monkeypatch.setattr(comfy_bridge, "_apply_worker_args", lambda _wa: None)
    monkeypatch.setattr(comfy_bridge, "_uma_memory_defaults", dict)
    monkeypatch.delenv("DGXM_COMPILE_DIT", raising=False)
    store = _PolicyStore()
    worker = _policy_worker(store)

    result = worker_env.apply_worker_args_impl(
        worker, {"slab_weights": False, "lora_low_rss": False})

    assert store.events == ["release"]
    assert result == {"lora_low_rss": False, "slab_weights": False}


def test_a_compile_policy_change_still_evicts_every_resident(monkeypatch):
    """Compilation is applied to the diffusion model at load, so a resident
    built under the previous compile policy cannot serve this one."""
    from dgx_monarch.actor import comfy_bridge

    monkeypatch.setattr(comfy_bridge, "_CUSTOM_NODES_DISABLED", None, raising=False)
    monkeypatch.setattr(comfy_bridge, "_apply_worker_args", lambda _wa: None)
    monkeypatch.setattr(comfy_bridge, "_uma_memory_defaults", dict)
    monkeypatch.delenv("DGXM_COMPILE_DIT", raising=False)
    store = _PolicyStore()
    worker = _policy_worker(store)

    worker_env.apply_worker_args_impl(worker, {"compile_dit": True})

    assert store.events == ["unload_all"]
    assert os.environ["DGXM_COMPILE_DIT"] == "1"


def test_a_shared_activation_scale_change_evicts_every_resident(monkeypatch):
    """The hook wraps forwards at load, so a resident built under the previous
    setting cannot serve this one. The worker arg carries the setting to every
    box, because it decides whether a rank issues a collective."""
    from dgx_monarch.actor import comfy_bridge

    monkeypatch.setattr(comfy_bridge, "_CUSTOM_NODES_DISABLED", None, raising=False)
    monkeypatch.setattr(comfy_bridge, "_apply_worker_args", lambda _wa: None)
    monkeypatch.setattr(comfy_bridge, "_uma_memory_defaults", dict)
    monkeypatch.delenv("DGXM_COMPILE_DIT", raising=False)
    monkeypatch.setenv("DGXM_SHARED_ACT_SCALE", "1")
    store = _PolicyStore()
    worker = _policy_worker(store)

    worker_env.apply_worker_args_impl(worker, {"shared_act_scale": False})

    assert os.environ["DGXM_SHARED_ACT_SCALE"] == "0"
    assert store.events == ["unload_all"]

    # The same setting again is not a change, so it keeps every resident.
    store.events.clear()
    worker_env.apply_worker_args_impl(worker, {"shared_act_scale": False})
    assert store.events == ["release"]

    # Removing the key restores the default, which is a change and evicts.
    store.events.clear()
    worker_env.apply_worker_args_impl(worker, {})
    assert "DGXM_SHARED_ACT_SCALE" not in os.environ
    assert store.events == ["unload_all"]


def test_an_absent_shared_activation_scale_key_leaves_a_hand_set_variable(
        monkeypatch):
    """The bare variable is a single-box aid; an omitted key must not pop it."""
    from dgx_monarch.actor import comfy_bridge

    monkeypatch.setattr(comfy_bridge, "_CUSTOM_NODES_DISABLED", None, raising=False)
    monkeypatch.setattr(comfy_bridge, "_apply_worker_args", lambda _wa: None)
    monkeypatch.setattr(comfy_bridge, "_uma_memory_defaults", dict)
    monkeypatch.delenv("DGXM_COMPILE_DIT", raising=False)
    monkeypatch.setenv("DGXM_SHARED_ACT_SCALE", "0")
    store = _PolicyStore()
    worker = _policy_worker(store)

    worker_env.apply_worker_args_impl(worker, {"lora_low_rss": False})

    assert os.environ["DGXM_SHARED_ACT_SCALE"] == "0"
    assert store.events == ["release"]


def test_a_family_override_change_evicts_every_resident(monkeypatch):
    """The family adapter binds once per load, so a resident built under the
    previous family cannot serve a different one."""
    from dgx_monarch.actor import comfy_bridge

    monkeypatch.setattr(comfy_bridge, "_CUSTOM_NODES_DISABLED", None, raising=False)
    monkeypatch.setattr(comfy_bridge, "_apply_worker_args", lambda _wa: None)
    monkeypatch.setattr(comfy_bridge, "_uma_memory_defaults", dict)
    monkeypatch.delenv("DGXM_COMPILE_DIT", raising=False)
    store = _PolicyStore()
    worker = _policy_worker(store)

    worker_env.apply_worker_args_impl(worker, {"family_override": "krea2"})

    assert store.family_override == "krea2"
    assert store.events == ["unload_all"]


def test_an_unchanged_family_override_keeps_every_resident(monkeypatch):
    """Re-applying the same policy inside a live render must not reload the
    checkpoint the graph-prep bind already made resident."""
    from dgx_monarch.actor import comfy_bridge

    monkeypatch.setattr(comfy_bridge, "_CUSTOM_NODES_DISABLED", None, raising=False)
    monkeypatch.setattr(comfy_bridge, "_apply_worker_args", lambda _wa: None)
    monkeypatch.setattr(comfy_bridge, "_uma_memory_defaults", dict)
    monkeypatch.delenv("DGXM_COMPILE_DIT", raising=False)
    store = _PolicyStore()
    store.family_override = "krea2"
    worker = _policy_worker(store)

    worker_env.apply_worker_args_impl(worker, {"family_override": "krea2"})

    assert store.events == ["release"]


def test_setup_hands_the_store_the_reserve_the_operator_configured(monkeypatch):
    """Setup copies the operator's reserve to the store on every fresh mesh.

    The capacity quote charges ``uma_reserve_gb`` from the worker and the slab
    wall charges it from the store. Without the copy, a host reserving 10 GiB
    is quoted against 10 GiB but admitted against the slab wall's default floor
    (``capacity_fit.slab_floor_bytes``), so the second wall admits what the
    first refused. test_store_slab_routing.py checks the live re-apply path;
    this test checks setup, whose value a first load reads.
    """
    from dgx_monarch.actor import comfy_bridge
    from dgx_monarch.actor.model_store import ModelStore

    store = ModelStore()
    worker = SimpleNamespace(
        store=store,
        _setup_key=None,
        _setup_cleanup_failed=False,
        _slab_mode_effective=worker_env.slab_mode_effective,
        _teardown_parallel_state=lambda: None,
    )
    monkeypatch.setattr(worker_env, "ensure_comfy", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(comfy_bridge, "_uma_memory_defaults", dict)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    # Stop the setup at the torch import, which follows the store policy block
    # and precedes every process group, so none is built here.
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(ImportError, match="torch"):
        worker_env.setup_impl(
            worker,
            _replacement_env(worker_args={"uma_reserve_gb": 10.0}),
        )

    assert worker.uma_reserve_gb == 10.0
    assert store.uma_reserve_gb == 10.0

