"""GPUWorker decision helpers: the slab-mode gate, the NCCL launch-order guard
that keeps FSDP and a second communicator from deadlocking (docs/VALIDATION.md,
2026-07-04), the refusals in _inject_for_topology, and source-text guards on
actor/sample_protocol.py. No CUDA and no monarch mesh."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch.actor import worker as worker_mod
from dgx_monarch.actor.worker import GPUWorker
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.pixeldit import Ideogram4Adapter

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("wa,topo,expected", [
    ({}, {}, False),                                        # off by default
    ({"slab_weights": True}, {}, True),                     # plain on
    ({"slab_weights": True}, {"fsdp": True}, False),        # DTensors reshard
    ({"slab_weights": True, "compile_dit": True}, {}, True),  # resolved per loaded family
    ({"slab_weights": False, "compile_dit": True}, {}, False),
    ({"slab_weights": True}, {"fsdp": False}, True),
    ({"slab_weights": "auto"}, {}, "auto"),                 # defer to load-time family
    ({"slab_weights": "auto"}, {"fsdp": True}, False),      # FSDP kills auto too
    ({"slab_weights": "auto", "compile_dit": True}, {}, "auto"),
])
def test_slab_mode_effective(wa, topo, expected):
    assert GPUWorker._slab_mode_effective(wa, topo) == expected


def test_slab_compile_guard_logs_only_when_compile_is_requested(monkeypatch):
    from dgx_monarch.actor import partial_load_guard

    events = []
    adapter = SimpleNamespace(family="krea2", dual_model_cfg_supported=False,
                              cfg_parallel_supported=True)
    monkeypatch.setattr(worker_mod.store_fsdp, "validate_injection", lambda *_args: None)
    monkeypatch.setattr(partial_load_guard, "install", lambda *_args: None)
    monkeypatch.setattr("dgx_monarch.adapters.get_adapter", lambda *_args: adapter)
    monkeypatch.setattr("dgx_monarch.adapters.quant_activation_scale.install", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker_mod, "_maybe_compile_dit", lambda *_args: events.append("compile"))
    log = SimpleNamespace(info=lambda message, *_args: events.append(message))
    monkeypatch.setattr(worker_mod, "log", log)
    worker = SimpleNamespace(topology={"cfg": 1, "ulysses": 1, "ring": 1, "dp": 1},
                              world=1, store=SimpleNamespace(family_override=None))
    patcher = SimpleNamespace(_dgxm_slab_resident=True,
                              model=SimpleNamespace(diffusion_model=object()), model_options={})

    monkeypatch.delenv("DGXM_COMPILE_DIT", raising=False)
    GPUWorker._inject_for_topology(worker, patcher, "bf16", [], None)
    assert events == []

    monkeypatch.setenv("DGXM_COMPILE_DIT", "1")
    GPUWorker._inject_for_topology(worker, patcher, "bf16", [], None)
    assert events == ["compile_dit: slab-resident model stays eager"]


@pytest.mark.parametrize("topo,expected", [
    ({"fsdp": True, "ulysses": 2}, True),    # fsdp + SP: two communicators
    ({"fsdp": True, "ring": 2}, True),
    ({"fsdp": True, "cfg": 2}, True),
    ({"fsdp": True}, False),                 # fsdp alone: one communicator
    ({"fsdp": True, "ulysses": 1, "ring": 1, "cfg": 1}, False),
    ({"ulysses": 2}, False),                 # SP alone: no fsdp all-gather
    ({}, False),
    ({"fsdp": False, "ulysses": 2}, False),
])
def test_nccl_launch_order_needed(topo, expected):
    assert GPUWorker._nccl_launch_order_needed(topo) is expected


def test_setup_sets_the_launch_order_guard_before_every_communicator(monkeypatch, isolated_environ):
    """NCCL reads NCCL_LAUNCH_ORDER_IMPLICIT once, at a process's first launch, and
    keeps it. If setup cleared it for plain topologies, a worker first set up plain
    (cfg2, uly2) would run a later uly2+fsdp setup unguarded, as it did before
    2026-09-28. Setup must set it for every topology, before any communicator
    exists, and never clear it."""
    import sys
    import types

    import torch

    from dgx_monarch.actor import comfy_bridge, worker_env

    class StopNow(BaseException):
        pass

    seen, retired = [], []
    monkeypatch.delenv("NCCL_LAUNCH_ORDER_IMPLICIT", raising=False)
    monkeypatch.setattr(worker_env, "ensure_comfy", lambda *args, **kwargs: None)
    monkeypatch.setattr(comfy_bridge, "_uma_memory_defaults", lambda args: dict(args))
    monkeypatch.setattr("dgx_monarch.config.fixup_fabric_ifaces", lambda env: (env, ""))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "init_process_group",
                        lambda *a, **k: seen.append(worker_env.os.environ.get("NCCL_LAUNCH_ORDER_IMPLICIT")))
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda: None)
    monkeypatch.setattr(worker_env.rendezvous, "generation_store", lambda *_a, **_k: None)
    retire = worker_env.teardown_existing_setup
    monkeypatch.setattr(worker_env, "teardown_existing_setup",
                        lambda w: retired.append(dict(w._setup_key[5])["fsdp"]) or retire(w))
    distributed = types.ModuleType("xfuser.core.distributed")
    distributed.init_distributed_environment = lambda **kwargs: None
    distributed.initialize_model_parallel = lambda **kwargs: None
    core = types.ModuleType("xfuser.core")
    core.distributed = distributed
    xfuser = types.ModuleType("xfuser")
    xfuser.core = core
    monkeypatch.setitem(sys.modules, "xfuser", xfuser)
    monkeypatch.setitem(sys.modules, "xfuser.core", core)
    monkeypatch.setitem(sys.modules, "xfuser.core.distributed", distributed)

    def stop(*_args, **_kwargs):
        raise StopNow("stop after the communicators")

    worker = SimpleNamespace(
        _setup_key=None, _setup_cleanup_failed=False, _latent_return=None,
        rank=None, world=None, topology={}, _attn=SimpleNamespace(configure=stop),
        _teardown_parallel_state=lambda: None,
        _slab_mode_effective=lambda *_args: False,
        _nccl_launch_order_needed=GPUWorker._nccl_launch_order_needed,
        store=SimpleNamespace(family_override=None, set_lora_mode=lambda *_a: None,
                              set_slab_mode=lambda *_a: None, swap_verify=2),
    )
    plain = {"ulysses": 1, "ring": 1, "cfg": 2, "dp": 1, "fsdp": False}
    fsdp_sp = {"ulysses": 2, "ring": 1, "cfg": 1, "dp": 1, "fsdp": True}
    for generation, topology in enumerate((plain, fsdp_sp, plain), start=1):
        env = {"setup_generation": generation, "world": 2, "rank": 0,
               "master_addr": "127.0.0.1", "master_port": 23460 + generation,
               "topology": topology, "fabric_env": {}, "comfy_dir": "/comfy",
               "worker_args": {}, "local_gpu_index": 0}
        with pytest.raises(StopNow):
            worker_env.setup_impl(worker, env)
        # Publish what a completed setup publishes, so the next topology meets
        # a live setup on this same worker and retires it first.
        worker._setup_key = worker_env._setup_key(env, {})
        worker._setup_cleanup_failed = False
    assert retired == [False, True]  # plain retired for fsdp+SP, then fsdp+SP for plain
    assert seen == ["1", "1", "1"]
    assert worker_env.os.environ.get("NCCL_LAUNCH_ORDER_IMPLICIT") == "1"


@pytest.mark.parametrize(
    "topology",
    [
        {"ulysses": 1, "ring": 1, "cfg": 2},
        {"ulysses": 1, "ring": 2, "cfg": 2},
    ],
    ids=["pure-cfg-pad-path", "hybrid-usp-path"],
)
def test_cfg_unsupported_family_refuses_before_usp_compile_or_wrapper(
    monkeypatch, topology
):
    events = []
    adapter = Ideogram4Adapter()
    adapter.inject_usp = lambda *_args: events.append("usp")
    adapter.inject_cfg_pad_forward = lambda *_args: events.append("cfg-pad")

    def get_adapter(_model):
        events.append("get-adapter")
        return adapter

    monkeypatch.setattr("dgx_monarch.adapters.get_adapter", get_adapter)
    monkeypatch.setattr(worker_mod.store_fsdp, "validate_injection", lambda *_args: None)
    monkeypatch.setattr(
        worker_mod, "_maybe_compile_dit", lambda _model: events.append("compile")
    )
    patcher = SimpleNamespace(
        model=SimpleNamespace(diffusion_model=object()), model_options={}
    )
    worker = SimpleNamespace(topology=topology, _attn=object())

    with pytest.raises(UnsupportedModelError, match="topology 'auto'"):
        GPUWorker._inject_for_topology(worker, patcher, "bf16", [], None)

    assert events == ["get-adapter"]


# _sample_impl needs a GPU, a mesh and store_fsdp, so it has no CPU end-to-end
# test; this source-text guard fails if an edit drops the dp_cond_exempt_keys
# wiring or wires only one sampler entry point.
def test_sample_impl_wires_dp_cond_exempt_keys_to_both_sampler_entry_points():
    source = (REPO / "src" / "dgx_monarch" / "actor" / "sample_protocol.py").read_text()
    assert source.count("patcher.model, family_override).dp_cond_exempt_keys") == 1
    assert source.count("dp_cond_exempt_keys=dp_cond_exempt_keys") >= 2


def test_sample_impl_reads_the_adapter_through_the_one_family_decision():
    """Both consultations here take the forced family, or a graph running a
    finetune would raise the override's own refusal mid-render, on a model
    whose forward the injection already patched."""
    source = (REPO / "src" / "dgx_monarch" / "actor" / "sample_protocol.py").read_text()
    assert "get_adapter(" not in source
    assert source.count("adapter_for(patcher.model, family_override)") == 1
    assert source.count("patcher.model, family_override).dp_cond_exempt_keys") == 1
    assert "family_override = override_for_worker(worker)" in source
