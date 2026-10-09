"""A reused rendezvous port must not expose an earlier generation's keys.

torch's tcp:// rendezvous shares one TCPStore server per port inside a process
(multi_tenant) and that server outlives destroy_process_group, while process-group
names restart after a destroy. The driver's master port repeats every 16 setup
generations, so without a per-generation namespace a new Gloo group could read an
old group's address, connect to a closed port and be refused. CPU only, world 1.
"""
import socket

import torch.distributed as dist

from dgx_monarch.actor.rendezvous import generation_store


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _env(port: int) -> dict:
    return {"master_addr": "127.0.0.1", "master_port": port, "nccl_timeout_s": 20}


def test_shared_server_keeps_keys_for_a_later_store_on_the_same_port():
    """The torch behaviour the prefix defends against: a second multi-tenant store on
    the same port in the same process reaches the first one's server and keys. A torch
    that changes this fails here and the prefix can be reconsidered."""
    from datetime import timedelta

    port = _free_port()
    first = dist.TCPStore("127.0.0.1", port, 1, True, timeout=timedelta(seconds=20), multi_tenant=True)
    first.set("gloo_address", "old")
    second = dist.TCPStore("127.0.0.1", port, 1, True, timeout=timedelta(seconds=20), multi_tenant=True)
    assert second.get("gloo_address") == b"old"
    del first, second


def test_generation_store_hides_an_earlier_generation_on_the_same_port():
    port = _free_port()
    dist.init_process_group("gloo", rank=0, world_size=1, store=generation_store(_env(port), 0, 1, 7))
    try:
        dist.distributed_c10d._get_default_store().set("gloo_address", "old")
    finally:
        dist.destroy_process_group()
    dist.init_process_group("gloo", rank=0, world_size=1, store=generation_store(_env(port), 0, 1, 8))
    try:
        store = dist.distributed_c10d._get_default_store()
        assert store.check(["gloo_address"]) is False
        store.set("gloo_address", "new")
        assert store.get("gloo_address") == b"new"
        dist.new_group([0], backend="gloo")  # a group still forms on the prefixed store
    finally:
        dist.destroy_process_group()


def test_setup_hands_init_process_group_the_generation_store(monkeypatch, isolated_environ):
    import sys
    import types
    from types import SimpleNamespace

    import pytest
    import torch

    from dgx_monarch.actor import comfy_bridge, worker_env

    class StopNow(BaseException):
        pass

    calls = {}
    monkeypatch.setattr(worker_env, "ensure_comfy", lambda *args, **kwargs: None)
    monkeypatch.setattr(comfy_bridge, "_uma_memory_defaults", lambda args: dict(args))
    monkeypatch.setattr("dgx_monarch.config.fixup_fabric_ifaces", lambda env: (env, ""))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda *a, **k: calls.setdefault("pg", k))
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda: None)
    monkeypatch.setattr(worker_env.rendezvous, "generation_store",
                        lambda env, rank, world, generation: ("store", rank, world, generation))
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
        raise StopNow("stop after the process group")

    worker = SimpleNamespace(
        _setup_key=None, _setup_cleanup_failed=False, _latent_return=None,
        rank=None, world=None, topology={}, _attn=SimpleNamespace(configure=stop),
        _teardown_parallel_state=lambda: None, _slab_mode_effective=lambda *_args: False,
        _nccl_launch_order_needed=lambda _topo: False,
        store=SimpleNamespace(family_override=None, set_lora_mode=lambda *_a: None,
                              set_slab_mode=lambda *_a: None, swap_verify=2),
    )
    env = {"setup_generation": 17, "world": 2, "rank": 1, "master_addr": "127.0.0.1",
           "master_port": 23471, "topology": {"ulysses": 2, "ring": 1, "cfg": 1, "dp": 1, "fsdp": False},
           "fabric_env": {}, "comfy_dir": "/comfy", "worker_args": {}, "local_gpu_index": 0}
    with pytest.raises(StopNow):
        worker_env.setup_impl(worker, env)
    assert calls["pg"]["store"] == ("store", 1, 2, 17)
    assert "init_method" not in calls["pg"]
