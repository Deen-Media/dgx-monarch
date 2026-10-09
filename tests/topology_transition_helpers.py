"""Future, endpoint and handle doubles shared by the mesh setup, dispatch, sample-lease, eviction,
topology transition and gate-abort tests."""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

from dgx_monarch import mesh as mesh_mod
from dgx_monarch import mesh_setup
from dgx_monarch.mesh import MeshHandle
from dgx_monarch.topology import Topology


def _acks(world: int = 2, torn_down: bool = True):
    return _ValueMesh([
        {"torn_down": torn_down, "cleanup_state": "UNSETUP"}
        for _ in range(world)
    ])


class _ValueMesh:
    def __init__(self, values):
        self.values = list(values)

    def items(self):
        return list(enumerate(self.values))


class _Future:
    def __init__(self, value=None, error: BaseException | None = None):
        self.value = value
        self.error = error
        self.timeouts: list[float | None] = []

    def get(self, timeout=None):
        self.timeouts.append(timeout)
        if self.error is not None:
            raise self.error
        return self.value


class _BlockingFuture(_Future):
    def __init__(self, value, entered: threading.Event, release: threading.Event):
        super().__init__(value)
        self.entered = entered
        self.release = release

    def get(self, timeout=None):
        self.timeouts.append(timeout)
        self.entered.set()
        assert self.release.wait(timeout=2.0)
        return self.value


class _LoopRefusingFuture(_Future):
    """Fails where monarch 0.6.0 only logs its on-loop Future.get warning."""

    def get(self, timeout=None):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return super().get(timeout=timeout)
        raise AssertionError(
            "Future.get must be driven off the event loop (issue #161)")


class _Endpoint:
    def __init__(self, *futures):
        self.futures = list(futures)
        self.calls = []

    def call(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.futures.pop(0)


class _SetupEndpoint:
    def __init__(self, future):
        self.future = future
        self.envs = []

    def call_one(self, env):
        self.envs.append(dict(env))
        if isinstance(self.future, BaseException):
            raise self.future
        return self.future


class _WorkerSlice:
    def __init__(self, setup):
        self.setup = setup


class _Workers:
    def __init__(self, teardown_futures, setup_futures=None):
        self.extent = SimpleNamespace(labels=("gpus",))
        self.teardown_group = _Endpoint(*teardown_futures)
        setup_futures = setup_futures or [
            _Future({"rank": 0}), _Future({"rank": 1})
        ]
        self.setup_endpoints = [_SetupEndpoint(f) for f in setup_futures]

    def slice(self, *, gpus):
        return _WorkerSlice(self.setup_endpoints[gpus])


class _Procs:
    def __init__(self):
        self.reasons = []

    def stop(self, reason):
        self.reasons.append(reason)
        return _Future(None)


class _ObservedLock:
    def __init__(self):
        self.inner = threading.RLock()
        self.attempted = threading.Event()

    def acquire(self, *args, **kwargs):
        self.attempted.set()
        return self.inner.acquire(*args, **kwargs)

    def release(self):
        return self.inner.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_args):
        self.release()

    def _is_owned(self):
        return self.inner._is_owned()


def _handle(workers: _Workers) -> tuple[MeshHandle, Topology, Topology]:
    config = SimpleNamespace(
        worker_args={},
        nccl_master_port=29500,
        rdma_latent_return=False,
        rdma_min_bytes=1024,
        source="local",
        hosts=(),
        comfy_dir="/comfy",
        resolved_fabric_env=lambda: {},
        resolved_master_addr=lambda: "127.0.0.1",
    )
    old = Topology(ulysses=2, world=2)
    new = Topology(ring=2, world=2)
    handle = MeshHandle(
        config=config,
        hosts=None,
        procs=_Procs(),
        workers=workers,
        world=2,
        gpus_per_host=2,
        n_hosts=1,
        comfy_dir="/comfy",
        owns_hosts=False,
        setup_key=mesh_mod._nccl_setup_key(old, "TORCH_FLASH", True),
        worker_args_key=MeshHandle._worker_args_key({}),
        topology=old,
        setup_generation=1,
    )
    return handle, old, new


def _dispatch_sample(handle, send, *, setup_token, authority=None):
    """Tests prepare the same caller-owned authority production requires."""
    if authority is None and isinstance(handle, MeshHandle):
        authority = mesh_setup.prepare_sample(handle, setup_token)
    return mesh_setup.dispatch_sample(
        handle, send, setup_token=setup_token, authority=authority)
