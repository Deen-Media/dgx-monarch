"""Processes for test_worker_proc_id_integration.py: a real Worker loop and a driver.

``host ADDRESS LOG_SUFFIX`` runs ``dgx_monarch.cli.worker_loop.main`` itself.
Its environment filter drops ``MONARCH_LOG_SUFFIX``, which would send this
test's Monarch log lines into the per-user log file that real loops on the
same machine write, so the wrapper puts the suffix back after the filter runs
and before Monarch is imported. Nothing else about the loop changes.

``client SPEC`` is the driver side. It reads each loop's id with the real
listener-generation probe (run locally with each loop's own HOME), dials
through ``mesh_attach.attach_cluster`` and ``mesh_runtime.attach_once``, and
prints one JSON line per step. Between steps it waits for a line on stdin, so
the test can restart a loop under it.

Mode ``render`` (issue #47) builds the mesh through the production
``mesh.get_mesh`` and its fault hook, with a small actor named as the fleet's
in place of GPUWorker, and dispatches the render's cross-rank price
(``capacity_agreement.agree_or_refuse``). After the test restarts every loop,
it passes the cached handle through ``mesh.ensure_live``, as a render does at
its start, and prices again: the price wait ends on the delivery failure that
names this mesh, and ``restart_reattach.heal`` retires the stale mesh and
attaches by the new ids, the call ``nodes/render_submit.submit_render`` makes,
before the price runs again on the new mesh. It never calls ``submit_render``
itself. Three seams stand in for what a CPU box cannot run: the policy
admission, the client lease thread, and the liveness scan, which counts every
actor process this user runs on the machine, real Worker services included.
The stand-in answers ``gone`` at once because the test killed the old loops'
process groups, and with them every process they spawned; the real scan's cost
is not measured here.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time


def host(address: str, suffix: str) -> None:
    from dgx_monarch.cli import worker_loop, worker_process_env

    real = worker_process_env.prepare_worker_process_environment

    def prepare(*args, **kwargs):
        result = real(*args, **kwargs)
        os.environ["MONARCH_LOG_SUFFIX"] = suffix
        return result

    worker_process_env.prepare_worker_process_environment = prepare
    sys.argv = ["worker_loop", "--address", address]
    worker_loop.main()


class _Log:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def _record(self, template, *args):
        self.lines.append(template % args if args else template)

    info = warning = error = debug = _record


def _say(**fields) -> None:
    print(json.dumps(fields), flush=True)


def _runtime(log: _Log) -> dict:
    from dgx_monarch import mesh_attach, mesh_factory, mesh_runtime

    def raise_attach(addresses, exc, **detail):
        mesh_attach.raise_attach(RuntimeError, addresses, exc, **detail)

    def raise_released(addresses, exc, *, auto_heal, restarted=None, **detail):
        mesh_attach.raise_released_attach(
            RuntimeError, addresses, exc, auto_heal=auto_heal, restarted=restarted,
            **detail)

    return {
        "mesh_factory": mesh_factory,
        "MeshAttachError": RuntimeError,
        "log": log,
        "_TRANSPORT_ENABLE_LOCK": threading.Lock(),
        "_TRANSPORT_POISON": None,
        "_TRANSPORT_BIND": None,
        "_attach_once": mesh_runtime.attach_once,
        "_heal_dead_loops": lambda *_a, **_k: None,
        "_raise_attach": raise_attach,
        "_raise_released_attach": raise_released,
    }


def _attach(runtime, config, log, step: str, *, released: bool = False):
    from dgx_monarch import attach_trace, mesh_attach, mesh_teardown

    if released:
        # The replacement path a driver takes after it proved a dead fleet gone.
        mesh_teardown.note_liveness_release(addresses=[h.address for h in config.hosts])
    started, first_line = time.monotonic(), len(log.lines)
    try:
        hosts = mesh_attach.attach_cluster(runtime, config, None)
    except BaseException as exc:
        _say(step=step, ok=False, wall_s=time.monotonic() - started,
             error=f"{type(exc).__name__}: {exc}"[:600])
        return None
    wall = time.monotonic() - started
    traced = [attach_trace.parse(line) or {} for line in log.lines[first_line:]]
    readings = [f for f in traced if f.get("event") == "loop_generation"]
    primed = [f["addresses"] for f in traced if f.get("event") == "proc_id_prime"]
    attempts = sum(1 for f in traced if f.get("event") == "phase" and f.get("phase") == "attach")
    procs = hosts.spawn_procs(per_host={"procs": 1}, name=f"p_{step}")
    procs.initialized.get(timeout=60)
    _say(step=step, ok=True, wall_s=wall, released=released, primed=primed,
         attempts=attempts,
         proc_ids={f["address"]: f.get("proc_id") for f in readings[-len(config.hosts):]},
         spawned_s=time.monotonic() - started)
    return hosts, procs


def _render_client(config, log: _Log) -> None:
    from monarch.actor import Actor, endpoint

    from dgx_monarch import (
        attach_trace,
        capacity_agreement,
        delivery_faults,
        mesh_helpers,
        mesh_liveness,
        monarch_launch,
        restart_reattach,
    )
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch.actor.capacity_quote import RankPrice

    row = RankPrice(verdict="fits", rung="stock_fits", reason="integration").to_row()

    class QuoteWorker(Actor):
        """The two endpoints a render's price and the fleet's teardown call."""

        def __init__(self, quote_row):
            self.row = quote_row

        @endpoint
        def capacity_quote(self, _request):
            return {"host": "loop", "quotes": [self.row]}

        @endpoint
        def teardown_group(self, release_models=False):
            return release_models

    def spawn(config, _use_cluster, gpus, attach, _bootstrap, handoff):
        hosts = attach(config, handoff)
        procs = hosts.spawn_procs(per_host={"gpus": gpus})
        procs.initialized.get(timeout=60)
        return hosts, procs, procs.spawn(delivery_faults.WORKER_MESH, QuoteWorker, row), True

    monarch_launch.prepare_driver = lambda _config: None
    mesh_helpers.start_client_lease = lambda: True
    mesh_mod.spawn_worker_fleet = spawn
    mesh_mod.find_config_path = lambda path: path or None
    mesh_mod.load_cluster_config = lambda _path: config
    mesh_mod._detect_comfy_dir = lambda _value: "/nonexistent/comfy"
    mesh_liveness.probe_fleet = lambda handle, **_kwargs: mesh_liveness.LivenessEvidence(
        mesh_liveness.GONE, tuple((host.address, mesh_liveness.GONE) for host in config.hosts),
        time.monotonic())
    mesh_mod.log = restart_reattach.log = log
    request = {"specs": [{"unet_name": "m.safetensors", "slot": "cond"}]}

    def price(handle) -> None:
        handle.setup_key = ("integration",)  # what the render's setup leaves
        capacity_agreement.agree_or_refuse(handle, request)

    handle = mesh_mod.get_mesh(mode="cluster", config_path=config.source)
    price(handle)
    mesh_uid = restart_reattach.watched_mesh(handle)
    _say(step="render_1", ok=True)
    sys.stdin.readline()  # the test restarts every loop here
    first_line = len(log.lines)
    started = time.monotonic()
    refused_after = None
    try:
        try:
            live = mesh_mod.ensure_live(handle)  # a render's first step
            if live is not handle:
                raise RuntimeError("ensure_live replaced the mesh before the price")
            price(live)
        except delivery_faults.WorkerDeliveryError as refused:
            # What nodes/render_submit.submit_render does with this refusal.
            refused_after = time.monotonic() - started
            healed = restart_reattach.heal(handle, refused)
            price(healed.handle)
        else:
            raise RuntimeError("the price answered on the old mesh")
    except BaseException as exc:
        _say(step="render_2", ok=False, wall_s=time.monotonic() - started,
             refused_after_s=refused_after, error=f"{type(exc).__name__}: {exc}"[:800])
        return
    traced = [fields for fields in map(attach_trace.parse, log.lines[first_line:]) if fields]
    readings = [f for f in traced if f.get("event") == "loop_generation"]
    named = delivery_faults.failed_since(
        0.0, [host.address for host in config.hosts], mesh_uid)
    _say(step="render_2", ok=True, wall_s=time.monotonic() - started,
         mesh_uid=mesh_uid, failures_named=named, new_mesh_uid=restart_reattach.watched_mesh(
             healed.handle),
         refused_after_s=refused_after, reattach_s=healed.wall_s, teardown=healed.teardown,
         old_retired=bool(handle.teardown_complete), new_mesh=healed.handle is not handle,
         reattach=[f for f in traced if f.get("event") == "restart_reattach"],
         primed=[f["addresses"] for f in traced if f.get("event") == "proc_id_prime"],
         proc_ids={f["address"]: f.get("proc_id") for f in readings[-len(config.hosts):]})


def client(spec_json: str) -> None:
    spec = json.loads(spec_json)
    from dgx_monarch.cli import lifecycle
    from dgx_monarch.config import ClusterConfig, HostConfig

    homes = spec["homes"]

    def run_on_host(_config, host_config, script, timeout=60, **_kwargs):
        env = dict(os.environ, HOME=homes[host_config.name])
        return subprocess.run(["bash", "-s"], input=script, capture_output=True,
                              text=True, timeout=timeout, env=env)

    lifecycle.run_on_host = run_on_host
    import monarch.actor

    if spec["mode"] == "render":
        config = ClusterConfig(
            hosts=tuple(HostConfig(name=name, address=address)
                        for name, address in spec["hosts"]),
            client_bind="tcp://127.0.0.1:0", auto_heal=False,
            transport_security="trusted_fabric", python_bin=sys.executable,
            source="/integration/cluster.toml")
        _render_client(config, _Log())
        _say(step="done")
        os._exit(0)

    # The production driver routes Monarch faults through a hook too
    # (mesh_helpers.install_unhandled_fault_hook); without one, the faults the
    # restarted loop's dead procs raise interrupt this process.
    faults: list[str] = []
    monarch.actor.unhandled_fault_hook = lambda failure: faults.append(str(failure)[:200])
    config = ClusterConfig(
        hosts=tuple(HostConfig(name=name, address=address)
                    for name, address in spec["hosts"]),
        client_bind="tcp://127.0.0.1:0", auto_heal=False,
        transport_security="trusted_fabric", python_bin=sys.executable)
    log = _Log()
    runtime = _runtime(log)
    keep = []
    if spec["mode"] == "restart":
        keep.append(_attach(runtime, config, log, "attach_1"))
        sys.stdin.readline()  # the test restarts one loop here
        keep.append(_attach(runtime, config, log, "attach_2", released=True))
    elif spec["mode"] == "bare":
        from monarch.actor import enable_transport

        from dgx_monarch import mesh_runtime

        enable_transport("tcp://127.0.0.1:0")
        runtime["_TRANSPORT_BIND"] = config.client_bind
        started = time.monotonic()
        try:
            mesh_runtime.attach_once([host.address for host in config.hosts])
            _say(step="bare", ok=True, wall_s=time.monotonic() - started)
        except BaseException as exc:
            _say(step="bare", ok=False, wall_s=time.monotonic() - started,
                 error=type(exc).__name__)
        keep.append(_attach(runtime, config, log, "by_id"))
    _say(step="done")
    os._exit(0)


if __name__ == "__main__":
    if sys.argv[1] == "host":
        host(sys.argv[2], sys.argv[3])
    else:
        client(sys.argv[2])
