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
