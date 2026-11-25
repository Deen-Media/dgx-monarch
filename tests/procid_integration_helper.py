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


def _spawn_named(hosts, gpus: int, **kwargs):
    """What mesh_factory.spawn_worker_fleet does: a ProcMesh under a fresh name the pack records.

    Monarch names the ProcMesh's controller after it, so a controller report
    can be told to belong to this mesh's stop (proc_stops).
    """
    from dgx_monarch import proc_stops

    name = proc_stops.fresh_name()
    procs = hosts.spawn_procs(per_host={"gpus": gpus}, name=name, **kwargs)
    proc_stops.remember(procs, name)
    return procs


class _Records:
    """The driver's mesh log (the fault hook's lines among them), with when each came."""

    def __init__(self):
        import logging

        from dgx_monarch.log import get_logger

        self.lines: list[tuple[float, int, str]] = []
        handler = logging.Handler()
        handler.emit = lambda record: self.lines.append(  # type: ignore[method-assign]
            (time.monotonic(), record.levelno, record.getMessage()[:240]))
        get_logger("dgx_monarch.mesh").addHandler(handler)

    def errors(self, since: float = 0.0) -> list[str]:
        return [line for at, level, line in self.lines if at >= since and level >= 40]

    def naming(self, text: str) -> list[tuple[float, int, str]]:
        return [entry for entry in self.lines if text in entry[2]]


def _noop_bootstrap() -> None:
    """A proc bootstrap with nothing to do; a stand-in for mesh._bootstrap_proc."""
    return None


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


def _render_client(config, log: _Log, recycle: bool = False) -> None:
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

        @endpoint
        def pid(self) -> int:
            return os.getpid()

    def spawn(config, _use_cluster, gpus, attach, _bootstrap, handoff):
        hosts = attach(config, handoff)
        procs = _spawn_named(hosts, gpus)
        procs.initialized.get(timeout=60)
        return hosts, procs, procs.spawn(delivery_faults.WORKER_MESH, QuoteWorker, row), True

    records = _Records()

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
    if recycle:
        _recycle_beside_a_late_report(handle, healed.handle, records)


def _recycle_beside_a_late_report(old, healed, records: _Records) -> None:
    """Recycle the healed mesh while the heal's own stop of the old mesh can still report.

    The heal stopped the old mesh with a 2 s bound and gave up; its controller
    reports about 30 s later, the old loops being gone. One new proc is frozen
    so the Recycle's proc stop runs about 20 s (Monarch's host agent then kills
    it), which holds that stop open across the report. The Recycle must still
    return recycled, and the report must be logged as the heal's stop's own.
    """
    import signal

    from dgx_monarch import proc_stops

    healed_at = time.monotonic()
    old_label = proc_stops.label_prefix(proc_stops.name_of(old.procs) or "unnamed")
    pids = [pid for _point, pid in healed.workers.pid.call().get(timeout=30).items()]
    time.sleep(8.0)
    os.kill(pids[0], signal.SIGSTOP)
    healed.setup_key = None  # no group teardown: the frozen proc would hold it its whole 60 s
    started = time.monotonic()
    outcome = healed.recycle_detailed()
    ended = time.monotonic()
    deadline = healed_at + 70
    while time.monotonic() < deadline and not records.naming(old_label):
        time.sleep(0.5)
    reports = records.naming(old_label)
    report_at = reports[0][0] if reports else None
    _say(step="recycle", status=str(outcome.status), recycle_s=round(ended - started, 1),
         teardown_complete=bool(healed.teardown_complete),
         replacement_blocked=healed.replacement_blocked,
         report_after_heal_s=None if report_at is None else round(report_at - healed_at, 1),
         overlapped=report_at is not None and started <= report_at <= ended,
         report_levels=[level for _at, level, _line in reports], label=old_label,
         stop_errors=[line for line in records.errors() if "error stopping proc mesh" in line])


def _lifecycle_client(config, log: _Log, mode: str) -> None:
    """Modes ``lost_host``, ``stop_recycle`` and ``stop_detach``: production teardown on 0.8.0.

    The mesh comes from ``mesh.get_mesh`` with the same stand-ins as ``render``,
    and its fault hook, whose mesh log this captures. ``lost_host`` waits while
    the test kills every loop's process group, then fails a call, hands the
    error to ``mesh_helpers.mark_defunct_on_supervision_failure`` as a render's
    call path does, and waits for the supervision reconcile's stop. The two
    stop modes leave a call in flight, run Recycle or the detach shutdown, send
    one more call after the stop, and report what each call read and whether
    the hook raised an ERROR.
    """
    import logging

    from monarch.actor import Actor, concurrent_endpoint, endpoint

    from dgx_monarch import (
        delivery_faults,
        mesh_factory,
        mesh_helpers,
        mesh_liveness,
        mesh_stop,
        monarch_launch,
        retired_meshes,
    )
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch.log import get_logger

    class HoldWorker(Actor):
        """The endpoints the fleet's teardown calls, and one that holds a call in flight."""

        @endpoint
        async def capacity_quote(self, _request):
            return {"host": "loop", "quotes": []}

        @endpoint
        async def teardown_group(self, release_models=False):
            return release_models

        @concurrent_endpoint
        async def hold(self, seconds: float) -> str:
            import asyncio

            await asyncio.sleep(seconds)
            return "finished"

    def spawn(config, _use_cluster, gpus, attach, _bootstrap, handoff):
        hosts = attach(config, handoff)
        # Production always passes a bootstrap, which makes Monarch spawn its
        # setup mesh on every proc beside the log_level mesh.
        procs = _spawn_named(hosts, gpus, bootstrap=_noop_bootstrap)
        procs.initialized.get(timeout=60)
        workers = procs.spawn(delivery_faults.WORKER_MESH, HoldWorker)
        workers.initialized.get(timeout=60)
        # What mesh_factory.spawn_worker_fleet records after readiness: the mesh's
        # name for the retired-mesh check, its boxes for the lost-Worker words, and
        # the exit flag's hook.
        mesh_factory._record_mesh(workers, config, True, gpus)
        return hosts, procs, workers, True

    class _Records(logging.Handler):
        def __init__(self):
            super().__init__()
            self.lines: list[tuple[int, str]] = []

        def emit(self, record):
            self.lines.append((record.levelno, record.getMessage()[:240]))

    records = _Records()
    get_logger("dgx_monarch.mesh").addHandler(records)
    monarch_launch.prepare_driver = lambda _config: None
    mesh_helpers.start_client_lease = lambda: True
    mesh_mod.spawn_worker_fleet = spawn
    mesh_mod.find_config_path = lambda path: path or None
    mesh_mod.load_cluster_config = lambda _path: config
    mesh_mod._detect_comfy_dir = lambda _value: "/nonexistent/comfy"
    mesh_liveness.probe_fleet = lambda handle, **_kwargs: mesh_liveness.LivenessEvidence(
        mesh_liveness.UNKNOWN, tuple((host.address, mesh_liveness.UNKNOWN) for host in config.hosts),
        time.monotonic())
    mesh_mod.log = log
    # attach_to_workers() creates this process's client, which on 0.8.0 sets the
    # root logger to INFO unless the pack puts it back (mesh_runtime.kept_root_log_level).
    logging.getLogger().setLevel(logging.WARNING)
    handle = mesh_mod.get_mesh(mode="cluster", config_path=config.source)
    mesh_mod.install_fault_hook()
    _say(step="built", ok=True, bound_s=mesh_stop.proc_stop_bound_s(),
         root_level=logging.getLevelName(logging.getLogger().level))

    def text_of(future) -> str:
        try:
            future.get(timeout=120)
        except BaseException as exc:
            return f"{type(exc).__name__}: {exc}"[:600]
        return "returned"

    if mode == "lost_host":
        mesh_name = handle.workers._name.get(timeout=10)
        sys.stdin.readline()  # the test kills every loop's process group here
        started = time.monotonic()
        try:
            handle.workers.capacity_quote.call({}).get(timeout=120)
            _say(step="lost_host", ok=False, error="the call answered after the kill")
            return
        except BaseException as exc:
            failure = exc
        detected = time.monotonic() - started
        mesh_helpers.mark_defunct_on_supervision_failure(handle, failure)
        deadline = time.monotonic() + 2 * mesh_stop.proc_stop_bound_s() + 45
        while time.monotonic() < deadline:
            thread = getattr(handle, "_supervision_reconcile_thread", None)
            if handle.teardown_complete or (thread is not None and not thread.is_alive()):
                break
            time.sleep(0.5)
        time.sleep(35.0)  # the dead mesh's liveness faults arrive about 30 s after the death
        _say(step="lost_host", ok=True, detected_s=round(detected, 1),
             reconciled_s=round(time.monotonic() - started - 35.0, 1),
             failure=type(failure).__name__, teardown_complete=bool(handle.teardown_complete),
             replacement_blocked=handle.replacement_blocked, mesh=mesh_name,
             retired=retired_meshes.quiet(f"MeshFailure(mesh_name={mesh_name}, "),
             errors=[line for level, line in records.lines if level >= logging.ERROR],
             notices=[line for level, line in records.lines
                      if level == logging.INFO and "notice" in line])
        return
    if mode == "lost_beside_live":
        _lost_beside_live(handle, mesh_helpers, mesh_stop)
        return
    if mode == "idle_loss":
        sys.stdin.readline()  # the test kills one loop's process group here; nothing is in flight
        deadline = time.monotonic() + 75.0  # an idle Worker's fault reaches the hook about 30 s in
        while time.monotonic() < deadline and not any(
                level >= logging.ERROR and line.startswith("cluster fault:") for level, line in records.lines):
            time.sleep(0.5)
        _say(step="idle_loss", ok=True, errors=[line for level, line in records.lines if level >= logging.ERROR])
        return
    in_flight = handle.workers.hold.call(60.0)
    time.sleep(1.0)
    first = len(records.lines)
    if mode == "stop_recycle":
        outcome = handle.recycle_detailed()
        result = {"outcome": str(outcome.status), "ok": bool(outcome.ok)}
    else:
        handle.shutdown(timeout_s=mesh_stop.proc_stop_bound_s())
        result = {"outcome": "shutdown", "ok": bool(handle.teardown_complete)}
    straggler = text_of(handle.workers.hold.call(0.1))
    time.sleep(3.0)
    _say(step=mode, in_flight=text_of(in_flight), straggler=straggler, **result,
         errors=[line for level, line in records.lines[first:] if level >= logging.ERROR],
         notices=sum("teardown notice" in line for _level, line in records.lines[first:]))


def _lost_beside_live(handle, mesh_helpers, mesh_stop) -> None:
    """One of two loops is lost: the real reconcile's proc stop ends on Monarch's report.

    The test kills loop a's process group and leaves b. Monarch's ProcMesh
    controller then cannot confirm a's procs, ends the stop with "error
    stopping proc mesh" after proc_stop_max_idle, and fails; the stop's own
    Future never completes. The report names the controller after the name
    the mesh was spawned under, which ties it to this stop (proc_stops).
    """
    from dgx_monarch import mesh_teardown, proc_stops

    records = _Records()
    name = proc_stops.name_of(handle.procs)
    real, timed = mesh_stop.stop_procs, {}

    def stop_procs(procs, reason, timeout_s, *args, **kwargs):
        timed["bound_s"], started = timeout_s, time.monotonic()
        try:
            return real(procs, reason, timeout_s, *args, **kwargs)
        except BaseException as exc:
            timed["error"] = f"{type(exc).__name__}: {exc}"[:600]
            raise
        finally:
            timed["stop_s"] = round(time.monotonic() - started, 1)

    mesh_stop.stop_procs = stop_procs
    sys.stdin.readline()  # the test kills loop a's process group here
    started = time.monotonic()
    try:
        handle.workers.capacity_quote.call({}).get(timeout=120)
        _say(step="lost_beside_live", ok=False, error="the call answered after the kill")
        return
    except BaseException as exc:
        failure = exc
    detected = time.monotonic() - started
    mesh_helpers.mark_defunct_on_supervision_failure(handle, failure)
    deadline = time.monotonic() + 2 * mesh_stop.proc_stop_bound_s() + 45
    while time.monotonic() < deadline:
        thread = getattr(handle, "_supervision_reconcile_thread", None)
        if handle.teardown_complete or (thread is not None and not thread.is_alive()):
            break
        time.sleep(0.5)
    _say(step="lost_beside_live", ok=True, detected_s=round(detected, 1), name=name,
         label=proc_stops.label_prefix(name or "unnamed"), **timed,
         teardown_complete=bool(handle.teardown_complete),
         replacement_blocked=handle.replacement_blocked,
         block_cause=mesh_teardown.block_cause(handle),
         reports=[line for _at, level, line in records.naming("error stopping proc mesh")]
         + [f"{level} {line}" for _at, level, line in records.naming("proc_mesh_controller_")],
         stop_errors=[line for line in records.errors() if "proc_mesh_controller_" in line])


def _exit_attach_client(config, delay_s: float) -> None:
    """Attach on a daemon thread, as ComfyUI's prompt worker would, while the
    test holds one loop frozen so attempt 1 waits; return from main ``delay_s``
    into that attempt, so the interpreter exits with the attach in flight.

    ``auto_heal`` is on, with its two Worker restarts stood in and counted. An
    exit hook registered before Monarch's own (so it runs after it) waits for
    the attach's thread to report, so a line it logs after the exit's notice is
    never lost to the race with the interpreter's end.
    """
    import atexit

    from dgx_monarch import driver_exit, mesh_attach
    from dgx_monarch.log import get_logger

    reported = threading.Event()
    atexit.register(reported.wait, 30)
    counts = {"attempts": 0, "pre_attach_heals": 0, "post_fail_restarts": 0}
    runtime = _runtime(_Log())
    runtime["log"] = get_logger("dgx_monarch.mesh")  # the production logger, to stderr
    runtime["_heal_dead_loops"] = lambda *_a: counts.__setitem__(
        "pre_attach_heals", counts["pre_attach_heals"] + 1)
    mesh_attach.restart_after_failed_attach = lambda *_a: counts.__setitem__(
        "post_fail_restarts", counts["post_fail_restarts"] + 1)
    real_once, entered = runtime["_attach_once"], threading.Event()

    def attach_once(dial):
        counts["attempts"] += 1
        entered.set()
        return real_once(dial)

    runtime["_attach_once"] = attach_once

    def attach() -> None:
        fields: dict = {"error": None}
        try:
            mesh_attach.attach_cluster(runtime, config, None)
        except BaseException as exc:
            fields["error"] = f"{type(exc).__name__}: {exc}"[:300]
        _say(step="exit_attach", exiting=driver_exit.exiting(),
             poisoned=runtime["_TRANSPORT_POISON"] is not None, **counts, **fields)
        reported.set()

    threading.Thread(target=attach, name="render", daemon=True).start()
    entered.wait(60)
    time.sleep(delay_s)
    _say(step="main_returns")


def client(spec_json: str) -> None:
    spec = json.loads(spec_json)
    from dgx_monarch import mesh_liveness
    from dgx_monarch.cli import lifecycle
    from dgx_monarch.config import ClusterConfig, HostConfig

    # These modes pin Monarch's own stop outcomes. A stop after a death would
    # otherwise also ask the hosts, and this box's other actors answer for them.
    mesh_liveness.fleet_gone = lambda *_args, **_kwargs: False

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
        _render_client(config, _Log(), recycle=bool(spec.get("recycle")))
        _say(step="done")
        os._exit(0)
    if spec["mode"] == "exit_attach":
        config = ClusterConfig(
            hosts=tuple(HostConfig(name=name, address=address)
                        for name, address in spec["hosts"]),
            client_bind="tcp://127.0.0.1:0", auto_heal=True,
            transport_security="trusted_fabric", python_bin=sys.executable,
            source="/integration/cluster.toml")
        _exit_attach_client(config, float(spec.get("delay_s", 0.5)))
        return  # a normal interpreter exit, with its exit hooks, while the attach waits
    if spec["mode"] in ("lost_host", "lost_beside_live", "idle_loss", "stop_recycle", "stop_detach"):
        config = ClusterConfig(
            hosts=tuple(HostConfig(name=name, address=address)
                        for name, address in spec["hosts"]),
            client_bind="tcp://127.0.0.1:0", auto_heal=False,
            transport_security="trusted_fabric", python_bin=sys.executable,
            source="/integration/cluster.toml")
        _lifecycle_client(config, _Log(), spec["mode"])
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
