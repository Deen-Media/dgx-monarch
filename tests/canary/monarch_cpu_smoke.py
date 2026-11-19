#!/usr/bin/env python3
"""Explicit local-process Monarch semantics canary; never attaches to Workers."""
from __future__ import annotations

import asyncio
import importlib.metadata
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

import monarch.actor as monarch_actor
from monarch._src.actor.proc_mesh import get_active_proc_meshes
from monarch.actor import (
    Actor,
    Future,
    concurrent_endpoint,
    endpoint,
    returns_future,
    shutdown_context,
    this_host,
)

SETUP_ACTOR_ENV = "DGXM_CANARY_SETUP_ACTOR"
SETUP_ACTOR = "monarch._src.actor.proc_mesh.SetupActor"
LOG_LEVEL_ACTOR = "monarch._src.actor.logging._LogLevelSetter"
ABORTED = "actor explicitly aborted due to: "


def record_setup_actor() -> None:
    """A proc bootstrap, as production passes one: note the actor Monarch runs it in."""
    from monarch._src.actor.actor_mesh import context

    instance = context().actor_instance
    os.environ[SETUP_ACTOR_ENV] = f"{instance.class_name} {instance.actor_id}"


class CpuProbe(Actor):
    def __init__(self):
        self.busy = False

    @endpoint
    async def is_busy(self) -> bool:
        return self.busy

    @endpoint
    async def identity(self) -> str:
        return importlib.metadata.version("torchmonarch")

    @concurrent_endpoint
    async def delayed(self) -> str:
        self.busy = True
        try:
            await asyncio.sleep(2)
            return "completed"
        finally:
            self.busy = False


class SyncProbe(Actor):
    @endpoint
    def has_event_loop(self) -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        return True


class ProcIdentity(Actor):
    """What the Worker service scripts and the fault hook read off an actor process."""

    @endpoint
    async def report(self) -> dict:
        argv = Path("/proc/self/cmdline").read_bytes().split(b"\0")
        environ = Path("/proc/self/environ").read_bytes().split(b"\0")
        module = next((argv[n + 1].decode() for n in range(len(argv) - 1) if argv[n] == b"-m"), None)
        return {"module": module,
                "bootstrap_marker": any(v.startswith(b"HYPERACTOR_MESH_BOOTSTRAP_MODE=") for v in environ),
                "setup_actor": os.environ.get(SETUP_ACTOR_ENV, "")}

    @endpoint
    async def exit_now(self) -> None:
        os._exit(1)


class RefusingProbe(Actor):
    def __init__(self):
        raise RuntimeError("deliberate constructor refusal")

    @endpoint
    async def status(self) -> str:
        return "ok"


def spawn_contracts(observations: dict[str, object]) -> None:
    """Bring-up rows mesh_factory rests on: the per-rank command, the constructor failure."""
    caller, seen, refused = threading.get_ident(), [], ValueError("deliberate bootstrap refusal")

    def refusing_command(_point: object) -> object:
        seen.append(threading.get_ident())
        raise refused

    before = len(get_active_proc_meshes())
    try:
        this_host().spawn_procs(per_host={"procs": 1}, bootstrap_command=refusing_command)
    except BaseException as exc:
        assert exc is refused, repr(exc)
    else:
        raise AssertionError("a refusing per-rank bootstrap command spawned procs")
    assert seen == [caller] and len(get_active_proc_meshes()) == before, (seen, caller)
    observations["bootstrap_command_raises_on_caller_before_spawn"] = True
    procs = this_host().spawn_procs(name="dgxm_cpu_canary_refusal")
    procs.initialized.get(timeout=15)
    # Its fault also reaches the hook, whose default ends this process.
    original, monarch_actor.unhandled_fault_hook = monarch_actor.unhandled_fault_hook, lambda _f: None
    try:
        procs.spawn("refusing_probe", RefusingProbe).initialized.get(timeout=30)
    except Exception as exc:
        text = str(exc)
    else:
        raise AssertionError("a refusing constructor initialized")
    finally:
        monarch_actor.unhandled_fault_hook = original
    words = text.find("deliberate constructor refusal")
    assert 0 <= text.find(ABORTED) < text.find("Traceback") < words, text[:2000]
    observations["constructor_failure_fails_initialized"] = True
    procs.stop().get(timeout=15)


def bootstrap_meshes(observations: dict[str, object]) -> None:
    """A spawn with a bootstrap, and the faults a dead proc of it raises.

    Monarch runs the bootstrap in a SetupActor mesh beside its log_level mesh,
    in a bootstrap_main process carrying the bootstrap marker. The fault hook
    reads each fault's mesh_name and keeps the setup- and log_level- ones quiet
    by their prefix and actor class (retired_meshes).
    """
    failures: list = []
    original = monarch_actor.unhandled_fault_hook
    monarch_actor.unhandled_fault_hook = failures.append
    try:
        procs = this_host().spawn_procs(name="dgxm_cpu_canary_boot", bootstrap=record_setup_actor)
        procs.initialized.get(timeout=15)
        probe = procs.spawn("identity_probe", ProcIdentity)
        report = probe.report.call_one().get(timeout=15)
        assert report["module"] == "monarch._src.actor.bootstrap_main", report
        assert report["bootstrap_marker"] is True, report
        assert report["setup_actor"].startswith(f"{SETUP_ACTOR} setup"), report
        observations["bootstrap_actor_process"] = True
        name = probe._name.get(timeout=5)
        probe.exit_now.call_one()
        texts: dict[str, str] = {}
        started = time.monotonic()
        read, deadline = 0, started + 90
        while time.monotonic() < deadline:
            fresh, read = failures[read:], len(failures)
            for failure in fresh:
                text = str(failure)
                assert type(failure).__name__ == "MeshFailure", type(failure)
                assert text.startswith(f"MeshFailure(mesh_name={failure.mesh_name},"), text[:300]
                texts[failure.mesh_name] = text
            internal = {prefix for key, text in texts.items() for prefix, actor in (
                ("setup-", SETUP_ACTOR), ("log_level-", LOG_LEVEL_ACTOR))
                if key.startswith(prefix) and actor in text}
            if name in texts and internal == {"setup-", "log_level-"}:
                break
            time.sleep(0.5)
        else:
            raise AssertionError(f"faults after a proc died: {sorted(texts)}")
        observations["dead_proc_fault_mesh_names"] = True
        observations["dead_proc_faults_s"] = round(time.monotonic() - started, 1)
    finally:
        monarch_actor.unhandled_fault_hook = original


def run() -> dict[str, object]:
    observations: dict[str, object] = {}
    events: list[str] = []

    @returns_future
    async def lazy() -> None:
        events.append("started")
        try:
            await asyncio.sleep(1)
            events.append("completed")
        finally:
            events.append("cancelled")

    try:
        # The first client-creating call sets the root logger to INFO, and the
        # pack's first is this_host() or attach_to_workers(), which
        # mesh_runtime.kept_root_log_level wraps; so this runs before anything else.
        logging.getLogger().setLevel(logging.WARNING)
        host = this_host()
        observations["root_level_after_this_host"] = logging.getLevelName(logging.getLogger().level)
        assert observations["root_level_after_this_host"] == "INFO", observations
        try:
            Future()
        except TypeError:
            observations["direct_future_rejected"] = True
        else:
            raise AssertionError("Future construction unexpectedly succeeded")
        future = lazy()
        assert not events, "returns_future body started before observation"
        try:
            future.get(timeout=0.01)
        except TimeoutError:
            pass
        else:
            raise AssertionError("returns_future did not time out")
        assert events == ["started", "cancelled"], events
        try:
            future.get(timeout=1)
        except TimeoutError:
            observations["returns_future_timeout_terminal"] = True
        else:
            raise AssertionError("cancelled body resumed")
        procs = host.spawn_procs(name="dgxm_cpu_canary")
        procs.initialized.get(timeout=15)
        actor = procs.spawn("async_probe", CpuProbe)
        expected = importlib.metadata.version("torchmonarch")
        assert actor.identity.call_one().get(timeout=15) == expected
        observations["child_version"] = expected
        sync = procs.spawn("sync_probe", SyncProbe)
        assert sync.has_event_loop.call_one().get(timeout=15) is False
        observations["sync_actor_has_no_event_loop"] = True
        delayed = actor.delayed.call_one()
        try:
            delayed.get(timeout=0.01)
        except TimeoutError:
            pass
        else:
            raise AssertionError("endpoint observation did not time out")
        assert actor.is_busy.call_one().get(timeout=1) is True
        assert delayed.get(timeout=5) == "completed"
        observations["concurrent_endpoint_timeout_reobservable"] = True
        stop = procs.stop()
        assert actor.identity.call_one().get(timeout=5) == expected
        observations["proc_stop_lazy"] = True
        stop.get(timeout=15)
        observations["proc_stop_completed"] = True
        # A timed-out stop cancels its remaining steps. A zero deadline lands
        # before the native stop starts; any nonzero one races it.
        doomed = this_host().spawn_procs(name="dgxm_cpu_canary_stop")
        doomed.initialized.get(timeout=15)
        survivor = doomed.spawn("stop_probe", CpuProbe)
        assert survivor.identity.call_one().get(timeout=15) == expected
        cancelled = doomed.stop()
        try:
            cancelled.get(timeout=0)
        except TimeoutError:
            pass
        else:
            raise AssertionError("ProcMesh stop finished inside a zero deadline")
        try:
            cancelled.get(timeout=1)
        except TimeoutError:
            observations["proc_stop_timeout_cancelled"] = True
        else:
            raise AssertionError("a timed-out ProcMesh stop resumed")
        time.sleep(1)
        assert survivor.identity.call_one().get(timeout=5) == expected
        observations["proc_stop_timeout_procs_survived"] = True
        doomed.stop().get(timeout=15)
        try:
            survivor.identity.call_one().get(timeout=5)
        except Exception as exc:
            observations["proc_stop_after_timeout_completed"] = type(exc).__name__
        else:
            raise AssertionError("procs answered after a completed stop")
        spawn_contracts(observations)
        bootstrap_meshes(observations)
    finally:
        shutdown_context(host_timeout=10).get(timeout=15)
    return observations


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(json.dumps(run(), indent=2) + "\n")
