"""Pin what bring-up and teardown rest on under torchmonarch 0.8.0's direct spawn.

0.8.0 spawns actor meshes directly by default (meta-pytorch/monarch#5311,
#5336), and the pack keeps that default. Three things changed under it:

- A GPUWorker whose constructor raises now fails ``initialized`` with a
  RuntimeError that opens with Monarch's supervisor chain ("error spawning
  actor mesh: statuses: Failed(supervisor proxy ..."), where 0.7.0 failed the
  first call. The factory's error must still open with the Worker's own words
  (mesh_factory.worker_words).
- ``ProcMesh.stop(reason)`` no longer carries the reason to a call in flight,
  which reads "stopped: parent draining"; ``ActorMesh.stop(reason)`` does.
  Teardown stops the actors with the reason first
  (mesh_teardown.stop_with_reason), so the teardown markers in mesh_safety
  still name it.
- The per-rank bootstrap callable runs before anything spawns, so its
  refusal raises from ``spawn_procs`` itself and leaves no ProcMesh
  (tests/test_mesh_factory_bootstrap_refusal.py relies on it).

The probe runs the real driver activation and the real factory in a fresh
interpreter, because monarch allows one transport per process and the suite
must stay reusable after this test.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch import mesh_teardown
from dgx_monarch.mesh_factory import WorkerSpawnError, worker_words

PROBE = Path(__file__).with_name("spawn_contract_probe.py")
SRC = Path(__file__).resolve().parents[1] / "src"
WORDS = "RuntimeError: deliberate constructor refusal"

# The two shapes, captured on CPU with the source paths shortened.
TEXT_080 = """error spawning actor mesh: statuses: Failed(supervisor proxy: failed: Supervision event: \
actor worker<a>.anon-0<b>@anon-0<b>.ipc://@c failed because it did not handle a supervision event \
from its child. The child's event was:
  Supervision event: actor <root>.<pkg.RefusingWorker dgxm_worker{'gpus': 0/1}> failed:
    actor explicitly aborted due to: Traceback (most recent call last):
      File "monarch/_src/actor/actor_mesh.py", line 1529, in _dispatch_loop
        await _handle_queued_message(actor, msg)
      File "monarch/_src/actor/actor_mesh.py", line 1903, in _prepare
        self.instance = Class(*args, **kwargs)
                        ^^^^^^^^^^^^^^^^^^^^^^
      File "pkg.py", line 56, in __init__
        raise RuntimeError("deliberate constructor refusal")
    RuntimeError: deliberate constructor refusal
)=0..1"""
TEXT_070 = """Endpoint call dgxm_worker.status() failed, The actor <root>.<pkg.RefusingWorker \
dgxm_worker{'gpus': 1/2}> and all its descendants have failed:
  actor explicitly aborted due to: Traceback (most recent call last):
    File "monarch/_src/actor/actor_mesh.py", line 1497, in _dispatch_loop
      await _handle_queued_message(actor, msg)
    File "pkg.py", line 57, in __init__
      raise RuntimeError("deliberate constructor refusal")
  RuntimeError: deliberate constructor refusal
"""


@pytest.mark.parametrize("text", [TEXT_080, TEXT_070], ids=["0.8.0", "0.7.0"])
def test_the_factory_error_opens_with_the_workers_own_words(text):
    assert worker_words(text) == WORDS
    error = WorkerSpawnError(RuntimeError(text), None)
    assert error.original_text.startswith(WORDS)
    assert str(error).startswith(WORDS)
    assert text in error.original_text


def test_a_chained_traceback_yields_the_exception_that_ended_it():
    text = TEXT_070.replace(
        "  RuntimeError: deliberate constructor refusal\n",
        "  KeyError: 'slot'\n\n  During handling of the above exception, another exception occurred:\n\n"
        "  Traceback (most recent call last):\n    File \"pkg.py\", line 9, in __init__\n"
        f"  {WORDS}\n")
    assert worker_words(text) == WORDS


@pytest.mark.parametrize("text", ["worker spawn failed", "Supervision event: actor failed: no traceback"])
def test_a_text_without_a_worker_traceback_is_left_alone(text):
    assert worker_words(text) is None
    assert WorkerSpawnError(RuntimeError(text), None).original_text == text


def test_an_abort_with_a_bare_message_is_the_workers_words():
    text = "Supervision event: actor <root>.<pkg.W w> failed:\n  actor explicitly aborted due to: out of slots"
    assert worker_words(text) == "out of slots"


class _Future:
    def __init__(self, calls, name, failure=None):
        self.calls, self.name, self.failure = calls, name, failure

    def get(self, timeout=None):
        self.calls.append((self.name, timeout))
        if self.failure is not None:
            raise self.failure
        return None


def _handle(calls, *, actor_failure=None, proc_failure=None, workers=True):
    def stop_for(name, failure):
        def stop(reason):
            calls.append((f"{name}.stop", reason))
            return _Future(calls, f"{name}.get", failure)
        return stop

    handle = SimpleNamespace(procs=SimpleNamespace(stop=stop_for("procs", proc_failure)))
    if workers:
        handle.workers = SimpleNamespace(stop=stop_for("workers", actor_failure))
    return handle


def test_teardown_stops_the_actors_with_the_reason_before_the_procs():
    calls: list = []
    mesh_teardown.stop_with_reason(_handle(calls), "dgx-monarch recycle", 60)
    assert calls == [("workers.stop", "dgx-monarch recycle"),
                     ("workers.get", mesh_teardown.ACTOR_STOP_S),
                     ("procs.stop", "dgx-monarch recycle"), ("procs.get", 60)]


@pytest.mark.parametrize("failure", [TimeoutError("actor stop"), RuntimeError("actors already gone")])
def test_an_actor_stop_failure_never_reads_as_the_proc_stops_outcome(failure):
    calls: list = []
    mesh_teardown.stop_with_reason(_handle(calls, actor_failure=failure), "dgx-monarch client detach", 30)
    assert calls[-2:] == [("procs.stop", "dgx-monarch client detach"), ("procs.get", 30)]


def test_the_proc_stop_stays_the_authority():
    calls: list = []
    with pytest.raises(TimeoutError, match="proc stop"):
        mesh_teardown.stop_with_reason(
            _handle(calls, proc_failure=TimeoutError("proc stop")), "dgx-monarch recycle", 60)
    calls.clear()
    mesh_teardown.stop_with_reason(_handle(calls, workers=False), "dgx-monarch recycle", 60)
    assert calls == [("procs.stop", "dgx-monarch recycle"), ("procs.get", 60)]


def test_the_root_level_goes_back_only_from_info():
    """Monarch sets INFO when it creates the client; a level another thread chose stays."""
    import logging

    from dgx_monarch.mesh_runtime import kept_root_log_level

    root = logging.getLogger()
    saved = root.level
    try:
        root.setLevel(logging.WARNING)
        with kept_root_log_level():
            root.setLevel(logging.INFO)
        assert root.level == logging.WARNING
        with kept_root_log_level():
            root.setLevel(logging.DEBUG)
        assert root.level == logging.DEBUG
    finally:
        root.setLevel(saved)


def _run_probe(tmp_path: Path) -> dict:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o700)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("MONARCH_", "HYPERACTOR_", "DGXM_MONARCH_POLICY"))}
    environment.update(HOME=str(tmp_path), XDG_RUNTIME_DIR=str(runtime_dir), CUDA_VISIBLE_DEVICES="",
                       PYTHONPATH=os.pathsep.join(filter(None, (str(SRC), os.environ.get("PYTHONPATH")))))
    proc = subprocess.run([sys.executable, str(PROBE)], capture_output=True, text=True,
                          timeout=240, env=environment)
    assert proc.returncode == 0, (
        f"probe crashed rc={proc.returncode}\n"
        f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_bring_up_and_teardown_on_the_real_runtime(tmp_path):
    pytest.importorskip("monarch.actor")
    result = _run_probe(tmp_path)
    assert result["native_readback"] == "verified", result
    # The real factory: the refused constructor reaches the user in its own words.
    assert result["factory_text"].startswith(WORDS), result
    assert result["factory_rollback_confirmed"] is True, result
    # 0.8.0 sets the root logger to INFO when this_host() creates the client; the pack puts it back.
    assert result["root_level_after_factory"] == "WARNING", result
    # Teardown: the call in flight names the pack's stop reason.
    assert "stopped: dgx-monarch recycle" in result["in_flight_after_stop"], result
    assert "parent draining" not in result["in_flight_after_stop"], result
    # Only the refused mesh's own failure reached the hook.
    assert len(result["faults"]) == 1 and "dgxm_worker" in result["faults"][0], result
    # The bootstrap callable runs before anything spawns: its own exception, no ProcMesh.
    assert result["bootstrap_spawn"] == "raised", result
    assert result["bootstrap_error_is_callables"] is True, result
    assert result["bootstrap_proc_meshes_added"] == 0, result
