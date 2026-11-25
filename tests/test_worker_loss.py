"""A lost Worker's cause, in plain words, read from the texts Monarch really sends.

Every text below was captured on CPU against torchmonarch 0.7.0 and 0.8.0: a
Worker process killed with SIGKILL, one that exited with code 3, a box whose
Worker service was killed or frozen (no answer at all), and a Worker service
restarted under an attached mesh. The probe actor's class name is replaced by
the pack's own (``dgx_monarch.actor.worker.GPUWorker``); nothing else changed.
The negatives are texts the parser must leave alone: Monarch's system meshes
reporting the same death, an undeliverable message, a Worker's own exception
and the deliberate stops.
"""
from __future__ import annotations

import pytest

from dgx_monarch import box_names, worker_loss

W = "<root>.<dgx_monarch.actor.worker.GPUWorker dgxm_worker"
FAILED = "and all its descendants have failed:\n  "
PAIR = ["the head", "the sibling"]

KILL_FAULT_070 = (f"MeshFailure(mesh_name=dgxm_worker-HzZjQgcCSUv, crashed_ranks=[1], event=Supervision "
                  f"event: actor {W}{{'gpus': 1/2}}> failed:\n  the process this actor was running on "
                  "failed: Killed(sig=9))")
KILL_FAULT_080 = KILL_FAULT_070.replace("HzZjQgcCSUv", "3Y3EiGJ8Bmi")
KILL_ERROR = (f"Endpoint call dgxm_worker.status() failed, The actor {W}{{'gpus': 1/2}}> {FAILED}"
              "the process this actor was running on failed: Killed(sig=9)")
EXIT_FAULT_070 = (f"MeshFailure(mesh_name=dgxm_worker-7kBigQ6FapA, crashed_ranks=[1], event=Supervision "
                  f"event: actor {W}{{'hosts': 0/1, 'gpus': 1/2}}> failed:\n  the process this actor "
                  "was running on exited with non-zero code 3)")
EXIT_FAULT_080 = EXIT_FAULT_070.replace("7kBigQ6FapA", "xXfhSbWYyQT")
EXIT_ERROR = (f"Endpoint call dgxm_worker.die() failed, The actor {W}{{'hosts': 0/1, 'gpus': 1/2}}> "
              f"{FAILED}the process this actor was running on exited with non-zero code 3")
LIVENESS_FAULT_080 = (f"MeshFailure(mesh_name=dgxm_worker-2x9JkQH1rcB, crashed_ranks=[1], event=Supervision "
                      f"event: actor {W}{{'hosts': 0/1, 'gpus': 1/2}}> failed:\n  supervision liveness "
                      "failed)")
MONITOR_ERROR_080 = (f"Endpoint call dgxm_worker.status() failed, The actor {W}{{'hosts': 0/1, 'gpus': 1/2}}> "
                     f"{FAILED}status request to monitored actor dgxm_worker<2x9JkQH1rcB>.anon-1<yMAt25rohmU>"
                     "@anon-1<yMAt25rohmU>.tcp://127.0.0.1:48873 timed out after 10000ms")
UNREACHABLE_ERROR_080 = (f"Endpoint call dgxm_worker.probe_block() failed, The actor {W}{{'hosts': 0/1, "
                         f"'gpus': 0/2}}> {FAILED}host unreachable: no status received from process "
                         "within 60s")
AGENT_TIMEOUT_ERROR_070 = (
    "Endpoint call dgxm_worker.status() failed, The actor anon-1<e5JYHuLbSuA>@anon-1<e5JYHuLbSuA>.ipc://"
    f"@d8bnZR6bquUx94kNoCsLgjj6,proc_agent {FAILED}timeout waiting for message from proc mesh agent while "
    'querying for "dgxm_worker-6DXsAfzuye4". The process likely crashed')
NO_STATUS_ERROR_070 = (f"Endpoint call dgxm_worker.status() failed, Supervision event: actor "
                       f"{W}{{'hosts': 0/1, 'gpus': 0/2}}> has status:\n  stopped: no status received "
                       "from process")
RESTART_ERROR_080 = (f"Endpoint call dgxm_worker.probe_info() failed, The actor {W}{{'hosts': 0/1, "
                     f"'gpus': 0/2}}> {FAILED}monitor for actor dgxm_worker<7AKXbvZJYAW>.anon-0<zLBLikMgEVG>"
                     "@anon-0<zLBLikMgEVG>.tcp://127.0.0.1:47717 stopped before reporting a failure")

SYSTEM_FAULT_080 = ("MeshFailure(mesh_name=log_level-gnXeZg3rNkL, crashed_ranks=[1], event=Supervision "
                    "event: actor <root>.<monarch._src.actor.logging._LogLevelSetter log_level{'gpus': "
                    "1/2}> failed:\n  the process this actor was running on failed: Killed(sig=9))")
UNDELIVERABLE_FAULT_080 = (
    "MeshFailure(mesh_name=<none>, crashed_ranks=[], event=Supervision event: actor "
    "<monarch._src.actor.actor_mesh.RootClientActor client> failed:\n  undeliverable message for "
    "dgxm_worker.status() (call):\n  \tdescription: delivery of message from sender to dest failed\n"
    "  \terror: delivery failure: invalid reference dgxm_worker<oZfHxBTQyUP>.anon-1<fWYuujdCf3G>"
    "@anon-1<fWYuujdCf3G>.ipc://@zpxx7QQs9FPcR4zopwGlwX5a: actor failed\n)")
CTOR_FAULT_080 = (f"MeshFailure(mesh_name=dgxm_worker-oZfHxBTQyUP, crashed_ranks=[1], event=Supervision "
                  f"event: actor {W}{{'gpus': 1/2}}> failed:\n  actor explicitly aborted due to: "
                  'Traceback (most recent call last):\n    File "monarch/_src/actor/actor_mesh.py", '
                  "line 1903, in _prepare\n      self.instance = Class(*args, **kwargs)\n"
                  "  RuntimeError: deliberate constructor refusal\n)")
RECYCLE_ERROR_070 = (f"Endpoint call dgxm_worker.probe_block() failed, Supervision event: actor "
                     f"{W}{{'gpus': 0/2}}> has status:\n  stopped: dgx-monarch recycle")
DRAINING_ERROR_080 = RECYCLE_ERROR_070.replace("dgx-monarch recycle", "parent draining")


@pytest.mark.parametrize("text, kind, ranks, mesh, signal, code", [
    (KILL_FAULT_070, "signal", (1,), "HzZjQgcCSUv", 9, None),
    (KILL_FAULT_080, "signal", (1,), "3Y3EiGJ8Bmi", 9, None),
    (KILL_ERROR, "signal", (1,), None, 9, None),
    (EXIT_FAULT_070, "exit_code", (1,), "7kBigQ6FapA", None, 3),
    (EXIT_FAULT_080, "exit_code", (1,), "xXfhSbWYyQT", None, 3),
    (EXIT_ERROR, "exit_code", (1,), None, None, 3),
    (LIVENESS_FAULT_080, "lost_contact", (1,), "2x9JkQH1rcB", None, None),
    (MONITOR_ERROR_080, "lost_contact", (1,), None, None, None),
    (UNREACHABLE_ERROR_080, "lost_contact", (0,), None, None, None),
    (AGENT_TIMEOUT_ERROR_070, "lost_contact", (), "6DXsAfzuye4", None, None),
    (NO_STATUS_ERROR_070, "lost_contact", (0,), None, None, None),
    (RESTART_ERROR_080, "lost_contact", (0,), None, None, None),
])
def test_each_captured_text_reads_as_its_cause(text, kind, ranks, mesh, signal, code):
    assert worker_loss.read(text) == worker_loss.WorkerLoss(kind, ranks, mesh, signal, code)


@pytest.mark.parametrize("text", [SYSTEM_FAULT_080, UNDELIVERABLE_FAULT_080, CTOR_FAULT_080,
                                  RECYCLE_ERROR_070, DRAINING_ERROR_080, "", "dgxm_worker"])
def test_texts_that_are_not_a_lost_worker_are_left_alone(text):
    assert worker_loss.read(text) is None
    assert worker_loss.explain_fault(text, None, {}) is None
    assert worker_loss.explain_error(text, PAIR) is None


def test_the_faults_own_mesh_name_wins_over_the_text():
    assert worker_loss.read(KILL_FAULT_080, "log_level-gnXeZg3rNkL") is None
    assert worker_loss.read(KILL_FAULT_080, "dgxm_worker-other").mesh == "other"


def test_a_kill_names_the_box_and_the_signal_and_only_points_at_memory():
    from dgx_monarch import mesh_safety

    text = worker_loss.explain_fault(KILL_FAULT_080, "dgxm_worker-3Y3EiGJ8Bmi",
                                     {"3Y3EiGJ8Bmi": PAIR})
    assert text.startswith("The Worker on the sibling was stopped by signal 9 (SIGKILL); "), text
    assert "the kernel's out-of-memory killer sends signal 9, and so does `kill -9`" in text
    assert "look for an out-of-memory line in `dmesg` or `journalctl -k` on that box" in text
    # The hint never states the cause, and the pack's memory classifier never reads it as one.
    assert "out of memory" not in text.lower()
    assert not mesh_safety.is_memory_exhaustion(RuntimeError(text))


def test_an_exit_code_and_a_lost_box_read_plainly():
    assert worker_loss.explain_error(EXIT_ERROR, PAIR) == "The Worker on the sibling exited with code 3."
    assert (worker_loss.explain_error(MONITOR_ERROR_080, PAIR)
            == "The driver lost contact with the sibling.")
    assert (worker_loss.explain_error(UNREACHABLE_ERROR_080, PAIR)
            == "The driver lost contact with the head.")
    assert (worker_loss.explain_error(AGENT_TIMEOUT_ERROR_070, PAIR)
            == "The driver lost contact with a Worker.")


def test_a_fault_from_a_mesh_the_driver_no_longer_holds_never_takes_todays_boxes():
    # On 0.8.0 an idle Worker's fault arrives about 30 s after the death; by then
    # the fleet may be a new one, with other boxes at the same ranks.
    replaced = {"aNewMeshUid": ["the sibling", "the head"]}
    assert (worker_loss.explain_fault(LIVENESS_FAULT_080, "dgxm_worker-2x9JkQH1rcB", replaced)
            == "The driver lost contact with a Worker of an earlier fleet.")
    text = worker_loss.explain_fault(KILL_FAULT_080, "dgxm_worker-3Y3EiGJ8Bmi", replaced)
    assert text.startswith("A Worker of an earlier fleet was stopped by signal 9"), text


def test_several_ranks_and_gpu_level_boxes():
    both = KILL_FAULT_080.replace("crashed_ranks=[1]", "crashed_ranks=[0, 1]")
    text = worker_loss.explain_fault(both, None, {"3Y3EiGJ8Bmi": PAIR})
    assert text.startswith("The Workers on the head and on the sibling were stopped by signal 9"), text
    labels = [box_names.label(box, gpu) for box, gpu in
              box_names.rank_boxes(["box-a"], 2, 2, lambda name: True)]
    assert worker_loss.explain_error(KILL_ERROR, labels).startswith(
        "The Worker for GPU 1 on this box was stopped by signal 9")
    assert (worker_loss.explain_error(MONITOR_ERROR_080, labels)
            == "The driver lost contact with this box.")


def test_the_memory_hint_says_where_to_look_by_how_many_boxes_were_named():
    hint = "look for an out-of-memory line in `dmesg` or `journalctl -k` on "
    one = worker_loss.explain_fault(KILL_FAULT_080, None, {"3Y3EiGJ8Bmi": PAIR})
    assert hint + "that box before" in one
    both = worker_loss.explain_fault(KILL_FAULT_080.replace("crashed_ranks=[1]", "crashed_ranks=[0, 1]"),
                                     None, {"3Y3EiGJ8Bmi": PAIR})
    assert hint + "those boxes before" in both
    gpus = [box_names.label(box, gpu) for box, gpu in box_names.rank_boxes(["box-a"], 2, 2, lambda name: True)]
    two_gpus = worker_loss.words(worker_loss.WorkerLoss(worker_loss.SIGNAL, (0, 1), signal=9), gpus)
    assert two_gpus.startswith("The Workers for GPU 0 on this box and for GPU 1 on this box"), two_gpus
    assert hint + "that box before" in two_gpus  # two GPUs, one box
    for unnamed in (worker_loss.explain_fault(KILL_FAULT_080, None, {"3Y3EiGJ8Bmi": []}),
                    worker_loss.explain_fault(KILL_FAULT_080, None, {"aNewMeshUid": PAIR}),
                    worker_loss.explain_error(KILL_ERROR, None)):
        assert unnamed is not None and "that box" not in unnamed and "those boxes" not in unnamed, unnamed
        assert hint + "each box of the fleet before" in unnamed, unnamed


def test_other_signals_name_themselves_without_the_memory_hint():
    text = worker_loss.explain_error(KILL_ERROR.replace("sig=9", "sig=15"), PAIR)
    assert text == "The Worker on the sibling was stopped by signal 15 (SIGTERM)."
    core = worker_loss.explain_error(KILL_ERROR.replace("Killed(sig=9)", "Killed(sig=11, core)"), PAIR)
    assert core == "The Worker on the sibling was stopped by signal 11 (SIGSEGV)."


@pytest.mark.parametrize("text", [KILL_FAULT_080, EXIT_FAULT_080, LIVENESS_FAULT_080, MONITOR_ERROR_080,
                                  RESTART_ERROR_080, AGENT_TIMEOUT_ERROR_070])
def test_no_sentence_names_an_address_a_uid_or_a_rank_number(text):
    for sentence in (worker_loss.explain_fault(text, None, {"3Y3EiGJ8Bmi": PAIR, "xXfhSbWYyQT": PAIR}),
                     worker_loss.explain_error(text, PAIR)):
        assert sentence is not None
        assert "127.0.0.1" not in sentence and "dgxm_worker" not in sentence
        assert "rank" not in sentence and "<" not in sentence
