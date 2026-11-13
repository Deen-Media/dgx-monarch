"""The pair's agreement between two real processes: every rank makes the same collectives, and none waits.

``tests/test_backup_agreement.py`` answers the exchange for the peer. Here two
processes join one ``torch.distributed`` group on the CPU (gloo) and run the
guard's real wrapper against each other, with ``all_reduce`` counted in each.
What a rank reads of its own box is set by the scenario; the exchange, the
decision and the second exchange are the real ones. A rank that waited for a
collective its peer never made would hang the pair, so each scenario also has
a time limit.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch.distributed  # noqa: F401  (must be there: a skip would hide the exchange in CI)

SRC = Path(__file__).resolve().parents[1] / "src"

_RANK = r"""
import json, os, sys, weakref
from types import SimpleNamespace

rank, port, scenario = int(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3])
mine = scenario[str(rank)]
os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", port
if mine.get("switch"):
    os.environ["DGXM_DISABLE_BACKUP_FREE"] = "1"

import torch
import torch.distributed as dist

if mine.get("own_memory"):
    # A discrete card on any box: torch would answer for the box's own GPU, integrated on a Spark.
    torch.cuda.get_device_properties = lambda device=None: SimpleNamespace(is_integrated=0)

dist.init_process_group("gloo", rank=rank, world_size=2)

from dgx_monarch.actor import backup_agreement, lora_partial_load, partial_load_guard

GIB = 2 ** 30
made = {"collectives": 0, "freed": 0, "gathers": 0}
real_all_reduce, real_all_gather = dist.all_reduce, dist.all_gather


def counted(*args, **kwargs):
    made["collectives"] += 1
    return real_all_reduce(*args, **kwargs)


def gathered(*args, **kwargs):
    made["gathers"] += 1
    return real_all_gather(*args, **kwargs)


dist.all_reduce, dist.all_gather = counted, gathered
room = {"now": mine["room"]}


class Slot:
    base_baked, unbake = False, None


slot = Slot()
model = SimpleNamespace(current_weight_patches_uuid="stack", model_lowvram=False)
setattr(model, lora_partial_load.STORED, weakref.ref(slot))
patcher = SimpleNamespace(model=model, backup={"w": object()} if mine.get("stack", True) else {},
                          patches={"w": [1]}, is_dynamic=lambda: False,
                          load_device=SimpleNamespace(type="cuda" if mine.get("own_memory") else "cpu"))


def host_room():
    if mine.get("weighing_raises"):
        raise OSError("this rank cannot read its memory")
    return int(room["now"] * GIB)


def free(freed_patcher, stored):
    made["freed"] += 1
    freed_patcher.backup.clear()
    stored.base_baked = True
    room["now"] = mine.get("room_after", room["now"])
    return "freed in the test"


lora_partial_load._host_room = host_room
lora_partial_load._sampling_ask = lambda patcher, shape, conds: int(18.6 * GIB)
backup_agreement._single_estimate = lambda patcher, shape, conds: int(6.2 * GIB)
lora_partial_load._free_backup = free
lora_partial_load._what_is_wrong_after_the_route = lambda patcher, was: mine.get("wrong")

outcome, text = "ran", ""
try:
    wrapper = partial_load_guard.make_prepare_sampling_guard(whole_model=True)
    wrapper(lambda *args, **kwargs: ("real model", "conds", []), patcher, (1, 16, 22, 104, 60), {"positive": []})
except Exception as exc:
    outcome, text = type(exc).__name__, str(exc)
print(json.dumps({"rank": rank, "outcome": outcome, "text": text, **made}))
dist.destroy_process_group()
"""

# Each scenario: what rank 0 and rank 1 read of their own box, then what BOTH ranks must report.
SCENARIOS = {
    "nobody short": ({"0": {"room": 40.0}, "1": {"room": 40.0}}, 1, 0, "ran"),
    "the switch on one rank": ({"0": {"room": 9.0}, "1": {"room": 40.0, "switch": True}}, 1, 0, "ran"),
    "a weighing that raises on one rank": (
        {"0": {"room": 10.0}, "1": {"room": 40.0, "weighing_raises": True}}, 1, 0, "ran"),
    "one rank short, both free": (
        {"0": {"room": 9.0, "room_after": 24.0}, "1": {"room": 40.0}}, 2, 1, "ran"),
    "a rank with no stack": (
        {"0": {"room": 9.0, "room_after": 24.0}, "1": {"room": 40.0, "stack": False}}, 2, None, "ran"),
    # A device with memory of its own (a stand-in CUDA device that is not integrated): the rank takes
    # no part, whatever its host reads, and makes the same exchanges as its peer.
    "a pair of dedicated GPUs low on host memory": (
        {"0": {"room": 1.0, "own_memory": True}, "1": {"room": 1.0, "own_memory": True}}, 1, 0, "ran"),
    "a mixed fleet goes by its shared rank": (
        {"0": {"room": 9.0, "room_after": 24.0}, "1": {"room": 1.0, "own_memory": True}}, 2, None, "ran"),
    "a mixed fleet refuses on its shared rank": (
        {"0": {"room": 2.0, "room_after": 6.0}, "1": {"room": 1.0, "own_memory": True}},
        2, None, "PairBackupRoomError"),
    "a free and a refusal": (
        {"0": {"room": 2.0, "room_after": 6.0}, "1": {"room": 40.0}}, 2, 1, "PairBackupRoomError"),
}


def _free_port() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return str(probe.getsockname()[1])


def _run(scenario: dict) -> list[dict]:
    """Both ranks of ``scenario``, each in its own process; what each reported, rank 0 first."""
    port = _free_port()
    env = {**os.environ, "PYTHONPATH": str(SRC), "OMP_NUM_THREADS": "1"}
    env.pop("DGXM_DISABLE_BACKUP_FREE", None)
    ranks = [subprocess.Popen([sys.executable, "-c", _RANK, str(rank), port, json.dumps(scenario)],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for rank in (0, 1)]
    reports = []
    try:
        for process in ranks:
            out, err = process.communicate(timeout=120)  # a rank left waiting in a collective ends here
            assert process.returncode == 0, err[-2000:]
            reports.append(json.loads(out.strip().splitlines()[-1]))
    finally:
        for process in ranks:
            if process.poll() is None:
                process.kill()
    return reports


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_both_ranks_make_the_same_collectives_and_neither_waits(name):
    scenario, collectives, freed, outcome = SCENARIOS[name]
    reports = _run(scenario)
    assert [report["collectives"] for report in reports] == [collectives, collectives], reports
    # A render that samples crosses no words; a refusal crosses them once on each rank.
    gathers = 0 if outcome == "ran" else 1
    assert [report["gathers"] for report in reports] == [gathers, gathers], reports
    assert [report["outcome"] for report in reports] == [outcome, outcome], reports
    if freed is not None:
        assert [report["freed"] for report in reports] == [freed, freed], reports
    else:  # rank 1 (no stack, or a device with memory of its own) frees nothing and still makes the second exchange
        assert [report["freed"] for report in reports] == [1, 0], reports


def test_the_wrong_ranks_own_words_reach_the_error_on_both_ranks():
    """Rank 1 finds its model wrong after the free: both ranks refuse with rank 1's cause first.

    The driver receives one rank's error for the fleet, whichever arrives first, so each must carry it.
    """
    cause = "ComfyUI holds its weights as unpatched"
    reports = _run({"0": {"room": 40.0}, "1": {"room": 9.0, "room_after": 24.0, "wrong": cause}})
    assert [report["outcome"] for report in reports] == ["PeerLoadFailureError", "OffloadedInputQuantError"]
    assert [report["collectives"] for report in reports] == [2, 2]
    assert [report["gathers"] for report in reports] == [1, 1]
    healthy, wrong = reports[0]["text"], reports[1]["text"]
    assert cause in wrong and "after rank 1 freed its weight backup" in wrong
    assert healthy.index("rank 1 found its model is not as a full load leaves it") < healthy.index(cause)
    assert healthy.index(cause) < healthy.index("a peer rank could not load or verify")
    assert "The model on rank 0 was right" in healthy and healthy.rstrip().endswith("[dgxm:rank 1]")


def test_a_room_refusal_gives_the_short_ranks_figures_on_both_ranks():
    reports = _run({"0": {"room": 2.0, "room_after": 6.0}, "1": {"room": 40.0}})
    assert [report["outcome"] for report in reports] == ["PairBackupRoomError"] * 2
    for report in reports:
        text = report["text"]
        assert "rank 0 is" in text and "Rank 0 read 6.0 GiB of room" in text
        assert text.index("Rank 0 read") < text.index("rank 1 read 40.0 GiB of room")
        assert text.rstrip().endswith("[dgxm:rank 0]")
