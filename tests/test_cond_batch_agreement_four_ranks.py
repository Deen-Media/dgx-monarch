"""Two data-parallel groups that shard their calls: each agrees within itself and never waits on the other.

Under a layout that both shards a model call and splits the batch (uly2+dp2 or
cfg2+dp2 at world 4, uly2+cfg2+dp2 at world 8), each data-parallel group
samples its own slice of the batch. An adaptive-step sampler (``dpm_adaptive``)
sizes its steps from the error of that slice, so the two groups can make
different numbers of model calls. The cond batch agreement once reduced over
the default group on every call, so the group with more calls waited in that
exchange, for ranks that had finished, until the group timeout.

One process per rank joins one gloo group on the CPU and builds xFuser's own
groups with the call the Worker makes (``actor/worker_env.py``). Each takes the
real ``install`` and its wrapper, then makes the model calls its group's
sampler would: two in one group, three in the other, each call running one
all-gather over each group the model call's own collectives use. Each rank
reads its own free memory, so the figure each call sees shows which ranks
took part in the exchange, and each rank records the group the exchange ran
over. ComfyUI's registration is a stand-in here; the two-rank test in
``tests/test_cond_batch_agreement.py`` runs ComfyUI's own batch rule.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import xfuser.core.distributed  # noqa: F401  (a required dependency: a skip would hide the exchange in CI)

SRC = Path(__file__).resolve().parents[1] / "src"
# Model calls each dp group's sampler makes, as two batch slices under
# dpm_adaptive can.
CALLS = [2, 3]
# The default group's timeout, which also bounds the rendezvous and every group
# xFuser makes, so a slow runner does not fail there.
INIT_S = 120
# How long a rank whose group has finished waits for the others before it
# exits. A rank still waiting in an exchange on it then fails on the closed
# peer, well before INIT_S.
WAIT_S = 8

_RANK = r"""
import datetime, json, os, sys, time
from types import ModuleType, SimpleNamespace

rank, port, scenario = int(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3])
world, topology = len(scenario["free_gib"]), scenario["topology"]
os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", port

import torch
import torch.distributed as dist

# xFuser binds a CUDA device, by a name it takes at import; this runs on the CPU.
torch.cuda.set_device = lambda *args, **kwargs: None
from xfuser.core.distributed import (
    get_cfg_group,
    get_data_parallel_rank,
    get_model_replica_group,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)

# Imports first: the timeout below also bounds the rendezvous, so the ranks
# reach it with their slow work done.
dist.init_process_group("gloo", rank=rank, world_size=world,
                        timeout=datetime.timedelta(seconds=scenario["init_s"]))
init_distributed_environment(rank=rank, world_size=world, local_rank=0, backend="gloo")
sp = int(topology.get("ulysses", 1)) * int(topology.get("ring", 1))
cfg = int(topology.get("cfg", 1))
initialize_model_parallel(
    data_parallel_degree=int(topology.get("dp", 1)),
    sequence_parallel_degree=sp,
    classifier_free_guidance_degree=cfg,
    ulysses_degree=int(topology.get("ulysses", 1)),
    ring_degree=int(topology.get("ring", 1)),
    backend="gloo",
)

# Stand-in for comfy.patcher_extension: the two calls install makes.
comfy = ModuleType("comfy")
pe = ModuleType("comfy.patcher_extension")
pe.WrappersMP = SimpleNamespace(CALC_COND_BATCH="calc_cond_batch")


def get_wrappers_with_key(kind, key, options, is_model_options=False):
    return options.get("wrappers", {}).get(kind, {}).get(key, [])


def add_wrapper_with_key(kind, key, wrapper, options, is_model_options=False):
    options.setdefault("wrappers", {}).setdefault(kind, {}).setdefault(key, []).append(wrapper)


pe.get_wrappers_with_key, pe.add_wrapper_with_key = get_wrappers_with_key, add_wrapper_with_key
comfy.patcher_extension = pe
sys.modules["comfy"], sys.modules["comfy.patcher_extension"] = comfy, pe

from dgx_monarch.actor import cond_batch_agreement

base = SimpleNamespace(model_options={})
cond_batch_agreement.install(base, world, topology)
wrappers = get_wrappers_with_key(pe.WrappersMP.CALC_COND_BATCH, cond_batch_agreement.WRAPPER_KEY,
                                 base.model_options)
assert len(wrappers) == 1, wrappers
wrapper = wrappers[0]

# The group each exchange runs over, by xFuser's name for it.
named = {"sp": get_sp_group().device_group, "cfg": get_cfg_group().device_group,
         "replica": get_model_replica_group().device_group}
exchanges = []
real_all_reduce = dist.all_reduce


def all_reduce(tensor, *args, group=None, **kwargs):
    name = "default" if group is None else next(
        (key for key, value in named.items() if value is group), "other")
    exchanges.append([name, sorted(dist.get_process_group_ranks(group or dist.group.WORLD))])
    return real_all_reduce(tensor, *args, group=group, **kwargs)


dist.all_reduce = all_reduce

GIB = 2 ** 30
call_groups = [named[key] for key, degree in (("sp", sp), ("cfg", cfg)) if degree > 1]
patcher = SimpleNamespace(get_free_memory=lambda device: int(scenario["free_gib"][rank] * GIB))
model = SimpleNamespace(current_patcher=patcher)
x_in = SimpleNamespace(device="cpu")
seen = []


def executor(*args, **kwargs):
    seen.append(patcher.get_free_memory("cpu") / GIB)
    # The model call's own collectives, over the ranks that share it.
    for group in call_groups:
        sizes = [torch.zeros(1, dtype=torch.int64) for _ in range(dist.get_world_size(group))]
        dist.all_gather(sizes, torch.tensor([1], dtype=torch.int64), group=group)
    return "out"


dp_rank = get_data_parallel_rank()
outcome, made = "ran", 0
try:
    for _ in range(scenario["calls"][dp_rank]):
        wrapper(executor, model, [], x_in, None, {})
        made += 1
except Exception as exc:
    outcome = "waited: " + type(exc).__name__
# A Worker whose group has finished waits for its next request, in no collective.
done = os.path.join(scenario["done_dir"], "done-%d" % rank)
open(done, "w").close()
deadline = time.monotonic() + scenario["wait_s"]
while time.monotonic() < deadline and not all(
        os.path.exists(os.path.join(scenario["done_dir"], "done-%d" % peer)) for peer in range(world)):
    time.sleep(0.05)
print(json.dumps({"rank": rank, "dp_rank": dp_rank, "outcome": outcome, "made": made, "seen": seen,
                  "exchanges": exchanges}), flush=True)
os._exit(0)  # a timed-out gloo group may not tear down cleanly
"""


def _free_port() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return str(probe.getsockname()[1])


def _run(tmp_path, topology: dict, free_gib: list[float]) -> list[dict]:
    """Every rank, each in its own process; what each reported, rank 0 first."""
    scenario = {"topology": topology, "free_gib": free_gib, "calls": CALLS,
                "init_s": INIT_S, "wait_s": WAIT_S, "done_dir": str(tmp_path)}
    port = _free_port()
    env = {**os.environ, "PYTHONPATH": str(SRC), "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
    ranks = [subprocess.Popen([sys.executable, "-c", _RANK, str(rank), port, json.dumps(scenario)],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for rank in range(len(free_gib))]
    reports = []
    try:
        for process in ranks:
            out, err = process.communicate(timeout=INIT_S + 60)
            assert process.returncode == 0, err[-3000:]
            reports.append(json.loads(out.strip().splitlines()[-1]))
    finally:
        for process in ranks:
            if process.poll() is None:
                process.kill()
                process.communicate()
    return sorted(reports, key=lambda report: report["rank"])


# Each rank's free memory: the leanest rank of dp group 0 has 2 GiB, that of
# dp group 1 has 4 GiB. The third field is the group the exchange must take:
# one the call's own collectives use where one has exactly the group's ranks.
LAYOUTS = {
    "uly2+dp2": ({"ulysses": 2, "ring": 1, "cfg": 1, "dp": 2, "fsdp": False},
                 [2.0, 10.0, 8.0, 4.0], "sp"),
    "cfg2+dp2": ({"ulysses": 1, "ring": 1, "cfg": 2, "dp": 2, "fsdp": False},
                 [2.0, 10.0, 8.0, 4.0], "cfg"),
    "uly2+cfg2+dp2": ({"ulysses": 2, "ring": 1, "cfg": 2, "dp": 2, "fsdp": False},
                      [9.0, 2.0, 10.0, 8.0, 6.0, 12.0, 4.0, 7.0], "replica"),
}


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_two_data_parallel_groups_with_different_call_counts_never_wait_on_each_other(
        tmp_path, name):
    topology, free_gib, group = LAYOUTS[name]
    reports = _run(tmp_path, topology, free_gib)
    half = len(free_gib) // 2
    assert [report["dp_rank"] for report in reports] == [0] * half + [1] * half, reports
    # Every rank finished its own group's calls; none waited in an exchange.
    assert [report["outcome"] for report in reports] == ["ran"] * len(free_gib), reports
    assert [report["made"] for report in reports] == CALLS[:1] * half + CALLS[1:] * half, reports
    # Each group still agrees: every rank of it reads its leanest rank's figure
    # on every call, and that figure is its own, not the other group's.
    assert [report["seen"] for report in reports] == (
        [[2.0] * CALLS[0]] * half + [[4.0] * CALLS[1]] * half), reports
    # Over the named group, whose ranks are exactly the rank's own dp group.
    for report in reports:
        ranks = list(range(half)) if report["dp_rank"] == 0 else list(range(half, 2 * half))
        assert report["exchanges"] == [[group, ranks]] * report["made"], reports
