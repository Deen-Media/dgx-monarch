"""Two ranks with different free memory batch their conds the same way.

The 2026-11-04 Wan 2.2 text-to-video identity gate stalled for 600 s: rank 1's
all-to-all and all-gather carried exactly twice rank 0's elements. The cause
these tests pin: ComfyUI's ``_calc_cond_batch`` runs cond and uncond as one
call only when ``memory_required * 1.5`` is under the free memory its patcher
reports, and on a pair of Sparks each rank reports its own box. The real
ComfyUI tests run in subprocesses on CPU (``--cpu``, no CUDA device visible)
and skip where ComfyUI is not importable; the two-rank one uses a Gloo group
on the loopback device. ``actor/cond_batch_agreement.py`` is the fix.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch.actor import cond_batch_agreement as agreement

SRC = Path(__file__).resolve().parents[1] / "src"
COMFY_DIR = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
GIB = 2 ** 30

# One child per rank: ComfyUI's own calc_cond_batch over a stub model whose
# memory_required is one GiB per batch row, with two conds that concatenate.
_CHILD = r'''
import datetime, json, os, sys, uuid
rank, world, init_file, free_gib, pin = (int(sys.argv[1]), int(sys.argv[2]), sys.argv[3],
                                         float(sys.argv[4]), sys.argv[5] == "pin")
sys.argv = ["cond-batch-agreement", "--cpu"]
import comfy.options
comfy.options.args_parsing = True
import torch
import torch.distributed as dist
if world > 1:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank,
                            world_size=world, timeout=datetime.timedelta(seconds=60))
import comfy.conds
import comfy.patcher_extension as pe
import comfy.samplers
from dgx_monarch.actor import cond_batch_agreement

GIB = 2 ** 30

class Patcher:
    def __init__(self, free):
        self.free = free
    def get_free_memory(self, device):
        return self.free
    def prepare_state(self, timestep, model_options):
        pass
    def apply_hooks(self, hooks):
        return {}

class Model:
    def __init__(self, free):
        self.current_patcher = Patcher(free)
        self.batches, self.peer_batches = [], []
    def memory_required(self, input_shape, cond_shapes=None):
        return input_shape[0] * GIB
    def apply_model(self, x, t, **c):
        self.batches.append(int(x.shape[0]))
        if pin and world > 1:
            # A sequence-parallel forward's collective: its size follows the batch.
            sizes = [torch.zeros(1, dtype=torch.int64) for _ in range(world)]
            dist.all_gather(sizes, torch.tensor([x.shape[0]], dtype=torch.int64))
            self.peer_batches.append([int(s.item()) for s in sizes])
        return torch.zeros_like(x)

def cond(value):
    return [{"model_conds": {"c_crossattn": comfy.conds.CONDCrossAttn(torch.full((1, 4, 8), value))},
             "uuid": uuid.uuid4()}]

model = Model(int(free_gib * GIB))
options = {}
if pin:
    pe.add_wrapper_with_key(pe.WrappersMP.CALC_COND_BATCH, cond_batch_agreement.WRAPPER_KEY,
                            cond_batch_agreement.make_wrapper(), options, is_model_options=True)
x_in = torch.zeros(1, 4, 8, 8)
for step in range(2):
    comfy.samplers.calc_cond_batch(model, [cond(1.0), cond(0.0)], x_in, torch.tensor([1.0]), options)
print(json.dumps({"rank": rank, "batches": model.batches, "peer_batches": model.peer_batches,
                  "override_left": "get_free_memory" in model.current_patcher.__dict__}))
if world > 1:
    dist.destroy_process_group()
'''


def _comfy_or_skip() -> None:
    if not os.path.isdir(os.path.join(COMFY_DIR, "comfy")):
        pytest.skip("no ComfyUI checkout (set COMFYUI_DIR); the comfy canary job covers it")


def _env() -> dict:
    return os.environ | {"PYTHONPATH": f"{COMFY_DIR}:{SRC}", "CUDA_VISIBLE_DEVICES": ""}


def _ranks(tmp_path, frees_gib, pin):
    import json

    init_file = str(tmp_path / "gloo-init")
    children = [subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(rank), str(len(frees_gib)), init_file,
         str(free), "pin" if pin else "stock"],
        env=_env(), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for rank, free in enumerate(frees_gib)]
    answers = []
    for child in children:
        out, err = child.communicate(timeout=240)
        assert child.returncode == 0, err[-4000:]
        answers.append(json.loads(out.strip().splitlines()[-1]))
    return sorted(answers, key=lambda answer: answer["rank"])


def test_comfy_batches_cond_and_uncond_by_each_box_free_memory(tmp_path):
    """The hypothesis, on ComfyUI's own code: 2 GiB free splits, 10 GiB batches.

    Each rank runs alone here (world 1), so nothing pins the choice: the short
    box runs two one-row calls per step and the roomy box one two-row call.
    Under Ulysses those are collectives of different sizes and counts.
    """
    _comfy_or_skip()
    short, roomy = (_ranks(tmp_path / name, [free], pin=False)[0]
                    for name, free in (("short", 2.0), ("roomy", 10.0)))
    assert short["batches"] == [1, 1, 1, 1]
    assert roomy["batches"] == [2, 2]


def test_two_ranks_take_the_leanest_rank_batch_with_the_agreement(tmp_path):
    """The fix, on a real two-rank Gloo group: both ranks split like the short one.

    Each model call also runs one all-gather of its batch size, the way a
    sequence-parallel forward's collectives follow the batch; it completes and
    every rank reads the same sizes on every call. The free-memory override is
    gone after each call.
    """
    _comfy_or_skip()
    (tmp_path / "pair").mkdir()
    head, sibling = _ranks(tmp_path / "pair", [2.0, 10.0], pin=True)
    assert head["batches"] == sibling["batches"] == [1, 1, 1, 1]
    assert head["peer_batches"] == sibling["peer_batches"] == [[1, 1]] * 4
    assert not head["override_left"] and not sibling["override_left"]


class _Patcher:
    def __init__(self, free):
        self.free = free

    def get_free_memory(self, device):
        return self.free


def _call(wrapper, patcher, seen, fail=False):
    model = SimpleNamespace(current_patcher=patcher)
    x_in = SimpleNamespace(device="cpu")

    def executor(*args, **kwargs):
        seen.append(patcher.get_free_memory("cpu"))
        if fail:
            raise RuntimeError("the model call failed")
        return "out"

    return wrapper(executor, model, [], x_in, None, {})


def test_the_pin_lasts_one_call_and_is_undone_on_a_failure(monkeypatch):
    monkeypatch.setattr(agreement, "_world", lambda: 2)
    monkeypatch.setattr(agreement, "leanest_free_memory", lambda local: min(local, 3 * GIB))
    patcher, seen = _Patcher(10 * GIB), []
    wrapper = agreement.make_wrapper()
    assert _call(wrapper, patcher, seen) == "out"
    with pytest.raises(RuntimeError, match="model call failed"):
        _call(wrapper, patcher, seen, fail=True)
    assert seen == [3 * GIB, 3 * GIB]
    assert "get_free_memory" not in patcher.__dict__
    assert patcher.get_free_memory("cpu") == 10 * GIB


def test_world_one_reads_its_own_box_and_never_exchanges(monkeypatch):
    real = agreement.leanest_free_memory
    assert real(5) == 5 and real(-3) == 0  # no process group in this process
    monkeypatch.setattr(agreement, "_world", lambda: 1)
    monkeypatch.setattr(agreement, "leanest_free_memory",
                        lambda local: pytest.fail("world 1 must not exchange"))
    patcher, seen = _Patcher(7 * GIB), []
    assert _call(agreement.make_wrapper(), patcher, seen) == "out"
    assert seen == [7 * GIB]


def test_install_is_world_gated_idempotent_and_rides_the_partial_load_guard():
    pe = pytest.importorskip(
        "comfy.patcher_extension", reason="no ComfyUI on sys.path (canary job covers it)")
    from dgx_monarch.actor import partial_load_guard

    def installed(patcher):
        return pe.get_wrappers_with_key(
            pe.WrappersMP.CALC_COND_BATCH, agreement.WRAPPER_KEY,
            patcher.model_options, is_model_options=True)

    solo = SimpleNamespace(model_options={})
    partial_load_guard.install(solo, 1)
    assert installed(solo) == []
    pair = SimpleNamespace(model_options={})
    partial_load_guard.install(pair, 2)
    partial_load_guard.install(pair, 2)
    assert len(installed(pair)) == 1
