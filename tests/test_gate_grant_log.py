"""A Worker logs one line naming the identity gate grant it accepted and the slots it covers.

The final hardware pass of the 1.5 round read the Ideogram 4 LoRA gate cycle
from the Worker journals: the PASS, the conditional slot's two hot-swaps and
both slots reused. No line named the grant over both model slots, so the leg
could not be checked from a journal (actor/worker_authorization.py).
"""
from __future__ import annotations

import copy
import logging

import pytest

import dgx_monarch
from dgx_monarch import gate_runtime
from dgx_monarch.actor import worker_authorization
from dgx_monarch.gate_dual_identity import dual_combo_key, dual_identity_summary
from dgx_monarch.gate_ledger import GATE_PROTOCOL_VERSION, gate_verdict_token
from dgx_monarch.mesh_residency import CAPACITY_RESCUE_CONSENT_CAPABILITY
from dgx_monarch.mesh_safety import request_combo_key

RUNTIME = {"torch": "0.0.1+cpu", "cuda": "1.0"}
TOPOLOGY = {"ulysses": 2, "ring": 1, "cfg": 1, "dp": 1, "fsdp": False}
WORKER_ARGS = {"slab_weights": True, "lora_low_rss": True}
COND = {"unet_name": "cond.safetensors", "options": {},
        "loras": [{"name": "adapter.safetensors", "strength": 0.8}]}
UNCOND = {"unet_name": "uncond.safetensors", "options": {}, "loras": []}
SETS = [{"comfy": "c", "digest": "d0", "artifacts": [{"signature": "s0"}]},
        {"comfy": "c", "digest": "d1", "artifacts": [{"signature": "s1"}]}]


class _Lines(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


@pytest.fixture
def journal(monkeypatch):
    monkeypatch.setattr(gate_runtime, "runtime_build_identity", lambda: dict(RUNTIME))
    handler = _Lines()
    worker_authorization.log.addHandler(handler)
    yield handler.lines
    worker_authorization.log.removeHandler(handler)


def _request(*, dual: bool) -> tuple[dict, list[dict]]:
    sets = SETS if dual else SETS[:1]
    context = {"worker_args": dict(WORKER_ARGS), "world": 2, "resolved_topology": dict(TOPOLOGY),
               "resolved_attention": "TORCH_FLASH", "sync_ulysses": False,
               gate_runtime.RUNTIME_FIELD: dict(RUNTIME)}
    if dual:
        key, identity = dual_combo_key(COND, UNCOND), dual_identity_summary(sets)
    else:
        key, identity = request_combo_key(COND), sets[0]
    grant = {"capability": "normal_render_residency", "gate_protocol": GATE_PROTOCOL_VERSION,
             "dgx_monarch": dgx_monarch.__version__, "comfy": identity["comfy"],
             "artifact_digest": identity["digest"],
             "gate_token": list(gate_verdict_token(key, identity["digest"], identity["comfy"], context)),
             "combo_key": key, "model_request": copy.deepcopy(COND),
             "uncond_model_request": copy.deepcopy(UNCOND) if dual else None,
             "artifact_sets": copy.deepcopy(sets), "capability_context": context,
             "setup_generation": 7, "worker_topology": dict(TOPOLOGY)}
    request = {"model": copy.deepcopy(COND), "sage_kernel": "TORCH_FLASH", "sync_ulysses": False,
               "_dgxm_normal_residency_mode": "required", "_dgxm_normal_residency_grant": grant,
               "_dgxm_artifact_sets": copy.deepcopy(sets)}
    if dual:
        request["uncond_model"] = copy.deepcopy(UNCOND)
    return request, sets


def _worker(rank: int = 1):
    import types

    return types.SimpleNamespace(rank=rank, world=2, topology=dict(TOPOLOGY),
                                 _setup_generation=7, _active_worker_args=dict(WORKER_ARGS))


def _verify(request: dict, sets: list[dict], rank: int = 1) -> list[dict]:
    identities = {spec["unet_name"]: item for spec, item in zip(
        [COND, UNCOND], sets, strict=False)}
    return worker_authorization.verify_sample_artifact_authorization(
        _worker(rank), request, lambda unet, _loras: identities[unet])


def test_a_dual_grant_the_worker_accepts_names_both_slots_in_one_line(journal):
    request, sets = _request(dual=True)
    assert _verify(request, sets) == sets
    assert journal == [
        "identity gate grant accepted on rank 1: the gate PASS covers both model slots, "
        "[cond] cond.safetensors with 1 LoRA and [uncond] uncond.safetensors with no LoRA; "
        "this render keeps the residency the gate proved"]


def test_a_single_slot_grant_names_its_one_slot(journal):
    request, sets = _request(dual=False)
    assert _verify(request, sets, rank=0) == sets
    assert journal == [
        "identity gate grant accepted on rank 0: the gate PASS covers its one model slot, "
        "[cond] cond.safetensors with 1 LoRA; this render keeps the residency the gate proved"]


def test_the_line_names_no_lora_file(journal):
    request, sets = _request(dual=True)
    _verify(request, sets)
    assert journal and not any("adapter.safetensors" in line for line in journal)


@pytest.mark.parametrize("mutation", [
    lambda request: request["_dgxm_normal_residency_grant"].update(artifact_digest="forged"),
    lambda request: request.update(_dgxm_artifact_sets=[SETS[1], SETS[0]]),
])
def test_a_refused_grant_logs_no_acceptance(journal, mutation):
    request, sets = _request(dual=True)
    mutation(request)
    with pytest.raises(RuntimeError):
        _verify(request, sets)
    assert journal == []


def test_a_render_with_no_grant_or_a_rescue_consent_logs_no_gate_line():
    request, _sets = _request(dual=True)
    request.pop("_dgxm_normal_residency_grant")
    assert worker_authorization.gate_grant_line(1, request) is None
    request["_dgxm_normal_residency_grant"] = {"capability": CAPACITY_RESCUE_CONSENT_CAPABILITY}
    assert worker_authorization.gate_grant_line(1, request) is None
