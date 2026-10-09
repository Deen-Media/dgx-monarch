"""Unit coverage for actor/gate_cycle.run() and nodes/slab_proof._cycle_state.

run(): the two early-return guards and the lazy-swap verdict, with a fake store
whose `ensure()` returns fixed transition strings; no comfy import is needed
because model_store's comfy touches are lazy, as in
tests/test_runtime_remediation.py::test_gate_swap_cycle_rejects_replacement_between_transitions.
_cycle_state: the driver accepts one current response from every rank and
rejects duplicate, missing, non-dict, out-of-range, wrong-world and stale ones."""
from __future__ import annotations

import types

from dgx_monarch.actor import gate_cycle
from dgx_monarch.nodes.slab_proof import _cycle_state

_RESIDENT = types.SimpleNamespace(family="krea2", slab=None)


def _worker(lora_low_rss: bool, transitions: list[str]):
    calls = iter(transitions)
    store = types.SimpleNamespace(
        lora_low_rss=lora_low_rss,
        current=_RESIDENT,
        ensure=lambda *_args, **_kwargs: (object(), next(calls)),
    )
    return types.SimpleNamespace(
        _setup_key=("live",), _setup_generation=7, rank=0, world=1, store=store,
        _inject_for_topology=lambda *_args, **_kwargs: None,
    )


def test_run_returns_early_with_no_lora_stack():
    worker = _worker(lora_low_rss=True, transitions=[])
    result = gate_cycle.run(worker, "m", {}, lora_stack=None)
    assert result["conclusive"] is False
    assert result["reason"] == "no lora stack: lazy swap is not applicable"
    assert "transitions" not in result
    assert (result["rank"], result["world"], result["setup_generation"]) == (0, 1, 7)


def test_run_returns_early_with_empty_lora_stack():
    worker = _worker(lora_low_rss=True, transitions=[])
    result = gate_cycle.run(worker, "m", {}, lora_stack=[])
    assert result["conclusive"] is False
    assert result["reason"] == "no lora stack: lazy swap is not applicable"


def test_run_returns_early_when_lora_low_rss_is_off():
    worker = _worker(lora_low_rss=False, transitions=[])
    result = gate_cycle.run(worker, "m", {}, lora_stack=[{"name": "l", "strength": 1.0}])
    assert result["conclusive"] is False
    assert result["reason"] == "lora_low_rss is off on this worker"
    assert "transitions" not in result


def test_run_is_lazy_when_nudge_and_back_both_hot_swap():
    worker = _worker(lora_low_rss=True, transitions=["load", "hot-swap", "hot-swap"])
    result = gate_cycle.run(worker, "m", {}, lora_stack=[{"name": "l", "strength": 1.0}])
    assert result["conclusive"] is True
    assert result["transitions"] == {"load": "load", "nudge": "hot-swap", "back": "hot-swap"}
    assert "reason" not in result


def test_run_is_not_lazy_when_nudge_falls_back_to_reload():
    worker = _worker(lora_low_rss=True, transitions=["load", "load", "hot-swap"])
    result = gate_cycle.run(worker, "m", {}, lora_stack=[{"name": "l", "strength": 1.0}])
    assert result["conclusive"] is False
    assert result["transitions"] == {"load": "load", "nudge": "load", "back": "hot-swap"}
    assert "fell back to reload" in result["reason"]


def _cycle_row(rank: int, **overrides):
    row = {
        "rank": rank,
        "world": 2,
        "setup_generation": 7,
        "conclusive": True,
        "family": "krea2",
        "slab_active": True,
    }
    row.update(overrides)
    return row


def test_slab_cycle_accepts_one_current_response_from_every_rank():
    cycle, complete, conclusive, reasons, active, any_active, family = _cycle_state(
        [_cycle_row(0), _cycle_row(1)],
        types.SimpleNamespace(world=2, setup_generation=7),
    )

    assert len(cycle) == 2
    assert complete is True
    assert conclusive is True
    assert reasons == []
    assert active is True
    assert any_active is True
    assert family == "krea2"


def test_slab_cycle_rejects_duplicate_rank_evidence():
    _cycle, complete, conclusive, reasons, active, _any_active, _family = _cycle_state(
        [_cycle_row(0), _cycle_row(0)],
        types.SimpleNamespace(world=2, setup_generation=7),
    )

    assert conclusive is False
    assert complete is False
    assert active is False
    assert "gate swap cycle rank evidence is incomplete or duplicated" in reasons


def test_slab_cycle_rejects_missing_or_non_dict_response():
    _cycle, complete, conclusive, reasons, active, _any_active, _family = _cycle_state(
        [_cycle_row(0), None],
        types.SimpleNamespace(world=2, setup_generation=7),
    )

    assert conclusive is False
    assert complete is False
    assert active is False
    assert "gate swap cycle did not report every rank" in reasons


def test_slab_cycle_rejects_out_of_range_rank_or_wrong_world():
    _cycle, complete, conclusive, reasons, active, _any_active, _family = _cycle_state(
        [_cycle_row(0), _cycle_row(2, world=1)],
        types.SimpleNamespace(world=2, setup_generation=7),
    )

    assert conclusive is False
    assert complete is False
    assert active is False
    assert "gate swap cycle rank evidence is incomplete or duplicated" in reasons
    assert "gate swap cycle world evidence is inconsistent" in reasons


def test_slab_cycle_rejects_stale_setup_generation():
    _cycle, complete, conclusive, reasons, active, _any_active, _family = _cycle_state(
        [_cycle_row(0), _cycle_row(1, setup_generation=6)],
        types.SimpleNamespace(world=2, setup_generation=7),
    )

    assert conclusive is False
    assert complete is False
    assert active is False
    assert "gate swap cycle setup generation drifted" in reasons


def test_run_is_not_lazy_when_back_falls_back_to_reload():
    worker = _worker(lora_low_rss=True, transitions=["load", "hot-swap", "load"])
    result = gate_cycle.run(worker, "m", {}, lora_stack=[{"name": "l", "strength": 1.0}])
    assert result["conclusive"] is False
    assert result["transitions"]["back"] == "load"
    assert "fell back to reload" in result["reason"]
