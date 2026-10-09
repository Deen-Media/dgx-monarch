"""Fail-closed unit coverage for the FSDP clean-reload cycle."""
from __future__ import annotations

import types

import pytest

from dgx_monarch.actor import gate_fsdp_cycle, model_store
from dgx_monarch.actor.store_detect import LivePrecisionEvidence


def _worker(expected, transition="load", *, ready=True, slab=None):
    events = []
    precision = LivePrecisionEvidence(
        quant_kind="bf16",
        live_dtype_profile="all_bf16",
        checkpoint_kind="bf16",
    )

    def resident():
        return types.SimpleNamespace(
            family="wan",
            slab=slab,
            artifact_identity=expected,
            quant_kind="bf16",
            precision_evidence=precision,
            base_patcher=types.SimpleNamespace(_dgxm_fsdp_ready=ready),
        )

    store = types.SimpleNamespace(current=resident())

    def unload_all():
        events.append("unload")
        store.current = None

    def ensure(*_args, **kwargs):
        events.append(("ensure", kwargs.get("fsdp_launch")))
        store.current = resident()
        return object(), transition

    store.unload_all = unload_all
    store.ensure = ensure
    worker = types.SimpleNamespace(
        _setup_key=("live",),
        _setup_generation=4,
        rank=0,
        topology={"fsdp": True},
        store=store,
        _inject_for_topology=lambda *_args, **_kwargs: None,
    )
    return worker, events


def test_fsdp_reload_cycle_requires_bound_artifact_identity():
    worker, events = _worker({"digest": "expected"})
    with pytest.raises(RuntimeError, match="requires an artifact identity"):
        gate_fsdp_cycle.run(worker, "m", {}, expected_artifact_identity=None)
    assert events == []


def test_fsdp_reload_cycle_proves_one_fresh_ready_non_slab_load(monkeypatch):
    expected = {"digest": "expected"}
    worker, events = _worker(expected)
    monkeypatch.setattr(
        model_store,
        "request_artifact_identity",
        lambda *_args: expected,
    )

    result = gate_fsdp_cycle.run(
        worker,
        "m",
        {"weight_dtype": "bf16"},
        expected_artifact_identity=expected,
    )

    assert result["conclusive"] is True
    assert result["rank"] == 0
    assert result["setup_generation"] == 4
    assert result["proof"] == "fsdp_clean_reload"
    assert result["baseline_verified"] is True
    assert result["baseline_live_dtype_profile"] == "all_bf16"
    assert result["baseline_checkpoint_precision"] == "bf16"
    assert result["transitions"] == {"reload": "load"}
    assert result["fsdp_ready"] is True
    assert result["artifact_identity_verified"] is True
    assert events == ["unload", ("ensure", True)]


@pytest.mark.parametrize(
    ("transition", "ready", "slab"),
    [
        ("reuse", True, None),
        ("load", False, None),
        ("load", True, object()),
    ],
)
def test_fsdp_reload_cycle_rejects_incomplete_evidence(
    monkeypatch,
    transition,
    ready,
    slab,
):
    expected = {"digest": "expected"}
    worker, _events = _worker(
        expected,
        transition=transition,
        ready=ready,
        slab=slab,
    )
    monkeypatch.setattr(
        model_store,
        "request_artifact_identity",
        lambda *_args: expected,
    )

    result = gate_fsdp_cycle.run(
        worker,
        "m",
        {},
        expected_artifact_identity=expected,
    )

    assert result["conclusive"] is False
    assert "fresh, matching, ready shard set" in result["reason"]
