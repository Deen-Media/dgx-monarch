"""Auto takes cfg-parallel only on exact, in-band measured evidence."""
from __future__ import annotations

from dataclasses import replace

import pytest

from dgx_monarch.topology import AUTO_TABLE, AutoRule, choose_auto_topology
from dgx_monarch.topology_cfg_evidence import (
    CFG2_EVIDENCE,
    CfgEvidence,
    cfg_evidence_for,
    cfg_row_is_eligible,
)

CFG_ROWS = tuple(row for row in AUTO_TABLE if int(row.topology.get("cfg", 1)) > 1)


def test_cfg_rows_all_have_exact_eligible_records() -> None:
    assert CFG_ROWS
    for row in CFG_ROWS:
        record = cfg_evidence_for(row)
        assert record is not None, row.family
        assert record.row == row.row
        assert record.family == row.family
        assert record.min_mp == row.mp_min
        assert record.max_mp == row.mp_max
        assert record.min_world == row.world_min
        assert record.in_band and record.faster and record.exact


def test_missing_record_never_authorizes_cfg(monkeypatch) -> None:
    row = AutoRule(999, "invented", (), 0.0, 1.2, 2, {"cfg": 2}, False, "")
    assert not cfg_row_is_eligible(row, "bf16")
    monkeypatch.setitem(CFG2_EVIDENCE, 999, CfgEvidence(
        row=999, family="invented", quants=("bf16",), min_mp=0.0, max_mp=1.2,
        min_world=2, speedup="2x", fidelity="0.0",
        faster=True, exact=True, in_band=False, basis="outside the row",
    ))
    assert not cfg_row_is_eligible(row, "bf16")


@pytest.mark.parametrize("field", ["faster", "exact", "in_band"])
def test_each_required_evidence_field_gates_cfg(monkeypatch, field: str) -> None:
    row = AutoRule(999, "invented", (), 0.0, 1.2, 2, {"cfg": 2}, False, "")
    values = {"faster": True, "exact": True, "in_band": True}
    values[field] = False
    monkeypatch.setitem(CFG2_EVIDENCE, 999, CfgEvidence(
        row=999, family="invented", quants=("bf16",), min_mp=0.0, max_mp=1.2,
        min_world=2, speedup="2x", fidelity="0.0",
        basis="test", **values,
    ))
    assert not cfg_row_is_eligible(row, "bf16")


@pytest.mark.parametrize(
    "changed",
    [
        {"mp_max": 1.3},
        {"mp_min": -0.1},
        {"world_min": 1},
    ],
)
def test_a_row_cannot_broaden_the_evidence_scope(changed: dict[str, float | int]) -> None:
    original = next(row for row in AUTO_TABLE if row.row == 20)
    widened = AutoRule(
        original.row, original.family, original.quants,
        changed.get("mp_min", original.mp_min),
        changed.get("mp_max", original.mp_max),
        changed.get("world_min", original.world_min),
        original.topology, original.sage, original.note,
    )
    assert not cfg_row_is_eligible(widened, "bf16")


@pytest.mark.parametrize("changed", [{"topology": {"cfg": 4}}, {"topology": {"cfg": 2, "ulysses": 2}}, {"sage": True}])
def test_cfg_evidence_cannot_authorize_different_parallel_math(changed) -> None:
    row = next(row for row in AUTO_TABLE if row.row == 20)
    assert not cfg_row_is_eligible(replace(row, **changed), "bf16")


@pytest.mark.parametrize("quant", ["bf16", "fp8", "int8"])
def test_chroma_measured_quants_take_the_cfg_grant(quant: str) -> None:
    decision = choose_auto_topology("chroma", quant, 1.05, 2, cfg_value=3.5)
    assert decision.topology.cfg == 2
    assert "faster than uly2" in decision.reason
    assert "bit-identical to one GPU" in decision.reason


@pytest.mark.parametrize("quant", ["fp16", "fp32"])
def test_chroma_unmeasured_quant_falls_back_to_conservative_ulysses(quant: str) -> None:
    decision = choose_auto_topology("chroma", quant, 1.05, 2, cfg_value=3.5)
    assert decision.topology.ulysses == 2
    assert decision.topology.cfg == 1
    assert f"no passing {quant} result" in decision.reason


@pytest.mark.parametrize("family", ["flux", "flux2", "krea2", "lens", "qwen_image", "zimage"])
def test_unmeasured_cfg_rows_use_ulysses_until_their_own_evidence_exists(family: str) -> None:
    decision = choose_auto_topology(family, "bf16", 1.0, 2, cfg_value=3.5)
    assert decision.topology.ulysses == 2
    assert decision.topology.cfg == 1


def test_longcat_still_uses_its_in_band_bf16_cfg_evidence() -> None:
    decision = choose_auto_topology("longcat", "bf16", 1.05, 2, cfg_value=3.5)
    assert decision.topology.cfg == 2
    assert "2.00x faster" in decision.reason
