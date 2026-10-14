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


# Campaign K (2026-10-14): one record per row, each granting only the kinds it
# measured at every size in its band.

@pytest.mark.parametrize(
    ("family", "quant", "megapixels", "cfg", "row"),
    [
        ("krea2", "fp8", 1.2, 4.5, 15),
        ("krea2", "fp8", 2.359, 4.5, 15),
        ("chroma", "bf16", 1.5, 3.5, 19),
        ("chroma", "fp8", 1.63, 3.5, 19),
        ("qwen_image", "fp8", 1.76, 4.0, 68),
        ("boogu", "fp8", 1.05, 4.0, 79),
        ("anima", "bf16", 1.05, 4.0, 47),
    ],
)
def test_campaign_k_rows_grant_their_measured_kinds(family, quant, megapixels, cfg, row) -> None:
    decision = choose_auto_topology(family, quant, megapixels, 2, cfg_value=cfg)
    assert decision.topology.describe() == "cfg2"
    assert decision.rule.row == row and decision.skipped_cfg is None
    assert f"table row {row}" in decision.reason
    assert "bit-identical to one GPU on the shipped prompt pair" in decision.reason
    assert "Campaign K 2026-10-14" in decision.reason
    assert "head capped at 2100 MHz" in decision.reason


@pytest.mark.parametrize(
    ("family", "quant", "megapixels", "skipped", "fallback"),
    [
        ("krea2", "bf16", 1.5, 15, 12),
        ("krea2", "int8", 2.359, 15, 12),
        ("chroma", "int8", 1.5, 19, 22),
        ("qwen_image", "bf16", 1.5, 68, 72),
        ("boogu", "bf16", 1.05, 79, 78),
        ("anima", "fp8", 1.05, 47, 48),
    ],
)
def test_campaign_k_unmeasured_kinds_fall_through_to_their_ulysses_row(
        family, quant, megapixels, skipped, fallback) -> None:
    decision = choose_auto_topology(family, quant, megapixels, 2, cfg_value=4.0)
    assert decision.topology.describe() == "uly2"
    assert decision.rule.row == fallback
    assert decision.skipped_cfg is not None and decision.skipped_cfg.row == skipped


@pytest.mark.parametrize(
    ("family", "quant", "edge", "row"),
    [
        ("krea2", "fp8", 2.36, 11),
        ("chroma", "fp8", 1.64, 21),
        ("chroma", "bf16", 1.64, 22),
        ("qwen_image", "fp8", 1.77, 71),
        ("boogu", "fp8", 1.2, 78),
        ("anima", "bf16", 1.2, 48),
    ],
)
def test_campaign_k_bands_stop_at_the_last_measured_size(family, quant, edge, row) -> None:
    decision = choose_auto_topology(family, quant, edge, 2, cfg_value=4.0)
    assert decision.topology.cfg == 1
    assert decision.rule.row == row


def test_kandinsky5_and_longcat_above_1_2_mp_keep_ulysses() -> None:
    # Kandinsky5 image and video share one family and kind, so its measured
    # explicit cfg2 earns no row; LongCat's band cells read cfg2 off one GPU
    # where uly2 read bit-identical, so no LongCat row above 1.2 MP.
    kandinsky = choose_auto_topology("kandinsky5", "bf16", 1.05, 2, cfg_value=3.5)
    assert kandinsky.topology.describe() == "uly2" and kandinsky.rule.row == 44
    assert "so there is no cfg2 row" in kandinsky.reason
    longcat = choose_auto_topology("longcat", "bf16", 1.5, 2, cfg_value=4.0)
    assert longcat.topology.describe() == "uly2" and longcat.rule.row == 29


def test_the_auto_sage_routes_carry_their_campaign_k_reading() -> None:
    krea2 = choose_auto_topology("krea2", "fp8", 2.4, 2, cfg_value=4.0)
    assert krea2.rule.row == 11 and krea2.sage is True
    assert "had no reading" not in krea2.reason
    assert "1-step NRMS 0.030 to 0.058" in krea2.reason
    chroma = choose_auto_topology("chroma", "fp8", 2.4, 2, cfg_value=3.5)
    assert chroma.rule.row == 21 and "1-step NRMS 0.057 to 0.078" in chroma.reason
