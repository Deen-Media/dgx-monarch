"""Auto takes cfg-parallel only on exact, in-band measured evidence."""
from __future__ import annotations

from dataclasses import replace

import pytest

from dgx_monarch.topology import AUTO_TABLE, AutoRule, choose_auto_topology
from dgx_monarch.topology_cfg_evidence import (
    CFG2_EVIDENCE,
    KROMA_MERGE_METADATA_KEYS,
    CfgEvidence,
    cfg_evidence_for,
    cfg_row_is_eligible,
    checkpoint_variant,
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
    assert cfg_row_is_eligible(original, "bf16", None, True)
    assert not cfg_row_is_eligible(widened, "bf16", None, True)


@pytest.mark.parametrize("changed", [{"topology": {"cfg": 4}}, {"topology": {"cfg": 2, "ulysses": 2}}, {"sage": True}])
def test_cfg_evidence_cannot_authorize_different_parallel_math(changed) -> None:
    row = next(row for row in AUTO_TABLE if row.row == 20)
    assert cfg_row_is_eligible(row, "bf16", None, True)
    assert not cfg_row_is_eligible(replace(row, **changed), "bf16", None, True)


@pytest.mark.parametrize("quant", ["bf16", "fp8", "int8"])
def test_chroma_measured_quants_take_the_cfg_grant(quant: str) -> None:
    decision = choose_auto_topology("chroma", quant, 1.05, 2, cfg_value=3.5, pair_differs=True)
    assert decision.topology.cfg == 2
    assert "faster than uly2" in decision.reason
    assert "bit-identical to one GPU" in decision.reason


@pytest.mark.parametrize("quant", ["fp16", "fp32"])
def test_chroma_unmeasured_quant_falls_back_to_conservative_ulysses(quant: str) -> None:
    decision = choose_auto_topology("chroma", quant, 1.05, 2, cfg_value=3.5, pair_differs=True)
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
        ("krea2", "bf16", 1.5, 4.5, 15),
        ("qwen_image", "fp8", 1.76, 4.0, 68),
    ],
)
def test_campaign_k_rows_grant_their_measured_kinds(family, quant, megapixels, cfg, row) -> None:
    for pair_differs in (False, True):  # these records grant every prompt pair
        decision = choose_auto_topology(family, quant, megapixels, 2, cfg_value=cfg,
                                        pair_differs=pair_differs)
        assert decision.topology.describe() == "cfg2"
        assert decision.rule.row == row and decision.skipped_cfg is None
        assert f"table row {row}" in decision.reason
        assert "bit-identical to one GPU on the shipped prompt pair" in decision.reason
        assert "Campaign K 2026-10-14" in decision.reason
        assert "head capped at 2100 MHz" in decision.reason


# Issue #52, 2026-11-07: row 79 was measured on the template's pair, whose
# text rows differ, so one GPU runs it as two calls; two pairs of equal rows
# read off one GPU (check 10), so the record grants only a pair the sampler
# tagged as differing.
# Chroma's rows 20 and 19 take the same scope on their own equal-pair reading
# (BF16 NRMS 0.034); below 1.2 MP Chroma has no Ulysses row, so the fallback
# (row 99) catches an equal pair, and from 1.2 MP rows 21 (FP8) and 22 do.
# Anima's row 47 had the same scope until 2026-11-10, when the text branch fix
# made one GPU fold every Anima pair, as stock does (TROUBLESHOOTING #151).
PAIR_SCOPED = (("boogu", "fp8", 79, 78),
               ("chroma", "bf16", 20, 99), ("chroma", "int8", 20, 99))
CHROMA_BAND = (("chroma", "bf16", 1.5, 19, 22), ("chroma", "fp8", 1.63, 19, 21),
               ("chroma", "int8", 1.5, 19, 22))


def test_only_the_boogu_and_chroma_rows_are_scoped_to_pairs_of_different_text_rows() -> None:
    scoped = {row for row, record in CFG2_EVIDENCE.items() if record.pairs != "any"}
    assert scoped == {19, 20, 79}
    assert {CFG2_EVIDENCE[row].pairs for row in scoped} == {"unequal"}


@pytest.mark.parametrize(("family", "quant", "row", "fallback"), PAIR_SCOPED)
def test_a_pair_scoped_row_grants_cfg2_when_the_text_rows_differ(family, quant, row, fallback) -> None:
    decision = choose_auto_topology(family, quant, 1.05, 2, cfg_value=4.0, pair_differs=True)
    assert decision.topology.describe() == "cfg2"
    assert decision.rule.row == row and decision.skipped_cfg is None
    assert "on the shipped prompt pair, whose text rows differ" in decision.reason or (
        "bit-identical to one GPU on a prompt pair whose text rows differ" in decision.reason)
    assert "2026-11-07" in decision.reason


@pytest.mark.parametrize(("family", "quant", "megapixels", "row", "fallback"), CHROMA_BAND)
def test_chromas_upper_band_grants_a_differing_pair_and_sends_an_equal_one_to_ulysses(
        family, quant, megapixels, row, fallback) -> None:
    granted = choose_auto_topology(family, quant, megapixels, 2, cfg_value=3.5, pair_differs=True)
    assert granted.topology.describe() == "cfg2" and granted.rule.row == row
    assert "bit-identical to one GPU on a prompt pair whose text rows differ" in granted.reason
    assert "Campaign K 2026-10-14" in granted.reason and "head capped at 2100 MHz" in granted.reason
    equal = choose_auto_topology(family, quant, megapixels, 2, cfg_value=3.5)
    assert equal.topology.describe() == "uly2" and equal.rule.row == fallback
    assert equal.skipped_cfg is not None and equal.skipped_cfg.row == row


@pytest.mark.parametrize(("family", "quant", "row", "fallback"), PAIR_SCOPED)
def test_an_equal_or_unread_pair_takes_the_ulysses_row(family, quant, row, fallback) -> None:
    # The default is the unread answer: a caller that cannot tag the pair
    # gets the row that costs speed, never the one that costs fidelity.
    for kwargs in ({}, {"pair_differs": False}):
        decision = choose_auto_topology(family, quant, 1.05, 2, cfg_value=4.0, **kwargs)
        assert decision.topology.describe() == "uly2"
        assert decision.rule.row == fallback
        assert decision.skipped_cfg is not None and decision.skipped_cfg.row == row
        assert (f"cfg2 table row {row} was measured on prompt pairs whose text rows differ, "
                "and this pair's rows are equal or unread") in decision.reason
    rule = next(rule for rule in AUTO_TABLE if rule.row == row)
    assert cfg_row_is_eligible(rule, quant, None, True)
    assert not cfg_row_is_eligible(rule, quant)


def test_the_pair_clause_names_only_a_row_the_pair_alone_kept(monkeypatch) -> None:
    # A quant miss keeps its own reason; an unequal pair does not rescue it.
    for pair_differs in (False, True):
        decision = choose_auto_topology("boogu", "bf16", 1.05, 2, cfg_value=4.0,
                                        pair_differs=pair_differs)
        assert decision.rule.row == 78 and decision.skipped_cfg.row == 79
        assert "was measured on prompt pairs whose text rows differ" not in decision.reason
        assert "has no passing bf16 result" not in decision.reason  # row 78 was chosen
    # With no Ulysses row behind it, the fallback note names the pair.
    from dgx_monarch import topology

    row = AutoRule(999, "invented", (), 0.0, 1.2, 2, {"cfg": 2}, False, "")
    monkeypatch.setattr(topology, "AUTO_TABLE", (row,))
    monkeypatch.setitem(CFG2_EVIDENCE, 999, CfgEvidence(
        row=999, family="invented", quants=("bf16",), min_mp=0.0, max_mp=1.2,
        min_world=2, speedup="2x", fidelity="0.0",
        faster=True, exact=True, in_band=True, basis="test", pairs="unequal",
    ))
    decision = choose_auto_topology("invented", "bf16", 1.05, 2, cfg_value=4.0)
    assert decision.rule.row == 99
    assert "cfg2 table row 999 was measured on prompt pairs whose text rows differ" in decision.reason
    granted = choose_auto_topology("invented", "bf16", 1.05, 2, cfg_value=4.0, pair_differs=True)
    assert granted.topology.describe() == "cfg2" and granted.rule.row == 999


@pytest.mark.parametrize(
    ("family", "quant", "megapixels", "skipped", "fallback"),
    [
        ("krea2", "int8", 2.359, 15, 12),
        ("qwen_image", "bf16", 1.5, 68, 72),
        ("boogu", "bf16", 1.05, 79, 78),
    ],
)
def test_campaign_k_ungranted_kinds_fall_through_to_their_ulysses_row(
        family, quant, megapixels, skipped, fallback) -> None:
    for pair_differs in (False, True):
        decision = choose_auto_topology(family, quant, megapixels, 2, cfg_value=4.0,
                                        pair_differs=pair_differs)
        assert decision.topology.describe() == "uly2"
        assert decision.rule.row == fallback
        assert decision.skipped_cfg is not None and decision.skipped_cfg.row == skipped


@pytest.mark.parametrize("quant", ["bf16", "fp8"])
def test_anima_takes_ulysses_for_every_pair_since_its_text_branch_fix(quant) -> None:
    # 2026-11-10: a Worker pads Anima's text to 512 rows in extra_conds, as stock
    # does, so one GPU folds every pair of at most 512 T5 ids into one batch-2
    # call that cfg2's two batch-1 calls cannot reproduce; row 47 was withdrawn
    # and no cfg row remains to skip.
    assert not any(rule.family == "anima" and rule.topology.get("cfg", 1) != 1
                   for rule in AUTO_TABLE)
    assert all(record.family != "anima" for record in CFG2_EVIDENCE.values())
    for megapixels in (0.26, 1.05, 1.5):
        for pair_differs in (False, True):
            decision = choose_auto_topology("anima", quant, megapixels, 2, cfg_value=4.0,
                                            pair_differs=pair_differs)
            assert decision.topology.describe() == "uly2"
            assert decision.rule.row == 48 and decision.skipped_cfg is None
            assert "before the batching decision" in decision.reason


def test_krea2_int8_measured_exact_but_no_faster_keeps_ulysses() -> None:
    # Group c measured RAW INT8 cfg2 bit-identical at 1152, 1216 and 1280, but
    # at 0.97x, 0.98x and 1.03x against uly2 (1.08x at 1536): exact is not
    # enough, the record also needs a win at every size, so INT8 keeps row 12.
    for megapixels in (1.33, 1.48, 1.64, 2.359):
        decision = choose_auto_topology("krea2", "int8", megapixels, 2, cfg_value=4.0)
        assert decision.topology.describe() == "uly2"
        assert decision.rule.row == 12
        assert decision.skipped_cfg is not None and decision.skipped_cfg.row == 15
        assert "0.97x to 1.08x" in decision.reason


def test_kroma_is_measured_off_the_krea2_cfg2_row() -> None:
    # 2026-10-16: on Kroma's shipped ConditioningZeroOut pair one GPU folds cond
    # and uncond into one batch-2 call and cfg2 read 1-step NRMS 0.0087 to 0.0108
    # at 1152 and 1536, so row 15's record excludes Kroma and it keeps row 12.
    assert CFG2_EVIDENCE[15].excludes == ("kroma",)
    row15 = next(row for row in AUTO_TABLE if row.row == 15)
    assert cfg_row_is_eligible(row15, "bf16")
    assert not cfg_row_is_eligible(row15, "bf16", "kroma")
    for megapixels in (1.2, 1.327, 2.359):
        for cfg_value, keeps_uncond in ((1.5, False), (1.0, True)):
            kroma = choose_auto_topology("krea2", "bf16", megapixels, 2, cfg_value=cfg_value,
                                         keeps_uncond=keeps_uncond, variant="kroma")
            assert kroma.topology.describe() == "uly2" and kroma.rule.row == 12
            assert kroma.skipped_cfg is not None and kroma.skipped_cfg.row == 15
            assert "was measured on this kroma checkpoint and does not grant it" in kroma.reason
            krea2 = choose_auto_topology("krea2", "bf16", megapixels, 2, cfg_value=cfg_value,
                                         keeps_uncond=keeps_uncond)
            assert krea2.topology.describe() == "cfg2" and krea2.rule.row == 15


def test_an_excluded_variant_only_narrows_its_own_record(monkeypatch) -> None:
    row = AutoRule(999, "invented", (), 0.0, 1.2, 2, {"cfg": 2}, False, "")
    monkeypatch.setitem(CFG2_EVIDENCE, 999, CfgEvidence(
        row=999, family="invented", quants=("bf16",), min_mp=0.0, max_mp=1.2,
        min_world=2, speedup="2x", fidelity="0.0",
        faster=True, exact=True, in_band=True, basis="test", excludes=("other",),
    ))
    assert cfg_row_is_eligible(row, "bf16")
    assert cfg_row_is_eligible(row, "bf16", "kroma")
    assert not cfg_row_is_eligible(row, "bf16", "other")
    # A variant no record excludes changes no other row.
    chroma = choose_auto_topology("chroma", "bf16", 1.05, 2, cfg_value=3.5, variant="kroma",
                                  pair_differs=True)
    assert chroma.topology.cfg == 2 and chroma.rule.row == 20


def test_only_a_krea2_header_with_the_kroma_merge_metadata_reads_kroma() -> None:
    keys = tuple(sorted(KROMA_MERGE_METADATA_KEYS | {"scale", "trim"}))
    assert checkpoint_variant("krea2", keys) == "kroma"
    assert checkpoint_variant("krea2", ()) is None
    assert checkpoint_variant("krea2", ("_quantization_metadata",)) is None
    assert checkpoint_variant("krea2", ("copy_prefixes", "trim")) is None
    assert checkpoint_variant("chroma", keys) is None


@pytest.mark.parametrize(
    ("family", "quant", "edge", "row"),
    [
        ("krea2", "fp8", 2.36, 11),
        ("krea2", "bf16", 2.36, 12),
        ("chroma", "fp8", 1.64, 21),
        ("chroma", "bf16", 1.64, 22),
        ("chroma", "int8", 1.64, 22),
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
