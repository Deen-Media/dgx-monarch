"""Topology dataclass + auto-mode table behavior."""
import hashlib

import pytest

from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.topology import (
    AUTO_TABLE,
    PRESETS,
    Topology,
    choose_auto_topology,
    topology_from_preset,
)


def test_validate_divisibility():
    Topology(ulysses=2, world=2).validate()
    Topology(ulysses=2, world=4).with_derived_dp().validate()
    with pytest.raises(ValueError):
        Topology(ulysses=3, world=4).validate()
    with pytest.raises(ValueError):
        Topology(ulysses=0, world=2).validate()


def test_validate_rejects_undersubscribed_dp():
    # world != dp*ulysses*ring*cfg would only fail later in an xfuser assert.
    with pytest.raises(ValueError):
        Topology(ulysses=2, world=4).validate()


def test_derived_dp():
    topo = Topology(ulysses=2, world=4).with_derived_dp()
    assert topo.dp == 2
    assert topo.model_parallel == 2


def test_describe():
    assert Topology(world=1).describe() == "single"
    assert Topology(ulysses=2, cfg=2, world=4).describe() == "uly2+cfg2"


def test_presets_all_valid_at_world_8():
    for name in PRESETS:
        if name == "ring2+fsdp":
            continue  # refuses by design since 2026-08-27, pinned below
        topo = topology_from_preset(name, 8)
        topo.validate()


def test_ring_fsdp_refuses_typed():
    """A reload changes ring attention output (measured 2026-08-27: NRMS
    0.004436 on flux1 bf16 with uly2 exact), so the clean-reload gate can
    never certify ring+fsdp and the topology refuses it before any load."""
    from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag

    with pytest.raises(ValueError) as raised:
        topology_from_preset("ring2+fsdp", world=2)
    tag = parse_leading_refusal_tag(str(raised.value))
    assert tag is not None and tag.refusal_class is RefusalClass.KNOWN_WRONG
    # Guard-less on purpose: registering a guard needs a frozen-vocabulary
    # kind, a new kind burns the current protocol number and must ship the
    # next one, and no waiver is designed.
    assert tag.guard is None
    assert "uly2+fsdp" in str(raised.value) and "#340" in str(raised.value)


def test_auto_rejects_unresolved():
    with pytest.raises(ValueError):
        topology_from_preset("auto", 2)


# Rules seeded from the benchmarks in docs/DESIGN.md Appendix A.

def test_krea2_crossover():
    low = choose_auto_topology("krea2", "fp8", 1.0, 2, cfg_value=4.0)
    assert low.topology.cfg == 1 and low.topology.ulysses == 2
    high = choose_auto_topology("krea2", "fp8", 2.4, 2, cfg_value=4.0)
    assert high.topology.ulysses == 2 and high.topology.cfg == 1


def test_cfg_parallel_skipped_at_cfg_one():
    # cfg2 has no uncond pass to parallelize at cfg == 1.0.
    decision = choose_auto_topology("krea2", "fp8", 1.0, 2, cfg_value=1.0)
    assert decision.topology.cfg == 1
    assert decision.topology.sequence_parallel == 2


def test_sage_gate_fp8_only_high_res():
    fp8_high = choose_auto_topology("krea2", "fp8", 2.4, 2, cfg_value=4.0)
    assert fp8_high.sage is True
    bf16_high = choose_auto_topology("krea2", "bf16", 2.4, 2, cfg_value=4.0)
    assert bf16_high.sage is False
    fp8_low = choose_auto_topology("krea2", "fp8", 1.0, 2, cfg_value=4.0)
    assert fp8_low.sage is False


@pytest.mark.parametrize("megapixels", [1.2, 1.76, 2.4, 8.0])
def test_qwen_image_fp8_auto_keeps_the_declared_kernel(megapixels):
    """In the 2026-09 campaign SAGE at Ulysses read 1-step NRMS 0.162 on the
    shipped graph (docs/VALIDATION.md, 2026-10-05 record),
    over the 0.10 floor, while TORCH_FLASH read 0.082. Row 71 keeps uly2 and
    leaves the Init node's kernel in place."""
    decision = choose_auto_topology(
        "qwen_image", "fp8", megapixels, 2, cfg_value=4.0)
    assert decision.rule.row == 71
    assert decision.topology.ulysses == 2 and decision.topology.cfg == 1
    assert decision.sage is False
    assert "table row 71" in decision.reason
    assert "f43477b46788" in decision.reason


@pytest.mark.parametrize("quant", ["fp8", "bf16"])
def test_ideogram_auto_uses_ulysses_never_cfg(quant):
    for mp in (0.5, 1.0, 2.4):
        decision = choose_auto_topology("ideogram4", quant, mp, 2, cfg_value=4.0)
        assert decision.topology.ulysses == 2
        assert decision.topology.ring == 1
        assert decision.topology.cfg == 1
        assert "NRMS 0.000" in decision.reason
        assert "tested fp8 pair only" in decision.reason


def test_zimage_cfg_auto_reason_matches_asymmetric_trim_contract():
    decision = choose_auto_topology("zimage", "bf16", 1.0, 2, cfg_value=4.0)
    assert decision.topology.cfg == 1 and decision.topology.sequence_parallel == 2
    assert "no zimage table row matches" in decision.reason


def test_zimage_ulysses_auto_reason_records_hardware_result():
    decision = choose_auto_topology("zimage", "bf16", 2.4, 2, cfg_value=4.0)
    assert decision.topology.ulysses == 2 and decision.topology.cfg == 1
    assert "NRMS 0.000" in decision.reason
    assert "pending benchmark" not in decision.reason


def test_lens_bf16_auto_reason_records_repaired_hardware_result():
    decision = choose_auto_topology("lens", "bf16", 2.4, 2, cfg_value=4.0)
    assert decision.topology.ulysses == 2 and decision.topology.cfg == 1
    assert "NRMS 0.060-0.061" in decision.reason
    assert "2026-10-06 bf16 uly2 bit-identical" in decision.reason
    assert "re-gate pending" not in decision.reason


@pytest.mark.parametrize(
    ("family", "quant", "megapixels", "evidence"),
    [
        ("flux", "bf16", 2.4, "2026-10-06 standard Flux 1 Dev uly2 bit-identical"),
        ("longcat", "bf16", 1.0, "NRMS 0.009"),
        ("longcat", "bf16", 2.4, "2026-10-06 LongCat BF16 uly2 bit-identical"),
        ("boogu", "fp8", 1.0, "2026-10-06 fp8 uly2 bit-identical"),
        ("qwen_image", "fp8", 2.4, "2026-10-06 base 2512 FP8 uly2 bit-identical"),
        ("krea2", "fp8", 2.4, "2026-10-06 Turbo fp8 uly2 on TORCH_FLASH bit-identical"),
        ("krea2", "bf16", 2.4, "2026-10-06 RAW BF16 and INT8 uly2 bit-identical"),
        ("ideogram4", "fp8", 2.4, "2026-10-06 fp8 conditional model uly2 at CFG 1 bit-identical"),
        ("hunyuan", "fp8", 2.4, "2026-10-06 Image 2.1 FP8 and Video 1.5 FP16 uly2 bit-identical"),
        ("ernie", "bf16", 2.4, "2026-10-06 BF16 uly2 bit-identical"),
    ],
)
def test_recent_hardware_auto_reasons_are_not_stale(
    family, quant, megapixels, evidence
):
    decision = choose_auto_topology(family, quant, megapixels, 2, cfg_value=4.0)
    assert evidence in decision.reason
    assert "pending" not in decision.reason


def test_video_families_usp():
    for family in ("ltx", "wan"):
        decision = choose_auto_topology(family, "fp8", 0.8, 2, cfg_value=4.0)
        assert decision.topology.sequence_parallel == 2


@pytest.mark.parametrize(
    ("family", "evidence"),
    [
        ("hunyuan", "HW-vouched for Image 2.1, refiner, and Video 1.5"),
        ("kandinsky5", "HW-vouched for Image and Video Lite"),
    ],
)
def test_hw_promoted_family_auto_reasons_are_not_stale(family, evidence):
    decision = choose_auto_topology(family, "fp8", 1.0, 2, cfg_value=4.0)
    assert decision.topology.ulysses == 2
    assert "unbenchmarked" not in decision.reason
    assert evidence in decision.reason


@pytest.mark.parametrize("quant", ["bf16", "fp16"])
def test_fsdp_only_for_capacity(quant):
    fits = choose_auto_topology("wan", quant, 1.0, 2, cfg_value=4.0, fits_resident=True)
    assert fits.topology.fsdp is False
    capacity = choose_auto_topology("wan", quant, 1.0, 2, cfg_value=4.0, fits_resident=False)
    assert capacity.topology.fsdp is True


def test_world_one_is_single():
    decision = choose_auto_topology("krea2", "fp8", 1.0, 1, cfg_value=4.0)
    assert decision.topology.describe() == "single"


def test_auto_batch_one_never_derives_dp():
    # Batch-1 renders are the default; spare ranks fold into ring, not dp.
    for world in (2, 4, 8):
        decision = choose_auto_topology("krea2", "fp8", 2.4, world, cfg_value=4.0, batch_size=1)
        assert decision.topology.dp == 1, f"world {world} derived dp"
        decision.topology.validate()


def test_auto_divisible_batch_keeps_dp():
    decision = choose_auto_topology("krea2", "fp8", 2.4, 4, cfg_value=4.0, batch_size=2)
    assert decision.topology.dp == 2


def test_auto_odd_world_falls_back_to_ring():
    decision = choose_auto_topology("krea2", "fp8", 2.4, 3, cfg_value=4.0, batch_size=1)
    assert decision.topology.ring == 3
    assert decision.topology.dp == 1
    decision.topology.validate()


def test_auto_cfg_one_reason_names_the_family():
    decision = choose_auto_topology("krea2", "fp8", 1.0, 2, cfg_value=1.0)
    assert "unknown family" not in decision.reason
    assert "krea2" in decision.reason
    assert decision.topology.ulysses == 2


def test_auto_no_matching_row_reason_does_not_blame_cfg_when_cfg_is_not_the_cause():
    # A negative megapixels value matches no krea2 row (mp_min <= megapixels
    # fails for every row), and the cfg == 1.0 guard excludes none, so the
    # reason must not blame cfg 1.0.
    decision = choose_auto_topology("krea2", "bf16", -1.0, 2, cfg_value=None)
    assert "unknown family" not in decision.reason
    assert "cfg 1.0" not in decision.reason
    assert "krea2" in decision.reason
    assert "no krea2 table row matches" in decision.reason
    assert decision.topology.ulysses == 2  # still the safe uly2 fallback


def test_auto_int8_never_selects_sage():
    decision = choose_auto_topology("krea2", "int8", 2.4, 2, cfg_value=4.0)
    assert decision.sage is False
    chroma = choose_auto_topology("chroma", "int8", 2.4, 2, cfg_value=4.0)
    assert chroma.sage is False


def test_every_rule_reason_carries_row_number():
    decision = choose_auto_topology("chroma", "bf16", 0.5, 2, cfg_value=4.0)
    assert "table row 20" in decision.reason


def test_pixeldit_auto_routes_to_validated_uly2_on_even_batch():
    decision = choose_auto_topology("pixeldit_comfy", "bf16", 4.0, 2, cfg_value=1.0, batch_size=2)
    assert decision.topology.dp == 1
    assert decision.topology.ulysses == 2
    assert decision.topology.cfg == 1
    assert "table row 31" in decision.reason
    assert "NRMS 0.000" in decision.reason


def test_pixeldit_auto_routes_to_validated_uly2_on_batch_one():
    decision = choose_auto_topology(
        "pixeldit_comfy", "bf16", 4.0, 2, cfg_value=1.0, batch_size=1
    )
    assert decision.topology.ulysses == 2
    assert decision.topology.dp == 1


def test_pixeldit_auto_uses_generic_ring_fallback_on_odd_world():
    decision = choose_auto_topology(
        "pixeldit_comfy", "bf16", 4.0, 3, cfg_value=1.0, batch_size=1
    )
    assert decision.topology.ring == 3
    assert decision.topology.dp == 1


@pytest.mark.parametrize("quant", ["bf16", "fp8", "fp16", "int8"])
@pytest.mark.parametrize("megapixels", [0.1, 0.26, 1.03, 4.0, 12.0])
@pytest.mark.parametrize("cfg_value", [None, 1.0, 4.0])
def test_minimax_h3_auto_is_uly2_everywhere(quant, megapixels, cfg_value):
    """Row 53 is one full-range, quant-agnostic row: no measurement splits H3
    by megapixels. Every world-2 resolution must land on plain uly2 whatever
    the quant, the area, or the CFG scale.
    """
    decision = choose_auto_topology(
        "minimax_h3", quant, megapixels, 2, cfg_value=cfg_value)
    assert decision.topology.ulysses == 2
    assert decision.topology.ring == 1
    assert decision.topology.cfg == 1
    assert decision.topology.dp == 1
    assert decision.sage is False
    assert "table row 53" in decision.reason
    decision.topology.validate()


@pytest.mark.parametrize("world", [2, 3, 4, 8])
def test_minimax_h3_auto_never_emits_cfg_or_dp(world):
    """The DiT caps batch at 1 and the AV latent is a packed NestedTensor, so
    cfg-parallel and data-parallel both typed-refuse for this family. Auto
    must resolve batch 1 on every world without folding spare ranks into dp,
    and there is no cfg row to select.
    """
    decision = choose_auto_topology(
        "minimax_h3", "bf16", 1.0, world, cfg_value=4.0, batch_size=1)
    assert decision.topology.cfg == 1
    assert decision.topology.dp == 1
    assert decision.topology.sequence_parallel == world
    decision.topology.validate()


def test_minimax_h3_auto_fold_grows_ring_not_ulysses():
    """H3 has no exact family-and-degree Ulysses fold grant. World 4 batch 1
    therefore folds to uly2+ring2, and a divisibility-padded packed sequence
    refuses there rather than rendering.
    """
    decision = choose_auto_topology(
        "minimax_h3", "bf16", 1.0, 4, cfg_value=4.0, batch_size=1)
    assert (decision.topology.ulysses, decision.topology.ring) == (2, 2)
    assert "folded into ring2" in decision.reason


def test_minimax_h3_auto_reason_carries_the_refusal_provenance():
    decision = choose_auto_topology("minimax_h3", "bf16", 1.0, 2, cfg_value=4.0)
    assert "packed video+audio single sequence" in decision.reason
    assert "typed-refuse on the driver" in decision.reason


def test_rules_are_well_formed():
    rows = [r.row for r in AUTO_TABLE]
    assert len(rows) == len(set(rows)), "duplicate table row ids"
    for rule in AUTO_TABLE:
        assert rule.mp_min < rule.mp_max
        assert rule.note


def test_an_unknown_family_note_names_the_miss_and_the_fix():
    """"unknown" means no signature matched, which is a renamed or absent key
    path. It is never a prefix the table failed to recognize: needles match at
    dotted boundaries wherever they fall, so an export under an unlisted
    wrapper still detects. The note must not tell the operator to look for a
    prefix."""
    decision = choose_auto_topology("unknown", "bf16", 1.0, 2)
    assert "no signature matched" in decision.reason
    assert "family_adapter" in decision.reason
    assert "prefix" not in decision.reason
    assert decision.topology.ulysses == 2


def test_a_family_without_a_table_row_says_so():
    """mage_flow and qwen_image21 have no auto table rows and reach this
    branch. A family without a row must not borrow the unknown text,
    because a registered family still arms its preflights. The set is pinned
    so a new row-less family is a reviewed edit."""
    from dgx_monarch.adapters import SELECTABLE_FAMILIES

    tabled = {rule.family for rule in AUTO_TABLE}
    rowless = set(SELECTABLE_FAMILIES) - tabled
    assert rowless == {"mage_flow", "qwen_image21"}
    for family in (*sorted(rowless), "not_a_registered_family"):
        assert family not in tabled
        decision = choose_auto_topology(family, "bf16", 1.0, 2)
        assert f"family {family!r} has no auto table row" in decision.reason
        assert "no signature matched" not in decision.reason
        assert decision.topology.ulysses == 2


def test_world_one_takes_no_note():
    decision = choose_auto_topology("unknown", "bf16", 1.0, 1)
    assert decision.reason == "auto: single; world size 1"


# Auto never resolves a cfg-plus-sequence composite.

_WORLD_TWO_AUTO_ROWS = "a1a0738c21f59124e6004a1f7eba20fe46526b892b1e9863b9e2db750a52f2c0"


def _auto_rows(world: int) -> list[str]:
    """Each sampled auto decision at this world, as one sortable line each."""
    families = sorted({rule.family for rule in AUTO_TABLE} | {"unknown"})
    rows = []
    for family in families:
        for quant in ("bf16", "fp8", "int8"):
            for megapixels in (0.26, 0.79, 1.05, 1.19, 1.5, 2.4, 8.0):
                for cfg in (1.0, 3.5):
                    for batch in (1, 2):
                        decision = choose_auto_topology(
                            family, quant, megapixels, world,
                            cfg_value=cfg, batch_size=batch)
                        rows.append(
                            f"{family}|{quant}|{megapixels}|{cfg}|{batch}="
                            f"{decision.topology.describe()}|row{decision.rule.row}"
                            f"|sage{int(decision.sage)}")
    return rows


def test_two_spark_pair_auto_rows_match_the_scoped_cfg_policy():
    """The digest pins the world-2 auto decision for every family, quant,
    megapixels, cfg and batch value in the _auto_rows grid, so a policy change
    is a reviewed digest edit."""
    digest = hashlib.sha256("\n".join(_auto_rows(2)).encode()).hexdigest()
    assert digest == "d0f532280e70f030e7bc63eb5eb7fbde31fc765f221633ee223b3454b0fcee83"


def test_auto_never_resolves_cfg_plus_sequence():
    """No world, batch or family reaches a composite the table never names."""
    for world in (2, 3, 4, 6, 8, 16):
        for family in sorted({rule.family for rule in AUTO_TABLE}):
            for batch in (1, 2, 3, 4):
                try:
                    topology = choose_auto_topology(
                        family, "fp8", 1.0, world, cfg_value=3.5, batch_size=batch
                    ).topology
                except UnsupportedModelError:
                    continue
                assert topology.cfg == 1 or topology.sequence_parallel == 1, (
                    f"{family} world {world} batch {batch}: {topology.describe()}")


@pytest.mark.parametrize("world", [4, 8])
def test_leftover_dp_on_a_cfg_row_refuses_instead_of_folding(world: int):
    # Folding the leftover dp here would build ringN+cfg2, a composite no
    # AUTO_TABLE row names; both worlds resolved it before 2026-08-20.
    with pytest.raises(UnsupportedModelError) as excinfo:
        choose_auto_topology("chroma", "bf16", 1.0, world, cfg_value=3.5, batch_size=1)
    message = str(excinfo.value)
    assert "[dgxm:K]" in message
    assert "There is no waiver for this refusal" in message
    assert f"multiple of {world // 2}" in message


def test_the_auto_refusal_names_presets_that_tile_the_world():
    with pytest.raises(UnsupportedModelError) as excinfo:
        choose_auto_topology("chroma", "bf16", 1.0, 4, cfg_value=3.5, batch_size=1)
    message = str(excinfo.value)
    for preset in ("uly4", "ring4", "uly2+cfg2"):
        assert preset in message
        assert preset in PRESETS
        assert topology_from_preset(preset, 4).dp == 1
    assert "not what auto would grant" in message


def test_the_escape_list_keeps_the_cfg_composite_and_says_who_owns_it():
    """uly2+cfg2 stays in the escape list although auto refuses that shape,
    and the message says why it can.

    Every preset that tiles world 4 is unmeasured at world 4 on a two-rank
    rig, so dropping only the composite would imply the other two carry
    evidence they do not have (docs/TROUBLESHOOTING.md #21 offers uly4 as an
    operator choice the same way). uly2+cfg2 is also the only one of the three
    that keeps ring at 1, where a padded Krea2 sequence gets the exact ulysses
    exclusion instead of the ring refusal in
    adapters/base.assert_ulysses_only_padding. docs/DESIGN.md section 5.9 puts
    a typed preset on the operator; auto still refuses to derive it.
    """
    with pytest.raises(UnsupportedModelError) as excinfo:
        choose_auto_topology("chroma", "bf16", 1.0, 4, cfg_value=3.5, batch_size=1)
    message = str(excinfo.value)
    assert "uly2+cfg2" in message
    assert "the operator's choice" in message
    assert "declined to derive" in message
    assert topology_from_preset("uly2+cfg2", 4).ring == 1
    assert topology_from_preset("ring4", 4).ring == 4


def test_a_divisible_batch_still_keeps_dp_on_a_cfg_row():
    decision = choose_auto_topology("chroma", "bf16", 1.0, 4, cfg_value=3.5, batch_size=2)
    assert decision.topology.describe() == "cfg2+dp2"


def test_a_non_cfg_row_still_folds_leftover_dp_into_ring():
    decision = choose_auto_topology("krea2", "fp8", 2.4, 4, cfg_value=3.5, batch_size=1)
    assert decision.topology.cfg == 1
    assert decision.topology.dp == 1
    assert decision.topology.ring == 2
