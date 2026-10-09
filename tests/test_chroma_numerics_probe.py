"""The dry leg of the chroma numerics probe (pure CPU math, no comfy).

The probe's real leg needs a checkpoint and a GPU and has never run. What runs
here is every line of arithmetic it would apply to those tensors, driven by
synthetic ones: the nvfp4 rounding, the shard terms it reads off a layer, the
cfg combine gain, and the note it writes.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dgxm_chroma_numerics_probe", REPO / "benchmark" / "chroma_numerics_probe.py"
)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
# A frozen dataclass resolves its own module out of sys.modules, so register
# the module before it executes.
sys.modules[SPEC.name] = probe
_argv_before = list(sys.argv)
SPEC.loader.exec_module(probe)
assert sys.argv == _argv_before


def _outlier_activation(rows: int = 32, width: int = 32) -> torch.Tensor:
    """Row 0 carries the outlier, so one token shard holds it and the other
    does not: the case that made the two ranks disagree on the amax."""
    generator = torch.Generator().manual_seed(7)
    activation = torch.randn(rows, width, generator=generator)
    activation[0] *= 50.0
    return activation


def test_the_quantizer_keeps_values_its_own_levels_can_hold():
    step = 0.5
    levels = torch.tensor(probe.E2M1_LEVELS)
    exact = (levels.repeat(2).reshape(1, -1) * step)
    exact = torch.cat([exact, -exact])
    scale = probe.nvfp4_tensor_scale(exact)
    assert torch.allclose(probe.quantize_nvfp4(exact, scale), exact)


def test_the_quantizer_returns_zero_for_a_tensor_with_no_scale():
    zeros = torch.zeros(4, 16)
    assert torch.equal(probe.quantize_nvfp4(zeros, 0.0), zeros)


def test_the_quantizer_stays_inside_one_bin_of_its_input():
    torch.manual_seed(0)
    activation = torch.randn(8, 64)
    rounded = probe.quantize_nvfp4(activation, probe.nvfp4_tensor_scale(activation))
    # E2M1's widest gap is 2 in 6 of the block amax; half of that bounds the
    # error, with room for the e4m3 block scale's own 6 percent step.
    assert float((rounded - activation).abs().max()) <= 0.25 * float(activation.abs().max())


def test_a_shard_that_misses_the_outlier_takes_a_smaller_scale():
    activation = _outlier_activation()
    shards = probe.token_shards(activation, 2)
    scales = [probe.nvfp4_tensor_scale(shard) for shard in shards]
    assert scales[0] > scales[1] * 5
    assert max(scales) == pytest.approx(probe.nvfp4_tensor_scale(activation))


def test_token_shards_refuse_an_indivisible_row_count():
    with pytest.raises(ValueError):
        probe.token_shards(torch.zeros(3, 8), 2)


def test_a_folded_pair_shards_on_its_tokens_and_not_on_its_batch():
    # A Linear inside a folded cond/uncond call sees batch 2. Splitting that
    # leading axis would emulate cfg-parallel, where ulysses gives every rank
    # both legs and half the tokens.
    activation = torch.arange(2 * 8 * 4, dtype=torch.float32).reshape(2, 8, 4)
    shards = probe.token_shards(activation, 2)
    assert [tuple(shard.shape) for shard in shards] == [(2, 4, 4), (2, 4, 4)]
    assert torch.equal(shards[0], activation[:, :4])
    assert torch.equal(shards[1], activation[:, 4:])
    terms = probe.layer_terms("block.0", activation,
                              generator=torch.Generator().manual_seed(1))
    assert terms.tokens == 8


def test_the_shared_scale_makes_the_split_call_bit_identical():
    # The claim the shared-scale hook is built on: amax over a partition is the
    # max of the parts, so each rank's rows come out as the whole call's rows.
    terms = probe.layer_terms("block.0", _outlier_activation(),
                              generator=torch.Generator().manual_seed(1))
    assert terms.nrms_shared_scale == 0.0
    assert terms.shard_amax_ratio > 5.0


def test_the_shared_scale_column_reads_zero_for_every_input():
    # The chroma nvfp4 gap record in docs/VALIDATION.md (2026-09-09, criterion A1)
    # says why this column reads 0 for every input and what it cannot see. Shapes
    # whose row count is not a multiple of the 16-element block are here on purpose.
    generator = torch.Generator().manual_seed(11)
    for index, shape in enumerate([(2, 64, 64), (1, 34, 48), (2, 18, 16), (6, 32)]):
        activation = torch.randn(shape, generator=generator)
        activation[..., 0, :] *= 60.0  # an outlier the two shards disagree on
        terms = probe.layer_terms(f"block.{index}", activation, generator=generator)
        assert terms.nrms_shared_scale == 0.0, shape
        assert terms.shard_amax_ratio > 1.0, shape


def test_a_layer_whose_shape_does_not_shard_is_skipped_and_named():
    # One odd tap must not cost the whole leg: this probe is taken once at a
    # boundary, after a checkpoint load and a render.
    layers = [
        ("block.0", torch.randn(1, 8, 16)),
        ("block.1", torch.randn(1, 63, 16)),  # an odd row count does not halve
        ("block.2", torch.randn(2, 4, 16)),
    ]
    terms, skipped = probe.price_layers(
        layers, generator=torch.Generator().manual_seed(2))
    assert [term.path for term in terms] == ["block.0", "block.2"]
    assert [path for path, _ in skipped] == ["block.1"]
    assert "63" in skipped[0][1]


def test_the_note_names_the_layers_it_skipped():
    terms, skipped = probe.price_layers(
        [("block.1", torch.randn(1, 63, 16))],
        generator=torch.Generator().manual_seed(3))
    note = probe.report(terms, list(probe.CFG_LEGS), 3.5, skipped=skipped)
    assert not terms
    assert "* skipped block.1" in note


def test_the_per_rank_scale_moves_the_split_call():
    terms = probe.layer_terms("block.0", _outlier_activation(),
                              generator=torch.Generator().manual_seed(1))
    assert terms.nrms_shard_scale > 0.0


def test_an_untouched_layer_reads_no_terms_and_gain_one():
    # An identity quantizer passes the bf16 move straight through, so the
    # amplification estimator must read exactly 1 and both shard terms 0.
    terms = probe.layer_terms("block.0", _outlier_activation(),
                              quantize=lambda tensor, _scale: tensor,
                              generator=torch.Generator().manual_seed(1))
    assert terms.nrms_shard_scale == 0.0 and terms.nrms_shared_scale == 0.0
    assert terms.amplification == pytest.approx(1.0)


def test_the_layer_verdicts_name_every_criterion():
    terms = [probe.layer_terms(f"block.{index}", _outlier_activation(),
                               generator=torch.Generator().manual_seed(index))
             for index in range(3)]
    lines = probe.layer_verdicts(terms)
    assert [line[:2] for line in lines[:3]] == ["A1", "A2", "A3"]
    assert probe.layer_verdicts([])[0].startswith("no nvfp4 layer")


def test_the_third_criterion_reads_against_the_family_it_was_given():
    # The control leg runs the same probe on krea2, whose own ratio is much
    # smaller, so the bar A3 clears has to travel with the family.
    terms = [probe.layer_terms("block.0", _outlier_activation(),
                               generator=torch.Generator().manual_seed(3))]
    strict = probe.layer_verdicts(terms, 1000.0)[-1]
    loose = probe.layer_verdicts(terms, 0.5)[-1]
    assert strict.startswith("A3 reads under")
    assert loose.startswith("A3 reads at or above")
    assert "krea2" in loose


def test_the_combine_gain_is_one_folded_and_quadrature_split():
    assert probe.combine_gain(3.5, folded=True) == 1.0
    assert probe.combine_gain(3.5, folded=False) == pytest.approx(4.301, abs=1e-3)


def _legs(**readings: float) -> list[probe.CfgLeg]:
    return probe.legs_with_readings(list(readings.items()))


def test_the_cfg_verdict_names_the_split_call():
    verdict = probe.cfg_verdict(_legs(**{
        "gate-bf16-cfg35-unequal": 0.150,
        "gate-bf16-cfg35-equal": 0.031,
        "gate-bf16-cfg35-short-equal": 0.030,
    }))
    assert verdict == "the split model call carries the term"


def test_the_cfg_verdict_names_the_short_stream():
    verdict = probe.cfg_verdict(_legs(**{
        "gate-bf16-cfg35-unequal": 0.150,
        "gate-bf16-cfg35-equal": 0.031,
        "gate-bf16-cfg35-short-equal": 0.148,
    }))
    assert verdict.startswith("the short stream carries the term")


def test_the_cfg_verdict_sends_a_low_reading_back_to_the_probe():
    verdict = probe.cfg_verdict(_legs(**{
        "gate-bf16-cfg35-unequal": 0.030,
        "gate-bf16-cfg35-equal": 0.031,
        "gate-bf16-cfg35-short-equal": 0.029,
    }))
    assert verdict == "every leg reads low, so the term is in the sweep's own probe"


def test_the_cfg_verdict_says_nothing_without_readings():
    assert "says nothing" in probe.cfg_verdict(list(probe.CFG_LEGS))
    partial = probe.cfg_verdict(_legs(**{"gate-bf16-cfg35-unequal": 0.150}))
    assert "no reading yet" in partial


def test_the_split_rows_divide_each_reading_by_its_own_gain():
    rows = probe.cfg_split_rows(_legs(**{"gate-bf16-cfg35-unequal": 0.150}), 3.5)
    unequal = next(row for row in rows if row["leg"] == "gate-bf16-cfg35-unequal")
    assert unequal["per_leg"] == pytest.approx(0.0349, abs=1e-4)
    assert unequal["calls"] == "two"


def test_an_unknown_leg_name_is_refused():
    with pytest.raises(SystemExit):
        probe.legs_with_readings([("gate-nothing", 0.1)])


def test_the_dry_leg_writes_both_tables_and_says_they_are_synthetic(tmp_path):
    note, rows = tmp_path / "note.md", tmp_path / "rows.json"
    code = probe.main([
        "--dry-run", "--layers", "4", "--out", str(note), "--json", str(rows),
        "--nrms", "gate-bf16-cfg35-unequal=0.150",
    ])
    assert code == 0
    text = note.read_text(encoding="utf-8")
    assert "SYNTHETIC INPUT" in text
    assert "## Probe A" in text and "## Probe B" in text
    assert text.count("| double_blocks.") == 4
    assert '"amplification"' in rows.read_text(encoding="utf-8")


def test_the_probe_refuses_both_modes_and_neither(capsys):
    assert probe.main([]) == 2
    assert probe.main(["--dry-run", "--unet", "x.safetensors"]) == 2
