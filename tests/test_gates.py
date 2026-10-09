"""Unit coverage for the benchmark fidelity gates (pure CPU math)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))

from gates import (  # the benchmark dir joins sys.path above
    AS_RENDERED,
    DEFAULT_STEP_NRMS,
    UnscorableLatentBatch,
    gate_fidelity,
    gate_step_fidelity,
    scored_latent,
)


def test_fidelity_passes_identical():
    t = torch.randn(1, 4, 8, 8)
    ok, detail = gate_fidelity(t, t.clone())
    assert ok, detail


def test_fidelity_fails_gross_divergence():
    t = torch.randn(1, 4, 8, 8)
    ok, _ = gate_fidelity(t, t + 1.0)
    assert not ok


def test_fidelity_fails_shape_mismatch():
    ok, detail = gate_fidelity(torch.zeros(1, 4, 8, 8), torch.zeros(1, 4, 8, 16))
    assert not ok and "shape" in detail


def test_step_fidelity_within_floor():
    torch.manual_seed(0)
    ref = torch.randn(1, 4, 16, 16)
    wobble = ref + 0.05 * DEFAULT_STEP_NRMS * torch.randn_like(ref)
    ok, detail = gate_step_fidelity(wobble, ref)
    assert ok, detail


def test_step_fidelity_catches_adapter_bug_scale():
    torch.manual_seed(0)
    ref = torch.randn(1, 4, 16, 16)
    # A wrong rope shard or a missing gather lands far above the floor; an
    # uncorrelated tensor stands in for one.
    ok, _ = gate_step_fidelity(torch.randn_like(ref), ref)
    assert not ok


def test_scored_latent_passes_a_leg_that_returned_its_own_batch():
    samples = torch.randn(1, 4, 8, 8)
    scored, form = scored_latent(samples, 1, 3.5)
    assert scored is samples and form == AS_RENDERED


def test_scored_latent_passes_a_dp_reference_batch():
    # A dp2 reference cell renders batch 2 and the caller slices row 0. Two
    # rows are its own batch here, not a cond/uncond pair.
    samples = torch.randn(2, 4, 8, 8)
    scored, form = scored_latent(samples, 2, 3.5)
    assert scored is samples and form == AS_RENDERED


def test_scored_latent_combines_a_cfg_pair():
    cond, uncond = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
    scored, form = scored_latent(torch.cat([cond, uncond]), 1, 3.5)
    assert scored.shape == cond.shape
    assert torch.allclose(scored, uncond + (cond - uncond) * 3.5)
    assert form == "cond/uncond pair combined at cfg 3.5"


def test_scored_latent_combines_a_cfg_pair_over_a_batch():
    cond, uncond = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)
    scored, _ = scored_latent(torch.cat([cond, uncond]), 2, 3.5)
    assert scored.shape == cond.shape
    assert torch.allclose(scored, uncond + (cond - uncond) * 3.5)


def test_scored_latent_at_cfg_one_keeps_the_cond_leg():
    cond, uncond = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
    scored, _ = scored_latent(torch.cat([cond, uncond]), 1, 1.0)
    assert torch.allclose(scored, cond)


def test_scored_latent_refuses_a_batch_it_cannot_read():
    with pytest.raises(UnscorableLatentBatch) as caught:
        scored_latent(torch.randn(3, 4, 8, 8), 1, 3.5)
    assert "3" in str(caught.value) and "2" in str(caught.value)


def test_scored_latent_refuses_a_pair_with_no_finite_cfg():
    with pytest.raises(UnscorableLatentBatch):
        scored_latent(torch.randn(2, 4, 8, 8), 1, float("nan"))


def test_scored_latent_refuses_a_latent_with_no_batch_axis():
    with pytest.raises(UnscorableLatentBatch):
        scored_latent(torch.tensor(1.0), 1, 3.5)


def test_a_combined_pair_scores_where_its_cond_rows_alone_would_not():
    # The reference holds the cfg combination. The pair's cond rows alone, which
    # a plain batch slice takes, fail the floor although the pair matches.
    torch.manual_seed(0)
    cond = torch.randn(1, 4, 16, 16)
    uncond = cond + 0.2 * torch.randn_like(cond)
    reference = uncond + (cond - uncond) * 3.5
    pair = torch.cat([cond + 1e-3 * torch.randn_like(cond), uncond])
    scored, _ = scored_latent(pair, 1, 3.5)
    assert gate_step_fidelity(scored, reference)[0]
    assert not gate_step_fidelity(pair[:1], reference)[0]
