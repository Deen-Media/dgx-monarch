"""The collect half of a render: cross-rank identity, then leader latents.

Every rank returns a result row. This module proves they agree, validates any
resident-adoption evidence they carry, reassembles the DP leaders' batch slices
in dp_rank order and hands back one stock-shaped latent dict.

A leaf by construction: nothing here reaches back into ``nodes/common``, so the
completion contract can be exercised on its own.
"""
from __future__ import annotations

from typing import Any

from ..adoption_evidence import (
    RESULT_KEY as _ADOPTION_EVIDENCE_RESULT_KEY,
)
from ..adoption_evidence import (
    validate_result_evidence as _validate_adoption_evidence,
)
from ..log import get_logger
from ..sampling_contract import latent_without_size_tags
from ..topology import Topology
from ..transfer import read_latent_result
from . import consent_waiver, render_preflight
from .latent_outputs import (
    collect_denoised_output,
    materialize_leader_samples,
)
from .render_validation import verify_cross_rank_signatures

log = get_logger(__name__)

_latent_without_topology_metadata = render_preflight.latent_without_topology_metadata


def _finish_render(
    results: list,
    topo: Topology,
    latent: dict,
    read_guard: Any = None,
    *,
    expected_adoption_context_sha256: str | None = None,
    expected_adoption_request_sha256: str | None = None,
    expected_adoption_render_id_sha256: str | None = None,
    expected_adoption_setup_generation: int | None = None,
) -> dict:
    """Cross-rank identity gate plus leader latent materialization."""
    verify_cross_rank_signatures(results, topo)
    adoption_evidence = _validate_adoption_evidence(
        results,
        expected_adoption_context_sha256,
        expected_request_sha256=expected_adoption_request_sha256,
        expected_render_id_sha256=expected_adoption_render_id_sha256,
        expected_setup_generation=expected_adoption_setup_generation,
    )

    # DP leaders each carry a batch slice; sort by dp_rank so the reconstructed
    # batch matches split_batch_for_dp's chunk order, never mesh-iteration order.
    leaders = sorted(
        (r for r in results if r.get("latent") is not None),
        key=lambda r: r.get("dp_rank", 0),
    )
    if not leaders:
        raise RuntimeError(f"no rank returned a latent (results: {results})")
    leader_dp_ranks = [int(r.get("dp_rank", 0)) for r in leaders]
    expected_dp_ranks = list(range(int(topo.dp)))
    if leader_dp_ranks != expected_dp_ranks:
        raise RuntimeError(
            "incomplete or duplicate DP leader results: expected dp ranks "
            f"{expected_dp_ranks}, got {leader_dp_ranks}"
        )
    for r in results:
        phases = (f" gpu_load {r['gpu_load_s']:.1f}s" if r.get("gpu_load_s", 0) > 1 else "")
        log.info("rank %s/%s: sample %.2fs (%s%s) stats=%s", r.get("rank"), r.get("host"),
                 r.get("sample_s", -1), r.get("transition"), phases, r.get("latent_stats"))

    def materialize(descriptor):
        if hasattr(read_guard, "begin_read"):
            return read_latent_result(descriptor, read_guard)
        return read_latent_result(descriptor)

    samples, tensors = materialize_leader_samples(
        leaders, leader_dp_ranks, int(topo.dp), materialize
    )

    # Stock samplers keep noise_mask in the output latent so a chained masked
    # sampler keeps inpainting; preserve it for stock-semantics parity. They
    # drop the size tags their fix consumed, or a chained sampler handed still
    # empty output (an empty step window) would rescale it a second time.
    out = latent_without_size_tags(_latent_without_topology_metadata(latent))
    out["samples"] = samples
    extras = [r.get("latent_extra") or {} for r in leaders]
    denoised = collect_denoised_output(extras, samples, tensors, leader_dp_ranks)
    if denoised is not None:
        out["_dgxm_denoised"] = denoised
    if adoption_evidence:
        out[_ADOPTION_EVIDENCE_RESULT_KEY] = adoption_evidence
    return consent_waiver.stamp_result(out, results, latent)
