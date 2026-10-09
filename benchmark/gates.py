"""Quality gates for benchmark runs.

Gate 1 (cross-rank identity) lives in production, not here:
nodes/render_validation.verify_cross_rank_signatures enforces it on every
multi-rank render, in nodes/render_result._finish_render, which every
submit_render path calls (run_render in nodes/common.py and the pipelined path
in nodes/pipeline.py alike). It has two tiers: exact equality, or under 1e-3
relative drift. The second tier covers comfy's dynamic offload, which reorders
reductions per box, so cross-rank stats agree to about 3e-5 relative, not
bitwise (docs/VALIDATION.md, Determinism tiers, 2026-07-03).

Gate 2 (fidelity, this module): the candidate latent must match the latent of
the reference cell its case names. Latents are compared directly, so the gate
needs no VAE: max|diff| and MSE on the same math path, normalized rms for one
step across topologies.
"""
from __future__ import annotations

import math

import torch

# Same-math noise floor: re-running an identical topology and kernel path must
# reproduce latents to allocator-level jitter. Exceeding this on a same-path
# comparison means a real math change.
DEFAULT_MAX_ABS = 5e-2
DEFAULT_MSE = 1e-4

# Cross-topology one-step error threshold: rms(diff) / std(reference).
# Pure Ulysses matches one GPU exactly for the named families and shapes in
# docs/VALIDATION.md. Other configurations can differ; LTX 2.5 video measured
# NRMS 0.044-0.094, and ring2 controls measured 0.012-0.065.
# Multi-step trajectories amplify small differences, so this threshold applies
# to one denoising step, not final-image quality. The 0.10 threshold came from
# Spark comparisons near 0.09. It is not sufficient to detect every adapter
# fault: the incorrect Chroma token order measured 0.041-0.052, while the
# incorrect PixelDiT layout measured 0.265-0.338.
DEFAULT_STEP_NRMS = 0.10


def gate_fidelity(candidate: torch.Tensor, reference: torch.Tensor,
                  max_abs: float = DEFAULT_MAX_ABS, mse: float = DEFAULT_MSE) -> tuple[bool, str]:
    """Same-math-path comparison (identical topology + kernels)."""
    if candidate.shape != reference.shape:
        return False, f"shape {list(candidate.shape)} != reference {list(reference.shape)}"
    diff = (candidate.float() - reference.float())
    got_max = float(diff.abs().max())
    got_mse = float((diff ** 2).mean())
    ok = got_max <= max_abs and got_mse <= mse
    return ok, f"max|diff|={got_max:.2e} (<= {max_abs:.0e}), mse={got_mse:.2e} (<= {mse:.0e})"


def gate_step_fidelity(candidate: torch.Tensor, reference: torch.Tensor,
                       nrms: float = DEFAULT_STEP_NRMS) -> tuple[bool, str]:
    """Cross-topology comparison of ONE denoise step (see DEFAULT_STEP_NRMS)."""
    if candidate.shape != reference.shape:
        return False, f"shape {list(candidate.shape)} != reference {list(reference.shape)}"
    diff = (candidate.float() - reference.float())
    got = float(diff.pow(2).mean().sqrt() / reference.float().std().clamp_min(1e-8))
    return got <= nrms, f"1-step nrms={got:.3f} (<= {nrms})"


class UnscorableLatentBatch(ValueError):
    """A returned latent whose batch is neither the leg's own nor a cfg pair."""


AS_RENDERED = "as rendered"


def scored_latent(candidate, latent_batch: int, cfg: float) -> tuple[torch.Tensor, str]:
    """Reduce a returned cond/uncond pair to the form a reference cell holds.

    ComfyUI folds cond and uncond into one model call whenever it runs both
    passes and ``can_concat_cond`` admits the pair. A family that publishes
    ``c_crossattn`` as ``CONDRegular`` (Flux, Chroma, Krea2 and most DiT
    families in comfy/model_base.py) folds only equal shapes; one that
    publishes ``CONDCrossAttn`` also folds unequal lengths at an LCM repeat of
    4 or less (comfy/conds.py). ComfyUI combines the two predictions as
    ``uncond + (cond - uncond) * cfg`` (``cfg_function`` in comfy/samplers.py,
    with ``sampling_function`` ordering the pair cond first, uncond second).
    A leg that hands this harness the pair unreduced would otherwise score its
    cond rows against a combined reference, which is a wrong number even where
    the shapes let it through. At cfg 1.0 the combination is the cond leg
    exactly, the rows a cfg++ leg at cfg 1.0 would score unreduced, so one rule
    covers both.

    A latent whose batch is already the batch the case asked for passes through
    untouched. That is every dp reference cell: it renders batch dp and the
    caller slices row 0. Any other batch is refused by name, because guessing
    which rows to keep would file a number no one can read back.
    """
    if int(latent_batch) < 1:
        raise UnscorableLatentBatch(
            f"latent batch {latent_batch} must be at least 1")
    if getattr(candidate, "ndim", 0) < 1:
        raise UnscorableLatentBatch("sample latent has no batch axis")
    rows = int(candidate.shape[0])
    if rows == int(latent_batch):
        return candidate, AS_RENDERED
    if rows != 2 * int(latent_batch):
        raise UnscorableLatentBatch(
            f"sample latent batch {rows} is neither the case's own "
            f"{latent_batch} nor the cond/uncond pair {2 * int(latent_batch)}")
    if not math.isfinite(float(cfg)):
        raise UnscorableLatentBatch(
            "a cond/uncond pair needs a finite cfg to combine")
    cond, uncond = candidate[:latent_batch], candidate[latent_batch:]
    combined = uncond + (cond - uncond) * float(cfg)
    return combined, f"cond/uncond pair combined at cfg {float(cfg):g}"
