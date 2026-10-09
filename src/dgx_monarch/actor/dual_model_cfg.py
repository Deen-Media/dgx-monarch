"""Dual-model asymmetric CFG split across the cfg group ("dm-cfg2").

At cfg world 2 each rank runs one of the two Ideogram4 checkpoints: cfg rank 0
the conditional (positive) model, cfg rank 1 the unconditional (negative)
model. Each denoise step both ranks run their own single forward, then one
all_gather over the cfg group exchanges the two noise predictions so both ranks
hold the same (cond, uncond) pair and combine them with the stock fp32 CFG
formula. The combined prediction is identical on both ranks, so the sampler
evolves identically and the cross-rank identity gate holds. That single forward
per rank, in place of the both-resident guider's two, lets each rank keep only
its own checkpoint resident (the residency halving).

At cfg world 1 (topology single or dp) there is no split: guidance falls back
to the both-resident staged guider (``dual_model_guider.make_both_resident_guider``),
which runs both models on the one rank.

Selection is symmetric across ranks: cfg comes from the guider spec, equal on
both ranks, and the cfg group and cfg rank come from xfuser. No rank-local
branch raises or returns between guider build and the all_gather, so one rank
can never skip the collective while its peer blocks. The only build-time
refusal, cfg close to 1.0, reads the spec cfg, so it fires on both ranks or
neither.
"""
from __future__ import annotations

import math

from ..adapters.base import UnsupportedModelError, cfg_rank, cfg_world
from ..refusal import RefusalClass, refusal

# The guider subclass, built once and held for the proc lifetime so its comfy
# base binds a single time. comfy.samplers is not importable at module import
# (the driver process has no comfy), so the class body waits for the first
# worker-side build and is never evicted after.
_SPLIT_GUIDER_CLASS: type | None = None


def _cfg_world_or_one() -> int:
    """Return CFG world size, or one when xFuser is not initialized.

    Use the sampling path's xFuser accessor. The fallback lets isolated unit
    contexts use the both-resident guider without requiring distributed setup.
    """
    try:
        return int(cfg_world())
    except Exception:
        return 1


def _split_guider_class() -> type:
    """Build (once) the per-rank split guider bound to comfy's CFGGuider."""
    global _SPLIT_GUIDER_CLASS
    if _SPLIT_GUIDER_CLASS is not None:
        return _SPLIT_GUIDER_CLASS
    import comfy.samplers

    class _CfgSplitGuider(comfy.samplers.CFGGuider):
        """One-model-per-rank CFG guider whose predict_noise exchanges the two
        predictions across the cfg group and combines them with stock cfg math.

        Stock outer_sample / inner_sample are unchanged: they prepare the one
        resident model, process this rank's single-key conditioning, and run
        process_latent_in/out. Only predict_noise is overridden, and it holds
        the render's one cross-rank point.
        """

        def __init__(self, model_patcher, local_key: str, conds, cfg: float):
            super().__init__(model_patcher)
            self._local_key = local_key
            self.set_cfg(cfg)
            # Single-key conds are stock-exercised: the both-resident dual-model
            # guider processes a {"negative": ...} dict the same way.
            self.inner_set_conds({local_key: conds})

        def predict_noise(self, x, timestep, model_options={}, seed=None):
            import comfy.samplers
            from xfuser.core.distributed import get_cfg_group

            local = comfy.samplers.calc_cond_batch(
                self.inner_model, [self.conds.get(self._local_key)],
                x, timestep, model_options)[0]
            # NCCL rejects non-contiguous inputs, and a family forward can end in
            # a crop view; make the local slice contiguous before the gather.
            gathered = get_cfg_group().all_gather(local.contiguous(), dim=0)
            n = local.shape[0]
            # all_gather concatenates in cfg-rank order: rank 0 (cond) first,
            # rank 1 (uncond) second.
            cond_pred, uncond_pred = gathered[:n], gathered[n:2 * n]
            # fp32 default combine uncond + (cond - uncond) * cfg. The model
            # argument is unused with no cfg hook (hooks cannot ride the wire),
            # so the two ranks' different inner models never enter the result.
            return comfy.samplers.cfg_function(
                self.inner_model, cond_pred, uncond_pred, self.cfg,
                x, timestep, model_options=model_options)

    _SPLIT_GUIDER_CLASS = _CfgSplitGuider
    return _CfgSplitGuider


def build_dual_model_guider(model_patcher, spec: dict, uncond_patcher=None):
    """Materialize a dual-model guider spec, per cfg world.

    world 1: the staged both-resident guider (both models on this rank).
    world 2: the split guider that runs this rank's own model and exchanges the
    prediction across the cfg group. On rank 1 the resident primary is the
    unconditional model, passed either as ``uncond_patcher`` (both-resident) or,
    under per-rank residency, as ``model_patcher`` with ``uncond_patcher`` None.
    """
    world = _cfg_world_or_one()
    negative = spec.get("negative")
    if negative is None:
        # No negative wired: the unconditional model runs image-only.
        negative = [[None, {}]]

    if world <= 1:
        if uncond_patcher is None:
            raise RuntimeError(
                "dual-model guidance needs the unconditional model resident on the workers. "
                "Connect DGXMonarchUncondUNETLoader to DGXMonarchDualModelGuider's model_negative input."
            )
        from .dual_model_guider import make_both_resident_guider

        guider = make_both_resident_guider(model_patcher, uncond_patcher)
        guider.set_conds(spec["positive"], negative)
        guider.set_cfg(float(spec["cfg"]))
        return guider

    cfg = float(spec["cfg"])
    if math.isclose(cfg, 1.0):
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "dual-model cfg-parallel (cfg2) needs a guider cfg above 1: at cfg 1.0 the "
            "unconditional rank has no guidance work and would skip the per-step exchange, "
            "hanging its peer. Raise the guider cfg above 1, or run this render at cfg 1.0 "
            "on topology uly2 instead.",
        ))

    if cfg_rank() == 0:
        patcher, key, conds = model_patcher, "positive", spec["positive"]
    else:
        # The resident primary carries the unconditional model on rank 1.
        patcher = uncond_patcher if uncond_patcher is not None else model_patcher
        key, conds = "negative", negative
    return _split_guider_class()(patcher, key, conds, cfg)
