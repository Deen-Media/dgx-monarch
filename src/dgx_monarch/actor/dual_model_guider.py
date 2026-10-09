"""Keep both CFG checkpoints resident while ComfyUI prepares each model.

Stock ``Guider_DualModel`` prepares the unconditional and conditional models
in separate calls. With ``DISABLE_SMART_MEMORY``, each call unloads models
outside its load list. This can leave the unconditional forward with CPU
weights or force an expensive reload of the conditional model. On unified
memory, offloading the checkpoint frees no physical memory.

Include the other checkpoint as an additional model in each prepare call.
Stage these registrations one at a time: ``ModelPatcher.clone`` recursively
clones additional models without a cycle guard, so reciprocal registrations
would recurse indefinitely. Register cond on uncond first. A keyed
PREPARE_SAMPLING wrapper then removes that registration after loading uncond
and registers uncond on cond for the next load. At cfg 1.0, stock skips the
uncond prepare; register only uncond on cond and install no wrapper. The
outer ``finally`` removes both keys with ComfyUI's pop-if-present cleanup.

The wrapper has no per-guider state. It reads the staged registration from
the patcher passed to it and becomes inert after cleanup. ComfyUI cannot
remove model-options wrappers, but ``sample_protocol`` supplies fresh render
clones, so their options die with the render. Reusing a patcher reuses the
same stateless wrapper.

The wrapper returns the original load result and composes with the partial-
load guard in either order. Additional-model cleanup may run more than once;
ComfyUI's patcher cleanup is idempotent and its finally blocks still run.
"""
from __future__ import annotations

_DUAL_KEY = "dgxm_dual_uncond"
_DUAL_COND_KEY = "dgxm_dual_cond"
_HANDOFF_KEY = "dgx_monarch_dual_model_handoff"


def handoff(executor, *args, **kwargs):
    """Move the staged registration from the uncond to the cond, after the load.

    Comfy passes the patcher being prepared as the first argument, the same
    shape the partial-load guard reads on this seam. Anything else has nothing
    staged and passes through.
    """
    result = executor(*args, **kwargs)
    uncond = args[0] if args else kwargs.get("model")
    staged = getattr(uncond, "get_additional_models_with_key", None)
    pending = list(staged(_DUAL_COND_KEY)) if staged is not None else []
    if pending:
        uncond.remove_additional_models(_DUAL_COND_KEY)
        pending[0].set_additional_models(_DUAL_KEY, [uncond])
    return result


def make_both_resident_guider(model_patcher, uncond_patcher):
    import math

    import comfy.patcher_extension as pe
    from comfy_extras.nodes_custom_sampler import Guider_DualModel

    class _DualModelBothResident(Guider_DualModel):
        def _install_handoff(self):
            """Register the swap that runs between the two prepares.

            Read the key before writing it: comfy appends on a second add
            under one key, and one stateless wrapper serves every render
            this patcher sees.
            """
            options = self.uncond_model_patcher.model_options
            if pe.get_wrappers_with_key(
                pe.WrappersMP.PREPARE_SAMPLING, _HANDOFF_KEY, options,
                is_model_options=True,
            ):
                return

            pe.add_wrapper_with_key(
                pe.WrappersMP.PREPARE_SAMPLING, _HANDOFF_KEY, handoff, options,
                is_model_options=True,
            )

        def outer_sample(self, *args, **kwargs):
            try:
                if math.isclose(self.cfg, 1.0):
                    self.model_patcher.set_additional_models(
                        _DUAL_KEY, [self.uncond_model_patcher])
                else:
                    self.uncond_model_patcher.set_additional_models(
                        _DUAL_COND_KEY, [self.model_patcher])
                    self._install_handoff()
                return super().outer_sample(*args, **kwargs)
            finally:
                self.uncond_model_patcher.remove_additional_models(_DUAL_COND_KEY)
                self.model_patcher.remove_additional_models(_DUAL_KEY)

    return _DualModelBothResident(model_patcher, uncond_patcher)
