"""Opt-in DiT block compilation for worker model loads."""
from __future__ import annotations

import os
import types

from ..log import get_logger
from ..transfer_utils import failure_summary, safe_call

log = get_logger(__name__)


def compile_dit_allowed(base_patcher) -> bool:
    """A slab-backed fresh load must never reach the block compiler."""
    return not bool(getattr(base_patcher, "_dgxm_slab_resident", False))


def maybe_compile_dit(diffusion_model) -> None:
    """Compile DiT blocks with the GB10-safe max-autotune configuration."""
    if os.environ.get("DGXM_COMPILE_DIT") != "1":
        return
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None:
        # Only a ``blocks`` list compiles. Krea2 uses it; families with other
        # block-list names remain eager.
        safe_call(
            log.warning,
            "compile_dit: %s has no blocks list, so nothing was compiled; running eager "
            "(docs/TROUBLESHOOTING.md #14)",
            type(diffusion_model).__name__,
        )
        return
    os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
    try:
        import torch._inductor.config as inductor_config

        # CUTLASS fp8 kernels do not dispatch on sm_121.
        inductor_config.max_autotune_gemm_backends = "ATEN,TRITON"
    except Exception as exc:
        safe_call(
            log.warning,
            "inductor config unavailable: %s",
            failure_summary(exc),
        )

    import torch.nn.functional as functional

    def _swiglu_out_of_place(self, value):
        return self.down(functional.silu(self.gate(value)) * self.up(value))

    try:
        compiled = 0
        for block in blocks:
            mlp = getattr(block, "mlp", None)
            if mlp is not None and all(
                hasattr(mlp, attribute) for attribute in ("gate", "up", "down")
            ):
                mlp.forward = types.MethodType(_swiglu_out_of_place, mlp)
            if getattr(block, "_compiled_call_impl", None) is not None:
                continue
            block.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
            compiled += 1
        if compiled:
            log.info(
                "DiT: %d blocks compiled (max-autotune-no-cudagraphs)",
                compiled,
            )
    except Exception as exc:
        safe_call(
            log.warning,
            "DiT block compile skipped (%s); running eager",
            failure_summary(exc),
        )
