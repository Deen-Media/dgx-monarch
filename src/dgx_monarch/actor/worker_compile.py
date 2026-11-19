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


def _block_lists(diffusion_model) -> str:
    """Name the model's own block lists for the eager warning; empty if unreadable.

    A child counts when it is a ModuleList, or a ModuleDict of ModuleLists
    (FLUX.3's content_mode_blocks), whose name ends in blocks, layers or
    refiner. It only words a log line and decides nothing.
    """
    try:
        from torch import nn

        names = [
            name for name, child in diffusion_model.named_children()
            if name.endswith(("blocks", "layers", "refiner")) and (
                isinstance(child, nn.ModuleList) or (
                    isinstance(child, nn.ModuleDict) and len(child) > 0
                    and all(isinstance(value, nn.ModuleList) for value in child.values())))
        ]
    except Exception:
        return ""
    return ", ".join(names)


def maybe_compile_dit(diffusion_model) -> None:
    """Compile DiT blocks with the GB10-safe max-autotune configuration."""
    if os.environ.get("DGXM_COMPILE_DIT") != "1":
        return
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None:
        # Only a ``blocks`` list compiles; compile_policy.py lists the families
        # that have one. The others stay eager: compiling their lists would put
        # unmeasured code under the compiler, such as the dual-stream adapters'
        # per-call text projection swap inside each block under pure Ulysses.
        lists = _block_lists(diffusion_model)
        safe_call(
            log.warning,
            "compile_dit: %s has no blocks list%s, so nothing was compiled; running eager "
            "(docs/TROUBLESHOOTING.md #14)",
            type(diffusion_model).__name__,
            f" (its blocks are in {lists})" if lists else "",
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

    # This guard covers wrapping only. block.compile() is lazy: dynamo traces
    # each block on its first forward, inside the render, where a trace failure
    # raises instead of falling back to eager.
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
