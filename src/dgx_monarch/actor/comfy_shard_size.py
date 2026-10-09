"""Make ComfyUI account for each rank's local FSDP shard.

ComfyUI uses two size estimates when placing a model. The adapter sets
``ModelPatcher.size`` to local bytes so ``partially_load`` can choose a full
load. This module patches ``comfy.model_management.module_size``, which
``ModelPatcher._load_list`` uses to charge individual modules.

Stock ``module_size`` sums state-dict ``nbytes``. For FSDP2 DTensors, that is
the global size, so the load loop can charge the full checkpoint on every
rank and partially load an otherwise fitting shard. Measurement M6 in
docs/VALIDATION.md records this failure.

Sum ``local_nbytes`` instead. Plain tensors retain their stock byte count,
so the process-wide patch changes only DTensor accounting.
``tests/canary/comfy_seam_contracts.py`` covers this ComfyUI interface.
"""
from __future__ import annotations

from typing import Any, cast

from ..log import get_logger
from ..transfer_utils import failure_summary, safe_call

log = get_logger("dgx_monarch.actor.comfy_bridge")


def install() -> bool:
    """Make comfy's module byte sum shard aware. Idempotent, best effort.

    True when this call installed it. False when it was already installed or
    when comfy is a stand-in without the function, which is not a failure: the
    price is then whatever that stand-in charges.
    """
    try:
        import comfy.model_management as mm

        original = mm.module_size
        if getattr(original, "_dgxm_local_shards", False):
            return False
        from ..adapters.fsdp_quant import local_nbytes

        def module_size(module: Any) -> int:
            return sum(local_nbytes(tensor)
                       for tensor in module.state_dict().values())

        shim = cast(Any, module_size)
        shim._dgxm_local_shards = True
        shim._dgxm_orig = original
        cast(Any, mm).module_size = shim
        log.info("comfy module size accounting: local shard bytes")
        return True
    except Exception as exc:
        safe_call(log.warning, "shard-aware module size not installed: %s",
                  failure_summary(exc))
        return False


def uninstall() -> bool:
    """Put comfy's own byte sum back. For tests and for a stock A/B."""
    try:
        import comfy.model_management as mm

        shim = mm.module_size
        if not getattr(shim, "_dgxm_local_shards", False):
            return False
        cast(Any, mm).module_size = cast(Any, shim)._dgxm_orig
        return True
    except Exception:
        return False
