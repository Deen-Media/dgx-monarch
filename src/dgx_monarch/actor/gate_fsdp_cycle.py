"""Bound FSDP clean-reload proof for the Identity Gate."""
from __future__ import annotations

import socket
from typing import Any

from monarch.actor import concurrent_endpoint

from ..adapters.fsdp import (
    FSDP_ADMITTED_CHECKPOINT_KINDS,
    fsdp_precision_profile_is_admitted,
)
from ..loader_options import weight_dtype_from_options
from . import store_fsdp
from .failure import cleanup_on_failure


class FsdpGateCycleMixin:
    """Actor endpoint surface for the FSDP clean-reload proof."""

    _gpu_lock: Any
    _on_gpu: Any

    @concurrent_endpoint
    async def gate_fsdp_reload_cycle(
        self,
        unet_name: str,
        options: dict | None = None,
        lora_stack: list | None = None,
        expected_artifact_identity: dict | None = None,
    ) -> dict:
        weight_dtype_from_options(options)
        async with self._gpu_lock:
            return await self._on_gpu(
                self._gate_fsdp_reload_cycle_impl,
                unet_name,
                options,
                lora_stack,
                expected_artifact_identity,
            )

    @cleanup_on_failure
    def _gate_fsdp_reload_cycle_impl(
        self,
        unet_name: str,
        options: dict | None = None,
        lora_stack: list | None = None,
        expected_artifact_identity: dict | None = None,
    ) -> dict:
        return run(
            self,
            unet_name,
            options,
            lora_stack,
            expected_artifact_identity,
        )


def run(
    worker,
    unet_name: str,
    options: dict | None = None,
    lora_stack: list | None = None,
    expected_artifact_identity: dict | None = None,
) -> dict:
    """Replace one clean FSDP load with an independently loaded shard set."""
    if worker._setup_key is None:
        raise RuntimeError(
            "gate_fsdp_reload_cycle before setup(): the Init node must run first"
        )
    if not store_fsdp.active(worker):
        raise RuntimeError("FSDP reload proof requires an active FSDP topology")
    if lora_stack:
        raise RuntimeError("FSDP reload proof does not admit a LoRA stack")
    if expected_artifact_identity is None:
        raise RuntimeError("FSDP reload proof requires an artifact identity")

    from ..mesh_safety import ArtifactBindingError
    from .model_store import request_artifact_identity

    def assert_current() -> None:
        if request_artifact_identity(unet_name, None) != expected_artifact_identity:
            raise ArtifactBindingError(
                "model/Comfy identity changed after the FSDP ceremony snapshot "
                f"on rank {worker.rank}"
            )

    assert_current()
    previous = worker.store.current
    previous_family = getattr(previous, "family", None)
    previous_precision = getattr(previous, "precision_evidence", None)
    baseline_artifact_verified = bool(
        previous is not None
        and previous.artifact_identity == expected_artifact_identity
    )
    baseline_slab_active = bool(previous is not None and previous.slab is not None)
    baseline_fsdp_ready = bool(
        previous is not None
        and getattr(previous.base_patcher, "_dgxm_fsdp_ready", False)
    )
    baseline_quant = getattr(previous, "quant_kind", None)
    baseline_profile = getattr(previous_precision, "live_dtype_profile", None)
    baseline_checkpoint = getattr(previous_precision, "checkpoint_kind", None)
    baseline_aux_count = getattr(
        previous_precision, "auxiliary_parameter_count", None
    )
    baseline_aux_bytes = getattr(
        previous_precision, "auxiliary_parameter_bytes", None
    )
    baseline_verified = bool(
        previous is not None
        and baseline_artifact_verified
        and not baseline_slab_active
        and baseline_fsdp_ready
        and baseline_quant in FSDP_ADMITTED_CHECKPOINT_KINDS
        and previous_precision is not None
        and previous_precision.quant_kind == baseline_quant
        and fsdp_precision_profile_is_admitted(
            baseline_profile,
            baseline_aux_count,
            baseline_aux_bytes,
            baseline_checkpoint,
        )
    )
    # Never retain the baseline StoredModel across unload/reload: a local
    # reference would pin the full first shard set and turn this proof into a
    # two-model capacity spike.
    previous = None
    previous_precision = None
    worker.store.unload_all()
    assert_current()
    _, transition = store_fsdp.ensure(
        worker,
        unet_name,
        options,
        None,
        slot="cond",
        on_base_loaded=worker._inject_for_topology,
    )
    assert_current()
    current = worker.store.current
    if (
        current is None
        or current.artifact_identity != expected_artifact_identity
    ):
        raise ArtifactBindingError(
            "resident FSDP model identity changed after the ceremony snapshot "
            f"on rank {worker.rank}"
        )

    precision = current.precision_evidence
    fsdp_ready = bool(getattr(current.base_patcher, "_dgxm_fsdp_ready", False))
    slab_active = current.slab is not None
    conclusive = (
        baseline_verified
        and transition == "load"
        and previous_family is not None
        and current.family == previous_family
        and not slab_active
        and fsdp_ready
        and current.quant_kind in FSDP_ADMITTED_CHECKPOINT_KINDS
        and precision.quant_kind == current.quant_kind
        and fsdp_precision_profile_is_admitted(
            precision.live_dtype_profile,
            precision.auxiliary_parameter_count,
            precision.auxiliary_parameter_bytes,
            precision.checkpoint_kind,
        )
    )
    return {
        "host": socket.gethostname(),
        "rank": worker.rank,
        "setup_generation": worker._setup_generation,
        "conclusive": conclusive,
        "proof": "fsdp_clean_reload",
        "baseline_verified": baseline_verified,
        "baseline_artifact_identity_verified": baseline_artifact_verified,
        "baseline_family": previous_family,
        "baseline_quant": baseline_quant,
        "baseline_live_dtype_profile": baseline_profile,
        "baseline_auxiliary_parameter_count": baseline_aux_count,
        "baseline_auxiliary_parameter_bytes": baseline_aux_bytes,
        "baseline_checkpoint_precision": baseline_checkpoint,
        "baseline_slab_active": baseline_slab_active,
        "baseline_fsdp_ready": baseline_fsdp_ready,
        "transitions": {"reload": transition},
        "family": current.family,
        "quant": current.quant_kind,
        **precision.public(),
        "slab_active": slab_active,
        "fsdp_ready": fsdp_ready,
        "artifact_identity_verified": True,
        **({} if conclusive else {
            "reason": "FSDP proof did not produce one fresh, matching, ready shard set"
        }),
    }
