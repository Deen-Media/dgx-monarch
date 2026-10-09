"""Identity-gate worker swap cycle bound to one artifact snapshot."""
from __future__ import annotations

import socket

from . import store_fsdp


def run(worker, unet_name: str, options: dict | None = None,
        lora_stack: list | None = None,
        expected_artifact_identity: dict | None = None) -> dict:
    """Leave weights swap-produced without ever crossing artifact identities."""
    if worker._setup_key is None:
        raise RuntimeError("gate_swap_cycle before setup(): the Init node must run first")

    from ..mesh_safety import ArtifactBindingError
    from .model_store import request_artifact_identity

    def assert_current() -> None:
        if expected_artifact_identity is None:
            return
        current = request_artifact_identity(unet_name, lora_stack)
        if current != expected_artifact_identity:
            raise ArtifactBindingError(
                "model/LoRA/Comfy identity changed after the ceremony snapshot "
                f"on rank {worker.rank}"
            )

    def ensure(stack) -> str:
        assert_current()
        _, transition = store_fsdp.ensure(
            worker, unet_name, options, stack, slot="cond",
            on_base_loaded=worker._inject_for_topology)
        if expected_artifact_identity is not None:
            resident = worker.store.current
            if (resident is None
                    or resident.artifact_identity != expected_artifact_identity):
                raise ArtifactBindingError(
                    "resident model identity changed after the gate artifact snapshot "
                    f"on rank {worker.rank}"
                )
        return transition

    def residency() -> dict:
        resident = worker.store.current
        slab = getattr(resident, "slab", None)
        certificate = getattr(slab, "certificate", None) if slab is not None else None
        return {
            "family": getattr(resident, "family", None),
            "slab_active": bool(resident and resident.slab is not None),
            # The byte-verify certificate of this rank's slab, in the frozen v9
            # wire shape. The ceremony's capacity branch folds every rank's copy
            # into the CAPACITY_CERTIFIED row; a rank with no slab sends None,
            # the fold returns None, and no row is written.
            "certificate": certificate.public() if certificate is not None else None,
        }

    assert_current()
    host = socket.gethostname()
    # This response is identity-gate evidence as well as diagnostics: the driver
    # must prove it received one current response from every member of the
    # setup cohort, never N copies of one rank's result as an all-rank PASS.
    cohort = {
        "rank": worker.rank,
        "world": worker.world,
        "setup_generation": worker._setup_generation,
    }
    if not lora_stack:
        # The structured half of the reason below: the driver classifies the
        # ceremony from this key, never from the prose
        # (nodes/gate_inconclusive.CYCLE_KEY). A rank that cannot stamp it is
        # read as unproven, the fail-closed side.
        return {"host": host, "conclusive": False, "no_material": True,
                "reason": "no lora stack: lazy swap is not applicable",
                **cohort, **residency()}
    if not worker.store.lora_low_rss:
        return {"host": host, "conclusive": False,
                "reason": "lora_low_rss is off on this worker",
                **cohort, **residency()}
    nudged = [dict(entry) for entry in lora_stack]
    nudged[0]["strength"] = round(float(nudged[0]["strength"]) + 0.05, 6)
    transitions = {
        "load": ensure(lora_stack),
        "nudge": ensure(nudged),
        "back": ensure(lora_stack),
    }
    lazy = transitions["nudge"] == "hot-swap" and transitions["back"] == "hot-swap"
    return {"host": host, "conclusive": lazy, "transitions": transitions,
            **cohort, **residency(),
            **({} if lazy else {"reason": "swaps fell back to reload (no usable "
                                          "un-bake record; see the "
                                          "Cluster Status (DGX Monarch) node or the dgxm top memory panel)"})}
