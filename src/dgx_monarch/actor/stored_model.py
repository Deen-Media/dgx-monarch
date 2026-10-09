"""Owned resources and identity for one resident model-store slot."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..transfer_utils import raise_with_distinct_cause, reconcile_error
from .store_detect import LivePrecisionEvidence


@dataclass
class StoredModel:
    base_key: tuple
    request_key: tuple
    base_patcher: Any
    active_patcher: Any
    quant_kind: str
    precision_evidence: LivePrecisionEvidence
    family: str
    artifact_identity: dict
    lora_sig: tuple = ()
    loaded_at: float = field(default_factory=time.time)
    unbake: Any = None
    slab: Any = None
    slab_auto_retry: bool = False
    fsdp_checkpoint_pin: Any = None
    residency_rung: str = ""
    # The low-RSS bake merged the stack into these weights and freed ComfyUI's
    # backup, so this base cannot be re-cloned for a different stack.
    base_baked: bool = False

    def close(self) -> None:
        """Close every owned resource; operational cancellation wins."""
        failure: BaseException | None = None
        failure_cause: BaseException | None = None
        for label, resource in (
            ("pinned checkpoint close also failed", self.fsdp_checkpoint_pin),
            ("weight slab close also failed", self.slab),
        ):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as exc:
                failure, failure_cause = reconcile_error(failure, exc, label)
        if failure is not None:
            raise_with_distinct_cause(failure, failure_cause)
